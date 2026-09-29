"""Sharded JSONL+zstd corpus reader and writer, safe for the Drive mount.

Shards are written to local disk, closed, then copied across: an open file on
the FUSE mount does not sync promptly. Shards are bucketed into part000/,
part001/, ... because a Drive folder with thousands of entries raises Errno 5.
"""
import io
import json
import shutil
import time
from pathlib import Path
from typing import Iterator

import zstandard as zstd

SHARD_TARGET_BYTES = 256 * 1024 * 1024      # uncompressed
FILES_PER_SUBDIR = 200


class ShardWriter:
    """Write documents to numbered .jsonl.zst shards, staging locally.

        with ShardWriter(f"{work}/1_raw/tel", "fineweb2") as w:
            w.write(doc.to_dict())
    """

    def __init__(self, out_dir: str, prefix: str, level: int = 3,
                 stage_dir: str = None):
        from common.io.paths import scratch_dir

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stage = Path(stage_dir or f"{scratch_dir()}/staging")
        self.stage.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.level = level
        self.idx = self._resume_index()
        self.written_bytes = 0
        self.n_docs = 0
        self._fh = None
        self._writer = None
        self._local_path = None
        self._open_new()

    def _resume_index(self) -> int:
        """Continue numbering past existing shards, so an interrupted run never
        overwrites completed work."""
        nums = []
        for p in self.out_dir.rglob(f"{self.prefix}-*.jsonl.zst"):
            stem = p.name[len(self.prefix) + 1:].split(".")[0]
            if stem.isdigit():
                nums.append(int(stem))
        return max(nums) + 1 if nums else 0

    def _dest_for(self, idx: int) -> Path:
        sub = self.out_dir / f"part{idx // FILES_PER_SUBDIR:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{self.prefix}-{idx:05d}.jsonl.zst"

    def _open_new(self, advance: bool = False):
        # Advance after shipping, so a shard is filed under the index it was
        # written with.
        self._close_and_ship()
        if advance:
            self.idx += 1
        self._local_path = self.stage / f"{self.prefix}-{self.idx:05d}.jsonl.zst"
        self._fh = open(self._local_path, "wb")
        cctx = zstd.ZstdCompressor(level=self.level, threads=-1)
        self._writer = cctx.stream_writer(self._fh)
        self.written_bytes = 0

    def _close_and_ship(self):
        if self._writer is None:
            return
        self._writer.flush(zstd.FLUSH_FRAME)
        self._writer.close()
        self._fh.close()
        self._writer = self._fh = None

        if self._local_path.stat().st_size == 0:
            self._local_path.unlink(missing_ok=True)
            return

        dest = self._dest_for(self.idx)
        for attempt in range(5):
            try:
                shutil.copyfile(self._local_path, dest)
                if dest.stat().st_size != self._local_path.stat().st_size:
                    raise IOError("size mismatch after copy")
                self._local_path.unlink()
                print(f"    wrote {dest.name} "
                      f"({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
                return
            except Exception as exc:
                wait = 5 * (attempt + 1)
                print(f"    ! Drive copy failed ({exc}); retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        raise IOError(f"could not ship {self._local_path} after 5 attempts")

    def write(self, record: dict):
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        self._writer.write(line)
        self.written_bytes += len(line)
        self.n_docs += 1
        if self.written_bytes >= SHARD_TARGET_BYTES:
            self._open_new(advance=True)

    def close(self):
        self._close_and_ship()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_shard(path: str) -> Iterator[dict]:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_dir(dir_path: str, pattern: str = "**/*.jsonl.zst") -> Iterator[dict]:
    """Yield every document from every shard, including part*/ subfolders.

    An unreadable shard is reported and skipped rather than killing the stage.
    A single truncated file on the Drive mount would otherwise abort a run
    several hours in, and because stages clear their own output first, the
    result is an empty directory that looks like a successful empty corpus.
    """
    root = Path(dir_path)
    if not root.exists():
        return
    for p in sorted(root.glob(pattern)):
        try:
            yield from read_shard(str(p))
        except Exception as exc:
            print(f"  ! unreadable shard skipped: {p} ({type(exc).__name__}: "
                  f"{exc})", flush=True)


def count_dir(dir_path: str) -> tuple:
    """Return (n_docs, n_chars, n_bytes)."""
    n_docs = n_chars = n_bytes = 0
    for d in read_dir(dir_path):
        n_docs += 1
        n_chars += d.get("n_chars") or len(d.get("text", ""))
        n_bytes += d.get("n_bytes") or len(d.get("text", "").encode("utf-8"))
    return n_docs, n_chars, n_bytes


def reset_dir(*dirs):
    """Delete a stage's own output before it is rewritten.

    ShardWriter resumes shard numbering so an interrupted run never overwrites
    completed shards. That is right within a run and wrong across runs, where it
    appends a second copy of everything instead. Collection stages are
    append-only by design and must not call this.
    """
    for d in dirs:
        p = Path(d)
        if p.exists() and any(p.rglob("*.jsonl.zst")):
            n = len(list(p.rglob("*.jsonl.zst")))
            print(f"  clearing {n} shard(s) from {p}", flush=True)
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)


def has_shards(dir_path: str) -> bool:
    """True if a directory holds at least one shard."""
    root = Path(dir_path)
    return root.exists() and any(root.rglob("*.jsonl.zst"))


def require_shards(dir_path: str, stage: str, fix: str):
    """Abort before a stage clears its own output, if its input is missing.

    Order matters: reset_dir deletes the previous run's output, so checking the
    input afterwards would destroy good data and leave an empty directory that
    reads as a successful empty corpus.
    """
    if not has_shards(dir_path):
        raise SystemExit(
            f"{stage} has no input: {dir_path} contains no shards.\n"
            f"  Nothing has been deleted. {fix}")
