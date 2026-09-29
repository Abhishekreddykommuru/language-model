"""Stage 4, reduced: exact-only deduplication and decontamination.

06_dedup.py removes near-duplicates with MinHash + LSH, which costs several
hours on a corpus of this size. This is the cheap alternative: it removes
byte-identical documents and any training document whose exact text also
appears in the held-out splits, in a single streaming pass.

Near-duplicate removal is NOT applied. The log records that explicitly, so the
report can state exactly what was and was not done rather than implying a full
deduplication.

Usage:
    python scripts/06b_dedup_exact.py --lang nep
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from common.clean.dedup import exact_hash
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import (ShardWriter, read_dir, require_shards, reset_dir)
from common.log import kv, progress, section


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    clean = f"{work}/{lang_dir(args.lang)}/3_clean"
    dst = f"{work}/{lang_dir(args.lang)}/4_dedup"

    require_shards(f"{clean}/train", "Stage 06b",
                   f"Run: python scripts/05_clean.py --lang {args.lang} "
                   f"--split train")
    reset_dir(f"{dst}/train", f"{dst}/val", f"{dst}/test")
    counters = Counter()

    section(f"Exact deduplication, {args.lang}")
    kv("near-duplicate removal", "NOT applied, see the log")

    # Held-out hashes first, so training documents that repeat evaluation text
    # can be dropped in the same pass.
    held = set()
    for split in ("val", "test"):
        for d in read_dir(f"{clean}/{split}"):
            held.add(exact_hash(d["text"]))
    kv("held-out documents indexed", f"{len(held):,}")

    seen = set()
    with ShardWriter(f"{dst}/train", "train") as writer:
        for d in progress(read_dir(f"{clean}/train"), f"dedup {args.lang}"):
            counters["in"] += 1
            h = exact_hash(d["text"])
            if h in held:
                counters["decontaminated"] += 1
                continue
            if h in seen:
                counters["exact_dup"] += 1
                continue
            seen.add(h)
            d["filters"] = list(d.get("filters", [])) + ["dedup:exact_only"]
            writer.write(d)
            counters["kept"] += 1
            if counters["in"] % 200_000 == 0:
                print(f"    {counters['in']:,} scanned, {counters['kept']:,} kept",
                      flush=True)

    for split in ("val", "test"):
        with ShardWriter(f"{dst}/{split}", split) as writer:
            for d in read_dir(f"{clean}/{split}"):
                writer.write(d)
                counters[f"{split}_passthrough"] += 1

    summary = {
        **dict(counters),
        "method": "exact_only",
        "near_duplicates_removed": 0,
        "note": ("MinHash near-duplicate removal was not applied to this "
                 "language: the full pass did not fit the compute budget. "
                 "Exact duplicates and held-out contamination were removed."),
    }

    section(f"Exact deduplication results, {args.lang}")
    kv("documents in", f"{counters['in']:,}")
    kv("exact duplicates removed", f"{counters['exact_dup']:,}")
    kv("removed as contamination", f"{counters['decontaminated']:,}")
    kv("kept", f"{counters['kept']:,}  "
               f"({counters['kept'] / max(counters['in'], 1):.1%})")

    log_dir = Path(f"{work}/_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"dedup_{args.lang}.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
