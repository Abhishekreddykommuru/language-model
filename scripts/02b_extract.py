"""Stage 1: HTML to article text, with two-pass boilerplate removal.

Pass 1 learns which lines repeat across a site: navigation, "Follow us on",
related-article rails. Pass 2 strips them and writes documents carrying their
provenance. Done per host, because every site has its own furniture.

Usage:
    python scripts/02b_extract.py --lang tel
"""
import argparse
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from common.collect.extract import BoilerplateDetector, extract_article
from common.io.paths import (ensure_drive_mounted, lang_dir, scratch_dir,
                             work_dir)
from common.io.schema import Doc, make_doc_id
from common.io.shards import ShardWriter
from common.log import kv, section

MIN_WORDS = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--html-dir", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    html_dir = Path(args.html_dir or f"{scratch_dir()}/raw_html/{args.lang}")
    out_root = f"{work}/{lang_dir(args.lang)}/1_raw/manual/web_crawl"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 02_crawl.py clears local disk as it archives, so scratch is normally
    # empty here, especially in a fresh session. Restore from Drive first.
    archive_dir = Path(f"{work}/_raw_html/{args.lang}")
    tars = sorted(archive_dir.glob("*.tar")) if archive_dir.exists() else []
    if tars:
        # Restoring is latency-bound: each archive costs a Drive round trip
        # whatever its size. A thread pool over tarfile avoids the process spawn
        # per archive, and Drive serves parallel reads happily.
        import tarfile
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # No total-size line here on purpose: stat()-ing every archive is
        # another round trip each, and with tens of thousands of them that is
        # minutes of silence before any output appears.
        section(f"Extracting articles, {args.lang}")
        kv("archives to restore", f"{len(tars):,}, 16 at a time")
        html_dir.mkdir(parents=True, exist_ok=True)

        def restore_one(tar_path: Path) -> int:
            # Each archive holds one host's batch, named html-<host>-NNNN-stamp.tar
            parts = tar_path.name.split("-")
            host = parts[1] if len(parts) > 2 else "unknown"
            dest = html_dir / host
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tar_path, "r") as tf:
                    tf.extractall(dest, filter="data")
                return 1
            except Exception:
                return 0

        done = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(restore_one, t): t for t in tars}
            for fut in as_completed(futures):
                ok = fut.result()
                done += ok
                failed += (1 - ok)
                if done % 2000 == 0 and done:
                    rate = done / max(time.time() - t0, 1)
                    eta = (len(tars) - done) / max(rate, 0.1) / 60
                    print(f"  restored {done:,} of {len(tars):,} at {rate:.0f}/s, "
                          f"about {eta:.0f} min left", flush=True)
        print(f"  restored {done:,} archives in {(time.time() - t0) / 60:.1f} min"
              + (f", {failed:,} unreadable" if failed else ""), flush=True)

    if not html_dir.exists() or not any(html_dir.iterdir()):
        print(f"  no crawled HTML for {args.lang} in {html_dir} or "
              f"{archive_dir}; run scripts/02_crawl.py --lang {args.lang} first")
        return

    total = 0
    for host_dir in sorted(p for p in html_dir.iterdir() if p.is_dir()):
        files = sorted(host_dir.glob("*.html.gz"))
        print(f"{host_dir.name}: {len(files):,} pages", flush=True)
        if not files:
            continue

        # --- pass 1: learn this site's boilerplate ---
        detector = BoilerplateDetector(doc_freq=0.001)
        extracted = {}
        for f in files:
            try:
                with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as fh:
                    blob = fh.read()
            except OSError:
                continue
            url, _, html = blob.partition("\n")
            art = extract_article(html, url)
            if art.get("text"):
                extracted[f.name] = (url, art["text"])
                detector.count(art["text"])
        boilerplate = detector.boilerplate_lines()
        print(f"  {len(extracted):,} articles extracted, "
              f"{len(boilerplate)} boilerplate lines identified", flush=True)

        # --- pass 2: strip boilerplate and write shards ---
        n_host = 0
        with ShardWriter(f"{out_root}/{host_dir.name}", "crawl") as writer:
            for url, text in extracted.values():
                clean = "\n".join(l for l in text.split("\n")
                                  if l.strip() not in boilerplate)
                if len(clean.split()) < MIN_WORDS:
                    continue
                doc = Doc(
                    doc_id=make_doc_id(args.lang, "manual", "crawl", total),
                    lang=args.lang, text=clean,
                    source_class="manual", method="self_crawl",
                    source=host_dir.name,
                    licence="site terms; text used for academic research only",
                    url=url, fetched_at=now,
                    extractor="trafilatura+boilerplate_v1",
                ).finalize()
                writer.write(doc.to_dict())
                total += 1
                n_host += 1
        print(f"  {n_host:,} documents written, {total:,} so far", flush=True)

    print()
    kv("crawled documents written", f"{total:,}")
    kv("location", out_root)


if __name__ == "__main__":
    main()
