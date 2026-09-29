"""Stage 1b: carve out validation and test before any cleaning.

Cleaning before splitting lets the pipeline leak information across the
boundary, so the split comes first and 06_dedup.py then decontaminates train
against the held-out data. Assignment is deterministic in the doc_id order and
proportional across sources, which keeps validation representative and makes
per-source perplexity possible in Phase 2.

Usage:
    python scripts/04_split.py --lang tel
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import ShardWriter, read_dir, reset_dir
from common.log import kv, section, table

SEED = 1337


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    raw = f"{work}/{lang_dir(args.lang)}/1_raw"
    out = f"{work}/{lang_dir(args.lang)}/2_splits"
    reset_dir(f"{out}/train", f"{out}/val", f"{out}/test")

    rng = random.Random(SEED)
    writers = {s: ShardWriter(f"{out}/{s}", s) for s in ("train", "val", "test")}
    counts = Counter()
    per_source = defaultdict(Counter)

    manifest_path = Path(out) / "split_manifest.tsv"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        manifest.write("doc_id\tsplit\tsource\tsource_class\n")
        for d in read_dir(raw):
            r = rng.random()
            if r < args.test_frac:
                split = "test"
            elif r < args.test_frac + args.val_frac:
                split = "val"
            else:
                split = "train"
            writers[split].write(d)
            manifest.write(f"{d['doc_id']}\t{split}\t{d.get('source', '?')}"
                           f"\t{d.get('source_class', '?')}\n")
            counts[split] += 1
            per_source[d.get("source", "?")][split] += 1

    for w in writers.values():
        w.close()

    section(f"Splits held out before cleaning, {args.lang}")
    table(["source", "train", "val", "test", "total"],
          [[src, f"{c['train']:,}", f"{c['val']:,}", f"{c['test']:,}",
            f"{sum(c.values()):,}"] for src, c in sorted(per_source.items())]
          + [["all sources", f"{counts['train']:,}", f"{counts['val']:,}",
              f"{counts['test']:,}", f"{sum(counts.values()):,}"]])
    print()
    kv("manifest", manifest_path)


if __name__ == "__main__":
    main()
