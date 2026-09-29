"""Print a spread of rejected documents, grouped by rejection reason.

The output is meant to be read, not skimmed: a threshold tuned on English can
quietly empty a low-resource corpus, and the only way to notice is to look at
what it deleted. Where a rejection looks wrong, adjust the threshold in that
language's clean.yaml and re-run 05_clean.py.

Usage:
    python scripts/audit_filters.py --lang nep --n 20
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import read_dir
from common.log import section, table

SAMPLES_PER_REASON = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="report/tables")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    rejects = f"{work}/{lang_dir(args.lang)}/3_clean/{args.split}_rejected"

    rng = random.Random(7)
    counts = Counter()
    by_reason: dict = {}

    for d in read_dir(rejects):
        reason = d.get("reject_reason", "unknown")
        counts[reason] += 1
        bucket = by_reason.setdefault(reason, [])
        # Reservoir sampling so we see a spread, not just the first documents.
        if len(bucket) < SAMPLES_PER_REASON:
            bucket.append(d)
        elif rng.random() < 0.01:
            bucket[rng.randrange(SAMPLES_PER_REASON)] = d

    if not counts:
        print(f"no rejected documents found at {rejects}")
        return

    total = sum(counts.values())
    section(f"Rejections by reason, {args.lang} {args.split}")
    table(["reason", "documents", "share"],
          [[r, f"{c:,}", f"{c / total:.1%}"] for r, c in counts.most_common()])

    section(f"{args.n} rejected documents, sampled across reasons")
    shown = 0
    for reason, _c in counts.most_common():
        for d in by_reason.get(reason, []):
            if shown >= args.n:
                break
            print(f"\n[{reason}]  {d['doc_id']}  from {d.get('source', '?')}")
            if d.get("reject_detail"):
                print(f"  {d['reject_detail']}")
            print(f"  {d.get('text', '')[:600]}".replace("\n", " "))
            shown += 1
        if shown >= args.n:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(counts), open(out_dir / f"rejections_{args.lang}.json", "w"),
              indent=2)

    cfg = f"{lang_dir(args.lang)}/configs/clean.yaml"
    print(f"\nFor each document above, decide whether you agree with the "
          f"filter.\nWhere you do not, adjust the threshold in {cfg} and re-run "
          f"scripts/05_clean.py.\nThe decisions belong in the report.")


if __name__ == "__main__":
    main()
