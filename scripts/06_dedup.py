"""Stage 4: deduplication and decontamination.

3-word shingles, a k=100 MinHash signature and b=20 bands of r=5, for an LSH
threshold of 0.55. Three passes, so that correctness and a 12.7 GB RAM budget
can both hold: exact deduplication by SHA-256 alongside signature computation,
then band-by-band clustering, then writing survivors.

Every source is read in one pass, so a crawled page that also appears inside a
downloaded corpus is caught; keep_policy prefer_manual makes the crawled copy
the survivor. The final pass also drops training documents duplicating val or
test text, without which Phase 2 perplexity would mean nothing.

Usage:
    python scripts/06_dedup.py --lang tel
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.clean.dedup import Decontaminator, MinHashClusterer, exact_hash
from common.io.paths import ensure_drive_mounted, lang_dir, scratch_dir, work_dir
from common.io.shards import (ShardWriter, read_dir, require_shards,
                              reset_dir)
from common.log import kv, progress, section, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    dcfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/clean.yaml",
                               encoding="utf-8"))["dedup"]
    clean = f"{work}/{lang_dir(args.lang)}/3_clean"
    dst = f"{work}/{lang_dir(args.lang)}/4_dedup"
    require_shards(f"{clean}/train", "Stage 06",
                   f"Run: python scripts/05_clean.py --lang {args.lang} "
                   f"--split train")
    reset_dir(f"{dst}/train", f"{dst}/val", f"{dst}/test")
    counters = Counter()

    section(f"Deduplicating {args.lang}")
    decon = Decontaminator(dcfg)
    for split in ("val", "test"):
        decon.index_heldout(read_dir(f"{clean}/{split}"))
    kv("held-out documents indexed", f"{decon.n_held:,}")

    print("\npass 1/3, exact deduplication and signatures", flush=True)
    clusterer = MinHashClusterer(dcfg, f"{scratch_dir()}/sigs_{args.lang}.npy")
    seen_exact: dict = {}
    for d in progress(read_dir(f"{clean}/train"), "pass 1/3 signatures"):
        counters["in"] += 1
        h = exact_hash(d["text"])
        if h in seen_exact:
            counters["exact_dup"] += 1
            continue
        seen_exact[h] = d["doc_id"]
        clusterer.add(d["doc_id"], d["text"],
                      d.get("source_class", "downloaded"),
                      d.get("source", "?"))
        if counters["in"] % 200_000 == 0:
            print(f"    {counters['in']:,} documents scanned", flush=True)
    del seen_exact
    clusterer.finalize_signatures()
    print(f"  {counters['in']:,} in, {counters['exact_dup']:,} exact duplicates, "
          f"{len(clusterer.doc_ids):,} unique", flush=True)

    print(f"\npass 2/3, clustering near-duplicates across "
          f"{dcfg.get('bands', 20)} bands", flush=True)
    stats = clusterer.cluster()
    keep_ids = clusterer.representatives()
    print(f"  {stats['documents']:,} documents fell into {stats['clusters']:,} "
          f"clusters, {stats['duplicates']:,} near-duplicates", flush=True)

    print("\npass 3/3, writing survivors and decontaminating", flush=True)
    with ShardWriter(f"{dst}/train", "train") as writer:
        for d in progress(read_dir(f"{clean}/train"), "pass 3/3 survivors"):
            if d["doc_id"] not in keep_ids:
                counters["duplicate_dropped"] += 1
                continue
            contaminated, _which = decon.is_contaminated(d["text"])
            if contaminated:
                counters["decontaminated"] += 1
                continue
            d["filters"] = list(d.get("filters", [])) + ["dedup"]
            writer.write(d)
            counters["kept"] += 1

    # --- val and test pass through untouched (they were held out first) ---
    for split in ("val", "test"):
        with ShardWriter(f"{dst}/{split}", split) as writer:
            for d in read_dir(f"{clean}/{split}"):
                writer.write(d)
                counters[f"{split}_passthrough"] += 1

    counters["near_dup_clusters"] = stats["clusters"]
    counters["near_dups_removed"] = stats["duplicates"]
    summary = dict(counters)

    section(f"Deduplication results, {args.lang}")
    kv("documents in", f"{counters['in']:,}")
    kv("exact duplicates removed", f"{counters['exact_dup']:,}")
    kv("near-duplicates removed", f"{counters['duplicate_dropped']:,}")
    kv("removed as contamination", f"{counters['decontaminated']:,}")
    kv("kept", f"{counters['kept']:,}  "
               f"({counters['kept'] / max(counters['in'], 1):.1%})")

    # Who republished whom. The same wire story across dozens of sites is the
    # classic failure mode, and this is where it becomes visible.
    dup_report = clusterer.duplication_report()
    rows = []
    for src, n in sorted(dup_report["dropped_by_source"].items(),
                         key=lambda kv: -kv[1])[:12]:
        total = dup_report["kept_by_source"].get(src, 0) + n
        top = dup_report["duplicate_of_source"].get(src, {})
        rows.append([src, f"{n:,}", f"{total:,}", f"{n / max(total, 1):.1%}",
                     ", ".join(f"{k} x{v:,}" for k, v in
                               sorted(top.items(), key=lambda kv: -kv[1])[:3])])
    if rows:
        print("\nwhere the duplicates came from")
        table(["source", "dropped", "of", "rate", "duplicated"], rows,
              aligns="lrrrl")

    log_dir = Path(f"{work}/_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"dedup_{args.lang}.json", "w") as fh:
        json.dump({**summary, "duplication_by_source": dup_report}, fh, indent=2)


if __name__ == "__main__":
    main()
