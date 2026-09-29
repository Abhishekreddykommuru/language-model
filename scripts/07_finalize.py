"""Stage 5: blocklists, source-mix capping, and corpus sizing.

Two requirements pull against each other: ~500M tokens per language, and at
least 20% of final training tokens from manual collection. Capping downloaded
text at 4x manual satisfies the second and can wreck the first, since 47M manual
tokens pin the corpus at 235M however much clean downloaded text is available.

The size target wins, so the cap is

    cap = max(manual_floor_cap, target_tokens - manual_tokens)

which enforces the 20% floor only while doing so still reaches the target, then
keeps whatever downloaded text the target needs and reports the resulting manual
fraction as it falls. Manual data is never downsampled. Pass --target-tokens 0
to enforce the floor strictly instead and accept a smaller corpus.

Usage:
    python scripts/07_finalize.py --lang tel
    python scripts/07_finalize.py --lang tel --target-tokens 500e6
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from common.collect.hf_download import BYTES_PER_TOKEN
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import (ShardWriter, read_dir, require_shards,
                              reset_dir)
from common.log import kv, millions, progress, section, table
from common.stats import downloaded_cap_for_manual

#: Deliberately short: an over-broad blocklist silently deletes legitimate text,
#: which a low-resource corpus cannot afford.
BLOCK_SUBSTRINGS = ["lorem ipsum", "javascript is disabled", "404 not found",
                    "page not found", "enable javascript"]

SEED = 1337


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--min-manual-fraction", type=float, default=0.20)
    ap.add_argument("--target-tokens", type=float, default=500e6,
                    help="corpus size to aim for. The manual-fraction cap is "
                         "relaxed as far as needed to reach this. Set to 0 to "
                         "enforce the 20%% floor strictly instead.")
    ap.add_argument("--max-source-share", type=float, default=0.60,
                    help="cap any single downloaded source's share of the "
                         "corpus; 1.0 disables it. Manual sources are never "
                         "capped, being scarce and already diverse.")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    src = f"{work}/{lang_dir(args.lang)}/4_dedup"
    dst = f"{work}/{lang_dir(args.lang)}/5_final"
    bpt = BYTES_PER_TOKEN[args.lang]
    counters = Counter()

    require_shards(f"{src}/train", "Stage 07",
                   f"Run: python scripts/06_dedup.py --lang {args.lang}")

    # Pass 1: measure what survived, by class and by source.
    bytes_by_class = defaultdict(int)
    bytes_by_source = defaultdict(int)
    manual_sources = set()
    for d in progress(read_dir(f"{src}/train"), "pass 1/2 measuring"):
        n_bytes = d.get("n_bytes", 0)
        cls = d.get("source_class", "downloaded")
        bytes_by_class[cls] += n_bytes
        bytes_by_source[d.get("source", "?")] += n_bytes
        if cls == "manual":
            manual_sources.add(d.get("source", "?"))

    manual_tokens = int(bytes_by_class.get("manual", 0) / bpt)
    downloaded_tokens = int(bytes_by_class.get("downloaded", 0) / bpt)
    target = int(args.target_tokens)

    section(f"Sizing the {args.lang} corpus")
    kv("manual available", millions(manual_tokens))
    kv("downloaded available", millions(downloaded_tokens))
    kv("target", millions(target) if target else "none, floor enforced strictly")

    # With no manual data the floor cap is zero, which would discard the whole
    # downloaded corpus and emit nothing. A corpus failing the 20% rule is
    # recoverable; an empty one is not.
    if manual_tokens == 0:
        print()
        print("  ! No manual data for this language, so the 20% requirement")
        print("    cannot be met. Keeping all downloaded text rather than")
        print("    emitting an empty corpus. To fix this, run:")
        print(f"      python scripts/run_all.py --only 02,02b --lang {args.lang}")
        print(f"      python scripts/run_all.py --only 03,03b --lang {args.lang}")
        floor_cap = downloaded_tokens
    else:
        floor_cap = downloaded_cap_for_manual(manual_tokens,
                                              args.min_manual_fraction)

    # Relax the floor cap by exactly what reaching the target requires, no more.
    # Where manual is plentiful the floor cap is already the larger of the two.
    cap = max(floor_cap, max(0, target - manual_tokens))
    if cap > floor_cap:
        print()
        kv("strict floor would allow", f"{millions(manual_tokens + floor_cap)}"
                                       f", short of target")
        kv("cap relaxed to", f"{millions(cap)} downloaded tokens")

    if downloaded_tokens > cap:
        keep_prob = cap / downloaded_tokens
        kv("downloaded downsampled to", f"{keep_prob:.1%}, to meet the cap")
    else:
        keep_prob = 1.0
        kv("downloaded kept", "all of it, the cap does not bind")
        if manual_tokens + downloaded_tokens < target:
            kv("shortfall", f"{millions(target - manual_tokens - downloaded_tokens)}"
                            f" below target even keeping everything")

    # Diversity cap, downloaded sources only: manual sources are scarce and
    # already diverse. Computed against the downloaded total so that trimming
    # for the floor and trimming for diversity do not delete the same documents
    # twice.
    dl_bytes = {s: nb for s, nb in bytes_by_source.items()
                if s not in manual_sources}
    dl_total = sum(dl_bytes.values()) or 1
    source_keep = {}
    # One source has nothing to diversify against, so capping its share would
    # delete good text for no benefit.
    if len(dl_bytes) < 2:
        kv("diversity cap", "skipped, only one downloaded source")
        dl_bytes = {}
    for source, nb in dl_bytes.items():
        share = nb / dl_total
        source_keep[source] = (args.max_source_share / share) \
            if share > args.max_source_share else 1.0
    if dl_bytes:
        print("\ndownloaded source shares")
        table(["source", "share", "action"],
              [[src, f"{nb / dl_total:.1%}",
                "downsampled" if source_keep[src] < 1 else "kept in full"]
               for src, nb in sorted(dl_bytes.items(), key=lambda kv: -kv[1])],
              aligns="lrl")

    reset_dir(f"{dst}/train", f"{dst}/val", f"{dst}/test")

    # Pass 2: write the final corpus.
    rng = random.Random(SEED)
    kept_by_class = defaultdict(int)
    for split in ("train", "val", "test"):
        with ShardWriter(f"{dst}/{split}", split) as writer:
            for d in progress(read_dir(f"{src}/{split}"), f"writing {split}"):
                head = d.get("text", "")[:500].lower()
                if any(b in head for b in BLOCK_SUBSTRINGS):
                    counters["blocklist"] += 1
                    continue
                if split == "train":
                    cls = d.get("source_class", "downloaded")
                    if cls == "downloaded" and rng.random() > keep_prob:
                        counters["downsampled_for_manual_floor"] += 1
                        continue
                    if rng.random() > source_keep.get(d.get("source", "?"), 1.0):
                        counters["downsampled_source_cap"] += 1
                        continue
                    kept_by_class[cls] += d.get("n_bytes", 0)
                writer.write(d)
                counters[f"{split}_kept"] += 1

    final_manual = int(kept_by_class.get("manual", 0) / bpt)
    final_downloaded = int(kept_by_class.get("downloaded", 0) / bpt)
    final_total = final_manual + final_downloaded
    fraction = final_manual / final_total if final_total else 0.0

    summary = {
        "estimated_manual_tokens": final_manual,
        "estimated_downloaded_tokens": final_downloaded,
        "estimated_total_tokens": final_total,
        "estimated_manual_fraction": round(fraction, 4),
        "target_tokens": target,
        "counters": dict(counters),
    }
    section(f"Final {args.lang} corpus")
    table(["", "tokens", "share"],
          [["manual", millions(final_manual), f"{fraction:.1%}"],
           ["downloaded", millions(final_downloaded), f"{1 - fraction:.1%}"],
           ["total", millions(final_total), ""]])
    print()
    kv("size vs target", f"{millions(final_total)} of {millions(target)}"
                         f"  {'met' if final_total >= target else 'short'}")
    kv("manual vs floor", f"{fraction:.1%} against {args.min_manual_fraction:.0%}"
                          f"  {'met' if fraction >= args.min_manual_fraction else 'below'}")
    kv("note", "these are byte estimates; 09_tokenize.py counts for real")

    log_dir = Path(f"{work}/_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"finalize_{args.lang}.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
