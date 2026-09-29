"""Stage 30: build the synthetic reasoning dataset for one language.

Writes train/val/test as JSONL plus a dataset card recording sizes, template
variety, answer balance and the leakage checks. The generator is seeded, so
re-running this reproduces the dataset byte for byte.

Usage:
    python scripts/30_make_reasoning.py --lang tel
    python scripts/30_make_reasoning.py --lang nep --out-dir some/other/place
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, table, warn
from common.reasoning.generate import (SPLITS, Generator, dataset_card,
                                       leakage_report, write_split)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="defaults to {work}/{language}/9_reasoning")
    ap.add_argument("--report", default="report")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    out = Path(args.out_dir or f"{work}/{langdir}/9_reasoning")

    gen = Generator(f"{langdir}/configs/reasoning.yaml")

    section(f"Reasoning dataset, {args.lang}")
    kv("config", f"{langdir}/configs/reasoning.yaml")
    kv("seed", gen.cfg.get("seed"))
    kv("output", out)

    splits, rows = {}, []
    for split in SPLITS:
        examples = gen.build(split)
        if gen.last_shortfall:
            for tid, missing in gen.last_shortfall.items():
                warn(f"{split}/{tid}: {missing} short of the requested count, "
                     f"the entity pool cannot produce more distinct prompts")
        write_split(examples, str(out / f"{split}.jsonl"))
        splits[split] = [asdict(e) for e in examples]
        rows.append([split, len(examples),
                     len({r["template"] for r in splits[split]}),
                     len({e for r in splits[split] for e in r["entities"]}),
                     f"{sum(len(r['text']) for r in splits[split]) / max(len(examples), 1):.0f}"])

    print()
    table(["split", "examples", "families", "entities", "mean chars"], rows)

    leakage = leakage_report(splits, gen.cfg)
    card = dataset_card(splits, gen.cfg, leakage)
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nper-family counts")
    families = sorted({r["template"] for s in splits.values() for r in s})
    table(["family"] + list(SPLITS),
          [[f] + [card["splits"][s]["by_template"].get(f, 0) for s in SPLITS]
           for f in families])

    print("\nleakage checks")
    ok = True
    for pair, shared in leakage["entity_overlap"].items():
        ok &= not shared
        kv(f"shared entities, {pair}", len(shared) or "none")
    for pair, n in leakage["prompt_overlap_counts"].items():
        ok &= n == 0
        kv(f"repeated prompts, {pair}", n or "none")
    kv("families held out of training",
       ", ".join(leakage["held_out_families"]) or "none")
    if not ok:
        raise SystemExit("leakage check failed; the splits are not disjoint")

    print("\nmajority-answer baseline, test split")
    kv("always answer the most common string",
       f"{card['splits']['test']['majority_answer_share']:.1%}")

    print("\nsample, one per family")
    for fam in families:
        ex = next(r for r in splits["test"] if r["template"] == fam)
        print(f"\n  [{fam}]  hops={ex['hops']}  attribute={ex['attribute']}")
        print(f"  {ex['text']}")

    tabs = Path(args.report) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"reasoning_dataset_{args.lang}.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    kv("dataset card", tabs / f"reasoning_dataset_{args.lang}.json")


if __name__ == "__main__":
    main()
