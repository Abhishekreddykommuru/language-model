"""Generate every table and figure for the Phase 1 report.

Computation lives in common/stats.py, plotting in common/viz.py, and every
figure carries a title, axis labels and a legend.

Usage:
    python scripts/10_stats.py --lang tel
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from common import stats as S
from common import viz as V
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import count_dir
from common.log import kv, section, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    figs = Path(args.out) / "figures"
    tabs = Path(args.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    token_report = json.load(
        open(f"{work}/{lang_dir(args.lang)}/6_tokenized/token_report.json",
             encoding="utf-8"))
    sweep = json.load(
        open(f"{work}/{lang_dir(args.lang)}/tokenizer/sweep_results.json",
             encoding="utf-8"))

    section(f"Tables and figures, {args.lang}")

    # 1. Manual versus downloaded tokens.
    mf = S.manual_fraction(token_report["train"]["tokens_by_class"])
    json.dump(mf, open(tabs / f"manual_fraction_{args.lang}.json", "w"), indent=2)
    V.plot_manual_split(mf, args.lang, str(figs / f"manual_split_{args.lang}.png"))
    kv("manual share of tokens", f"{mf['manual_fraction']:.2%}")

    # 2. Fertility against vocabulary size, with the Phase 2 parameter cost.
    grouped = {}
    for r in sweep:
        grouped.setdefault(r["algorithm"], []).append(
            (r["vocab_size"], r["chars_per_token"], r["phase2"]["total"]))
    for algo in grouped:
        grouped[algo].sort()
    V.plot_fertility_vs_vocab(grouped, args.lang,
                              str(figs / f"fertility_{args.lang}.png"))
    json.dump(sweep, open(tabs / f"tokenizer_sweep_{args.lang}.json", "w"), indent=2)

    # 3. Attrition across the pipeline stages.
    stages = []
    for name, folder in (("raw", "1_raw"), ("clean", "3_clean"),
                         ("dedup", "4_dedup"), ("final", "5_final")):
        path = f"{work}/{lang_dir(args.lang)}/{folder}"
        sub = path if name == "raw" else f"{path}/train"
        try:
            n_docs, n_chars, n_bytes = count_dir(sub)
            stages.append({"stage": name, "docs": n_docs, "chars": n_chars,
                           "bytes": n_bytes})
        except Exception as exc:
            print(f"  stage {name} skipped, {exc}")
    if stages:
        V.plot_attrition(stages, args.lang, str(figs / f"attrition_{args.lang}.png"))
        rows = S.attrition_table(stages)
        json.dump(rows, open(tabs / f"attrition_{args.lang}.json", "w"), indent=2)
        print()
        table(["stage", "documents", "of raw", "of previous"],
              [[r["stage"], f"{r['docs']:,}", f"{r['pct_of_raw']:.1%}",
                f"{r['pct_of_prev']:.1%}"] for r in rows])

    # 4. Corpus composition and document lengths.
    cs = S.corpus_stats(f"{work}/{lang_dir(args.lang)}/5_final/train")
    V.plot_doc_lengths(cs["doc_lengths"], args.lang,
                       str(figs / f"doclen_{args.lang}.png"))
    V.plot_source_mix(cs["by_source"], args.lang,
                      str(figs / f"sources_{args.lang}.png"))
    json.dump({k: v for k, v in cs.items() if k != "doc_lengths"},
              open(tabs / f"corpus_{args.lang}.json", "w"), indent=2)

    # 5. Token-frequency statistics, including the Zipf curve.
    tf = token_report.get("token_frequency")
    if tf:
        counts = tf["counts_sorted"]
        V.plot_zipf(list(range(1, len(counts) + 1)), counts, args.lang,
                    str(figs / f"zipf_{args.lang}.png"))
        json.dump({k: v for k, v in tf.items() if k != "counts_sorted"},
                  open(tabs / f"token_frequency_{args.lang}.json", "w"),
                  indent=2, ensure_ascii=False)
        total = token_report["train"]["tokens"]
        kv("vocabulary actually used", f"{tf['distinct_tokens_used']:,} pieces, "
                                       f"{tf['vocab_coverage']:.1%}")
        print()
        table(["rank", "piece", "count", "share"],
              [[i, r["piece"], f"{r['count']:,}", f"{r['count'] / total:.2%}"]
               for i, r in enumerate(tf["top_50"][:15], 1)], aligns="rlrr")

    # 6. Script composition, as evidence the corpus is monolingual.
    json.dump(S.script_composition(f"{work}/{lang_dir(args.lang)}/5_final/train"),
              open(tabs / f"script_{args.lang}.json", "w"), indent=2)

    # 7. Split sizes, with the byte counts Phase 2 needs for bits-per-byte.
    splits = {s: token_report[s] for s in ("train", "val", "test")
              if s in token_report}
    json.dump(splits, open(tabs / f"splits_{args.lang}.json", "w"), indent=2)

    print()
    kv("figures", figs)
    kv("tables", tabs)


if __name__ == "__main__":
    main()
