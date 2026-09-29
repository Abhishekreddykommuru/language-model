"""Stage 34, optional bonus: retrain one language without positional embeddings.

Trains the ablated model, runs the full Phase 2 evaluation suite over it --
perplexity and bits-per-byte, BLEU / chrF++ / ROUGE-L, diversity, attention
heatmaps, entropy and distance -- and assembles the comparison against the
standard model for that language.

Everything else is held fixed: same corpus, same tokenizer, same architecture,
same seed, same schedule. The single difference is `use_positional_embeddings`.

The evaluation stages are the Phase 2 ones, reached through `--variant nopos`,
so the ablated model is measured by exactly the same code as the standard model
rather than by a second implementation.

Usage:
    python scripts/34_ablation.py --lang tel
    python scripts/34_ablation.py --lang tel --skip-train   # just compare
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from common import viz as V
from common.io.paths import lang_dir, work_dir
from common.log import kv, section, table, warn

GEN_METRICS = [("bleu", "BLEU-4"), ("chrf_pp", "chrF++"),
               ("rouge_l_f", "ROUGE-L"), ("distinct_1", "distinct-1"),
               ("distinct_2", "distinct-2"),
               ("repetition_4gram", "4-gram repetition")]


def run(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    print(f"  {'finished' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc == 0


def matched_step(history: list, step: int):
    """The standard model's validation loss at the ablation's step count.

    If the ablation was given a shorter budget than the standard run, comparing
    final numbers would confuse the ablation with the shorter schedule. The
    pretraining history records validation loss every few hundred steps, so the
    standard model can be read at the same point.
    """
    if not history:
        return None
    nearest = min(history, key=lambda h: abs(h["step"] - step))
    return nearest if abs(nearest["step"] - step) <= 1000 else None


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="tel", choices=["tel", "nep"],
                    help="the language to ablate; the report uses Telugu")
    ap.add_argument("--work", default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="shorter budget; the comparison is then made at the "
                         "matching step of the standard run")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    py = sys.executable
    tabs = Path(args.out) / "tables"

    section(f"Bonus ablation, {args.lang} without positional embeddings")
    kv("config", f"{langdir}/configs/model_nopos.yaml")
    kv("checkpoints", f"{work}/{langdir}/7_checkpoints/{args.lang}_nopos_*.pt")

    common = ["--lang", args.lang, "--variant", "nopos"]
    if args.work:
        common += ["--work", args.work]
    if args.device:
        common += ["--device", args.device]

    stages = []
    if not args.skip_train:
        train = [py, "scripts/20_train.py", *common,
                 "--model-config", f"{langdir}/configs/model_nopos.yaml"]
        if args.max_steps:
            train += ["--max-steps", args.max_steps]
        stages.append(("20", train))
    stages += [
        ("21", [py, "scripts/21_evaluate.py", *common, "--out", args.out]),
        ("22", [py, "scripts/22_generate.py", *common, "--out", args.out]),
        ("23", [py, "scripts/23_attention.py", *common, "--out", args.out]),
    ]
    for sid, cmd in stages:
        section(f"{sid}  [{args.lang} nopos]")
        if not run(cmd):
            raise SystemExit(f"stage {sid} failed; later stages depend on it")

    # ---------------- comparison ----------------

    std_lm = load(tabs / f"lm_eval_{args.lang}.json")
    abl_lm = load(tabs / f"lm_eval_{args.lang}_nopos.json")
    std_gen = load(tabs / f"generation_{args.lang}.json")
    abl_gen = load(tabs / f"generation_{args.lang}_nopos.json")
    std_att = load(tabs / f"attention_{args.lang}.json")
    abl_att = load(tabs / f"attention_{args.lang}_nopos.json")
    if not all((std_lm, abl_lm, std_gen, abl_gen, std_att, abl_att)):
        raise SystemExit("missing result files; run Phase 2 for the standard "
                         "model before comparing against the ablation")

    section("Standard against ablated")
    kv("standard parameters", f"{std_lm['parameters']:,}")
    kv("ablated parameters",
       f"{abl_lm['parameters']:,} "
       f"({std_lm['parameters'] - abl_lm['parameters']:,} fewer)")
    kv("standard steps", f"{std_lm['step']:,}")
    kv("ablated steps", f"{abl_lm['step']:,}")

    matched = matched_step(std_lm.get("history", []), abl_lm["step"])
    if abl_lm["step"] < std_lm["step"] - 1000:
        if matched:
            warn(f"the ablation ran {abl_lm['step']:,} steps against "
                 f"{std_lm['step']:,}; at step {matched['step']:,} the standard "
                 f"model was at validation loss {matched['val_loss']:.4f}")
        else:
            warn("the two runs used different budgets and no matching point "
                 "was recorded, so the comparison below is not like for like")

    print("\nlanguage modelling, test split")
    table(["metric", "standard", "no positions", "change"],
          [[name,
            f"{std_lm['splits']['test'][key]:.4f}",
            f"{abl_lm['splits']['test'][key]:.4f}",
            f"{abl_lm['splits']['test'][key] - std_lm['splits']['test'][key]:+.4f}"]
           for key, name in (("cross_entropy_nats", "cross-entropy (nats)"),
                             ("perplexity", "perplexity"),
                             ("bits_per_byte", "bits per byte"))])

    print("\ngeneration, greedy decoding")
    table(["metric", "standard", "no positions", "change"],
          [[name, f"{std_gen['settings']['greedy'][key]:.4f}",
            f"{abl_gen['settings']['greedy'][key]:.4f}",
            f"{abl_gen['settings']['greedy'][key] - std_gen['settings']['greedy'][key]:+.4f}"]
           for key, name in GEN_METRICS])

    def head_means(stats):
        heads = stats["heads"]
        return (sum(h["entropy"] for h in heads) / len(heads),
                sum(h["mean_distance"] for h in heads) / len(heads))

    std_ent, std_dist = head_means(std_att)
    abl_ent, abl_dist = head_means(abl_att)
    print("\nattention, averaged over every head")
    table(["metric", "standard", "no positions", "change"],
          [["entropy (nats)", f"{std_ent:.3f}", f"{abl_ent:.3f}",
            f"{abl_ent - std_ent:+.3f}"],
           ["mean distance (positions)", f"{std_dist:.2f}", f"{abl_dist:.2f}",
            f"{abl_dist - std_dist:+.2f}"],
           ["maximum possible entropy", f"{std_att['max_entropy']:.3f}",
            f"{abl_att['max_entropy']:.3f}", "-"]])

    figs = Path(args.out) / "figures"
    V.plot_metric_comparison(
        [name for _, name in GEN_METRICS],
        {"standard": [std_gen["settings"]["greedy"][k] for k, _ in GEN_METRICS],
         "no positions": [abl_gen["settings"]["greedy"][k] for k, _ in GEN_METRICS]},
        f"Generation metrics with and without positional embeddings - {args.lang}",
        "Value", str(figs / f"ablation_generation_{args.lang}.png"))
    V.plot_attention_shift(
        std_att["layer_means"], abl_att["layer_means"], args.lang,
        str(figs / f"ablation_entropy_{args.lang}.png"),
        "entropy", "Entropy (nats)")
    V.plot_attention_shift(
        std_att["layer_means"], abl_att["layer_means"], args.lang,
        str(figs / f"ablation_distance_{args.lang}.png"),
        "mean_distance", "Mean attention distance")

    result = {
        "lang": args.lang,
        "standard": {"parameters": std_lm["parameters"], "step": std_lm["step"],
                     "lm": std_lm["splits"], "generation": std_gen["settings"],
                     "attention": {"entropy": std_ent, "distance": std_dist,
                                   "layer_means": std_att["layer_means"]}},
        "ablated": {"parameters": abl_lm["parameters"], "step": abl_lm["step"],
                    "lm": abl_lm["splits"], "generation": abl_gen["settings"],
                    "attention": {"entropy": abl_ent, "distance": abl_dist,
                                  "layer_means": abl_att["layer_means"]}},
        "matched_step_of_standard_run": matched,
    }
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"ablation_{args.lang}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print()
    kv("comparison", tabs / f"ablation_{args.lang}.json")


if __name__ == "__main__":
    main()
