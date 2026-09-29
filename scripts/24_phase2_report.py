"""Stage 24: assemble the Phase 2 report from the measured results.

Reads what stages 21-23 wrote and produces report/phase2_report.md, with both
models side by side. Nothing here computes a metric; it only arranges numbers
that already exist, so the report cannot drift from the results.

Usage:
    python scripts/24_phase2_report.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section

LANGS = [("tel", "Telugu", "Model H"), ("nep", "Nepali", "Model L")]


def load(path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def copy_checkpoint_configs(work: str) -> list:
    """Bring the config saved beside each checkpoint into the repo."""
    out = []
    for code, _, _ in LANGS:
        src = Path(f"{work}/{lang_dir(code)}/7_checkpoints/config_{code}.yaml")
        if src.exists():
            dst = Path(f"{lang_dir(code)}/configs/model_trained.yaml")
            shutil.copyfile(src, dst)
            out.append(str(dst))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default="report/phase2_report.md")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing report instead of writing "
                         "alongside it")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    tabs = Path("report/tables")

    lm = {c: load(tabs / f"lm_eval_{c}.json") for c, _, _ in LANGS}
    gen = {c: load(tabs / f"generation_{c}.json") for c, _, _ in LANGS}
    att = {c: load(tabs / f"attention_{c}.json") for c, _, _ in LANGS}
    samples = {c: load(tabs / f"samples_{c}.json") for c, _, _ in LANGS}

    d = ["# Phase 2 — Model Implementation, Pretraining and Evaluation", "",
         "Two decoder-only transformers, written from primitive PyTorch layers "
         "and trained independently on the Phase 1 corpora. Neither model was "
         "initialised from the other; they share no weights, no vocabulary and "
         "no data.", "",
         "## Architecture", "",
         "The forward pass, in order: token embedding plus learned absolute "
         "positional embedding, dropout, N pre-norm transformer blocks, a final "
         "LayerNorm, and a linear projection to the vocabulary tied to the input "
         "embedding. Each block is multi-head causal self-attention then a "
         "position-wise feed-forward network, each wrapped in a residual "
         "connection.", "",
         "Attention is implemented in `common/model/attention.py` without "
         "`nn.MultiheadAttention` or `F.scaled_dot_product_attention`: the "
         "projections, the head reshape, the scaled dot product, the causal mask "
         "and the output projection are all explicit.", "",
         "| choice | what | why |", "|---|---|---|",
         "| Positional encoding | learned absolute | Simplest to explain end to "
         "end. It fixes the context at `max_seq_len`: there is no embedding row "
         "beyond that, so the window cannot be extended after training without "
         "adding and training new rows. Sinusoidal or rotary encodings would "
         "extrapolate. |",
         "| Normalisation | pre-norm | LayerNorm sits inside the residual branch, "
         "leaving an unbroken identity path from first layer to last, so "
         "gradients reach early layers without passing through a normalisation "
         "at every step. Post-norm is implemented too and needs a more careful "
         "warmup. |",
         "| Output head | tied to input embedding | Saves `vocab_size x d_model` "
         "parameters and ties a token's input representation to its representation "
         "as a prediction target. |",
         "| Attention scaling | `1/sqrt(d_head)` | `q·k` sums `d_head` products of "
         "unit-variance terms, so its variance grows with `d_head`. Without the "
         "scaling, softmax saturates and gradients vanish. |",
         "| Causal mask | additive `-inf` upper triangle | Built here rather than "
         "imported, and verified empirically: see below. |", "",
         "### Parameter counts", "",
         "| | Telugu (H) | Nepali (L) |", "|---|---:|---:|"]

    def row(label, fn, fmt="{}"):
        vals = []
        for c, _, _ in LANGS:
            try:
                vals.append(fmt.format(fn(c)))
            except Exception:
                vals.append("—")
        return f"| {label} | " + " | ".join(vals) + " |"

    d.append(row("Trainable parameters", lambda c: lm[c]["parameters"], "{:,}"))
    d.append(row("Vocabulary", lambda c: lm[c]["vocab_size"], "{:,}"))
    d += ["", "Both models share `d_model`, head count and feed-forward width; "
          "only depth differs, so that each lands near the ~25M budget despite "
          "very different vocabulary sizes. A larger vocabulary costs more in the "
          "embedding and leaves less for transformer blocks. Keeping width and "
          "head count fixed leaves one architectural variable between the two "
          "models rather than three, at the cost of the embedding taking a much "
          "larger share of the model with the larger vocabulary.", "",
          "### Causal masking, verified", "",
          "`scripts/19_sanity.py` alters the token at position t+1 and compares "
          "logits. Positions up to t are unchanged to floating-point precision, "
          "while positions from t+1 do change, which confirms both that no "
          "position sees ahead and that the perturbation was real. The same script "
          "checks that the loss before any training equals `ln(vocab_size)`, "
          "9.68 for Telugu and 10.37 for Nepali, which is what an uninformed "
          "near-uniform softmax must give.", ""]

    # ---------------- language modelling ----------------
    d += ["## Language-modelling results", ""]
    if any(lm[c].get("splits") for c, _, _ in LANGS):
        d += ["| metric | split | Telugu (H) | Nepali (L) |",
              "|---|---|---:|---:|"]
        for split in ("val", "test"):
            for key, label, fmt in (("cross_entropy_nats", "Cross-entropy (nats)", "{:.4f}"),
                                    ("perplexity", "Perplexity", "{:.2f}"),
                                    ("bits_per_byte", "Bits per byte", "{:.4f}")):
                vals = []
                for c, _, _ in LANGS:
                    try:
                        vals.append(fmt.format(lm[c]["splits"][split][key]))
                    except Exception:
                        vals.append("—")
                d.append(f"| {label} | {split} | " + " | ".join(vals) + " |")
        d += ["", "Perplexity is **not** comparable between the two models: their "
              "tokenizers differ, so a token means something different in each, and "
              "a model with a coarser tokenizer is rewarded for predicting fewer, "
              "longer units. Bits-per-byte normalises by the underlying UTF-8 text "
              "and is the number to compare.", "",
              "![Telugu loss](figures/loss_tel.png)", "",
              "![Nepali loss](figures/loss_nep.png)", "",
              "_Discussion of the H-versus-L gap: to be written against the numbers "
              "above, covering corpus size, the manual/downloaded mix, tokenizer "
              "fertility from Phase 1 (1.810 tokens per word for Telugu against "
              "1.310 for Nepali), and the depth difference._", ""]
    else:
        d += ["_Stage 21 has not run yet._", ""]

    # ---------------- generation ----------------
    d += ["## Generation quality", ""]
    for c, name, tier in LANGS:
        g = gen.get(c, {}).get("settings")
        if not g:
            continue
        d += [f"### {name} ({tier})", "",
              "| decoding | BLEU-4 | chrF++ | ROUGE-L | Distinct-1 | Distinct-2 | "
              "4-gram repetition |", "|---|---:|---:|---:|---:|---:|---:|"]
        for setting, m in g.items():
            d.append(f"| {setting} | {m.get('bleu', float('nan')):.2f} | "
                     f"{m.get('chrf_pp', float('nan')):.2f} | "
                     f"{m.get('rouge_l_f', 0):.4f} | {m.get('distinct_1', 0):.3f} | "
                     f"{m.get('distinct_2', 0):.3f} | "
                     f"{m.get('repetition_4gram', 0):.3f} |")
        d.append("")
        s = samples.get(c, {}).get("greedy", [])
        if s:
            d += ["Sample, greedy decoding:", "", "```",
                  f"prompt    : {s[0]['prompt'][:200]}",
                  f"generated : {s[0]['generated'][:200]}",
                  f"reference : {s[0]['reference'][:200]}", "```", ""]

    d += ["### Why these metrics say little here", "",
          "BLEU, chrF++ and ROUGE-L all compare a generated continuation against "
          "*one* reference continuation. Language modelling is open-ended: a "
          "fluent, entirely reasonable continuation that happens to go somewhere "
          "else scores near zero. The absolute values are therefore not a measure "
          "of generation quality.", "",
          "What the numbers do support is comparison **across decoding settings "
          "for one model**. chrF++ is the most informative of the three for these "
          "languages because it works on character n-grams: Telugu and Nepali are "
          "agglutinative enough that one wrong suffix destroys a word-level match "
          "while leaving most characters intact, which BLEU and ROUGE-L both "
          "penalise fully.", "",
          "The diversity statistics are more useful. Repetition rate exposes the "
          "degenerate looping that greedy decoding produces in small models and "
          "that perplexity cannot see, and Distinct-1/2 show the vocabulary "
          "collapse that comes with low temperature.", ""]

    # ---------------- attention ----------------
    d += ["## Attention analysis", ""]
    for c, name, tier in LANGS:
        a = att.get(c)
        if not a:
            continue
        d += [f"### {name} ({tier})", "",
              f"{a['seq_len']}-token windows; maximum attainable entropy at this "
              f"length is {a['max_entropy']:.3f} nats.", "",
              "| layer | mean entropy (nats) | mean attention distance |",
              "|---:|---:|---:|"]
        for r in a["layer_means"]:
            d.append(f"| {r['layer']} | {r['entropy']:.3f} | "
                     f"{r['mean_distance']:.2f} |")
        heads = sorted(a["heads"], key=lambda r: r["mean_distance"])
        d += ["", f"Most local head: layer {heads[0]['layer']} head "
              f"{heads[0]['head']}, mean distance {heads[0]['mean_distance']:.2f} "
              f"positions, entropy {heads[0]['entropy']:.3f}. "
              f"Most long-range: layer {heads[-1]['layer']} head "
              f"{heads[-1]['head']}, mean distance "
              f"{heads[-1]['mean_distance']:.2f}, entropy "
              f"{heads[-1]['entropy']:.3f}.", "",
              f"![{name} early layer](figures/attention_early_{c}.png)", "",
              f"![{name} late layer](figures/attention_late_{c}.png)", "",
              f"![{name} entropy](figures/attn_entropy_{c}.png)", "",
              f"![{name} distance](figures/attn_distance_{c}.png)", ""]

    d += ["_Discussion to write against the figures: which heads are local or "
          "positional (low entropy, mean distance near 1, a visible diagonal), "
          "which are content-based (higher entropy, longer distance, structure "
          "that follows the sentence rather than the position), and how the two "
          "models differ given Model L's shallower stack._", "",
          "## Resource-level comparison", "",
          "_To write: how corpus size, manual fraction, tokenizer fertility and "
          "the architectural differences together explain the gap between the two "
          "models._", "",
          "## Reproducing", "", "```bash",
          "python scripts/19_sanity.py --lang tel      # parameters, ln(V) loss, causal mask",
          "python scripts/20_train.py  --lang tel      # resumable pretraining",
          "python scripts/21_evaluate.py --lang tel    # loss, perplexity, bits-per-byte",
          "python scripts/22_generate.py --lang tel    # BLEU, chrF++, ROUGE-L, diversity",
          "python scripts/23_attention.py --lang tel   # heatmaps, entropy, distance",
          "python scripts/24_phase2_report.py          # this file",
          "```", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        # The generated skeleton is a starting point. Once it has been written
        # up by hand, regenerating would silently discard that work.
        alt = out.with_name(out.stem + "_generated.md")
        alt.write_text("\n".join(d), encoding="utf-8")
        print(f"\n  {out} exists and was left alone.\n"
              f"  The freshly generated version is at {alt}.\n"
              f"  Pass --force to overwrite the report itself.")
        return
    out.write_text("\n".join(d), encoding="utf-8")

    section("Phase 2 report")
    for c in copy_checkpoint_configs(work):
        kv("config copied", c)
    kv("report", args.out)
    print("\n  Left to write by hand: the H-vs-L discussion, the attention\n"
          "  interpretation, and the resource-level comparison. Every number\n"
          "  they refer to is already in the tables above.")


if __name__ == "__main__":
    main()
