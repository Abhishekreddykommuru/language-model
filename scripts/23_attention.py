"""Stage 23: what the attention heads learned.

Heatmaps for an early and a late layer across several heads, plus entropy and
mean attention distance for every head in the model.

Usage:
    python scripts/23_attention.py --lang tel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from common import viz as V
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, table
from common.eval.attention import collect_attention, summarise
from common.eval.generate import load_tokenizer
from common.train.data import load_splits
from scripts_util import load_model, run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--seq-len", type=int, default=48,
                    help="short window, so heatmap axes stay readable")
    ap.add_argument("--n-sequences", type=int, default=16,
                    help="sequences averaged for entropy and distance")
    ap.add_argument("--heads", type=int, default=4, help="heads per heatmap")
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variant", default="",
                    help="evaluate a non-standard run, e.g. ft or nopos")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = run_name(args.lang, args.variant)
    ckpt = f"{work}/{langdir}/7_checkpoints/{run}_{args.checkpoint}.pt"
    if not Path(ckpt).exists():
        raise SystemExit(f"no checkpoint at {ckpt}; run scripts/20_train.py first")

    model, mcfg, ck = load_model(ckpt, device)
    splits = load_splits(work, langdir, args.seq_len)
    held = splits.get("test") or splits["val"]
    sp = load_tokenizer(langdir, args.lang)

    x, _ = held.batch(args.n_sequences, device)
    attentions = collect_attention(model, x)
    stats = summarise(attentions)

    section(f"Attention analysis, {run}")
    kv("checkpoint", f"step {ck['step']:,}")
    kv("layers x heads", f"{mcfg.n_layers} x {mcfg.n_heads}")
    kv("sequences", f"{args.n_sequences} of {args.seq_len} tokens")
    kv("maximum possible entropy", f"{stats['max_entropy']:.3f} nats")

    print("\nper-layer means")
    table(["layer", "entropy (nats)", "mean distance (positions)"],
          [[r["layer"], f"{r['entropy']:.3f}", f"{r['mean_distance']:.2f}"]
           for r in stats["layer_means"]])

    heads = sorted(stats["heads"], key=lambda r: r["mean_distance"])
    print("\nmost local heads")
    table(["layer", "head", "entropy", "mean distance"],
          [[r["layer"], r["head"], f"{r['entropy']:.3f}",
            f"{r['mean_distance']:.2f}"] for r in heads[:5]])
    print("\nmost long-range heads")
    table(["layer", "head", "entropy", "mean distance"],
          [[r["layer"], r["head"], f"{r['entropy']:.3f}",
            f"{r['mean_distance']:.2f}"] for r in heads[-5:]])

    # Heatmaps: one early layer, one late layer, several heads each.
    figs = Path(args.out) / "figures"
    tokens = [sp.id_to_piece(int(t)) for t in x[0].tolist()]
    if V.indic_font(args.lang) is None:
        print("\n  ! no Telugu/Devanagari font installed, so heatmap axes will\n"
              "    be labelled by position instead of by token. On Colab:\n"
              "      apt-get install -y fonts-noto-core fonts-noto-cjk\n"
              "    then re-run this stage.", flush=True)
    show_heads = list(range(min(args.heads, mcfg.n_heads)))
    for label, layer in (("early", 0), ("late", mcfg.n_layers - 1)):
        w = attentions[layer][0].float().cpu().numpy()
        V.plot_attention_heatmap(
            w, run, layer, show_heads, tokens,
            str(figs / f"attention_{label}_{run}.png"))
        kv(f"{label} layer heatmap", figs / f"attention_{label}_{run}.png")

    V.plot_attention_summary(stats["heads"], run,
                             str(figs / f"attn_entropy_{run}.png"),
                             "entropy", "Entropy (nats)")
    V.plot_attention_summary(stats["heads"], run,
                             str(figs / f"attn_distance_{run}.png"),
                             "mean_distance", "Mean attention distance")

    tabs = Path(args.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"attention_{run}.json").write_text(
        json.dumps({"lang": args.lang, "run": run, "step": ck["step"],
                    **stats}, indent=2))
    kv("summary", tabs / f"attention_{run}.json")


if __name__ == "__main__":
    main()
