"""Stage 33: attention before and after finetuning.

Runs the Phase 2 attention toolkit twice over the same comparative-reasoning
prompts -- once on the pretrained checkpoint, once on the finetuned one -- and
reports what moved. Heatmaps are drawn for an early and a late layer of each
model, and entropy and mean attention distance are tabulated per layer and per
head with the difference between the two.

Using reasoning prompts rather than corpus text is the point: the question is
whether finetuning changed how the model reads a comparison, and that only
shows on the inputs the finetuning was about.

Usage:
    python scripts/33_attention_compare.py --lang tel
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from common import viz as V
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, table
from common.eval.attention import collect_attention, summarise
from common.eval.generate import load_tokenizer
from common.reasoning.generate import read_split
from scripts_util import check_tokenizer, load_model, run_name

MODELS = [("pretrained", ""), ("finetuned", "ft")]


def prompt_batch(rows: list, sp, n_sequences: int, device: str):
    """A batch of equal-length reasoning prompts, and their token strings.

    The model has no padding mask, so every sequence in the batch must be the
    same length or the padded positions would sit inside the context the real
    positions attend over. The most populated length bucket is used, which also
    keeps the heatmap axes a sensible width.
    """
    buckets = {}
    for row in rows:
        ids = sp.encode(row["prompt"])
        buckets.setdefault(len(ids), []).append((row, ids))
    length = max(buckets, key=lambda k: len(buckets[k]))
    chosen = buckets[length][:n_sequences]
    idx = torch.tensor([ids for _, ids in chosen], dtype=torch.long,
                       device=device)
    tokens = [sp.id_to_piece(int(t)) for t in chosen[0][1]]
    return idx, tokens, [row for row, _ in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-sequences", type=int, default=16)
    ap.add_argument("--heads", type=int, default=4, help="heads per heatmap")
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir or f"{work}/{langdir}/9_reasoning")
    rows = read_split(str(data_dir / f"{args.split}.jsonl"))
    sp = load_tokenizer(langdir, args.lang)
    idx, tokens, used = prompt_batch(rows, sp, args.n_sequences, device)

    section(f"Attention, pretrained against finetuned, {args.lang}")
    kv("prompts", f"{idx.size(0)} of {idx.size(1)} tokens, from {args.split}")
    kv("families in the batch",
       ", ".join(sorted({r["template"] for r in used})))

    figs = Path(args.out) / "figures"
    if V.indic_font(args.lang) is None:
        print("\n  ! no Telugu/Devanagari font installed, so heatmap axes will\n"
              "    be labelled by position instead of by token. On Colab:\n"
              "      apt-get install -y fonts-noto-core fonts-noto-cjk\n"
              "    then re-run this stage.", flush=True)

    stats = {}
    for label, variant in MODELS:
        run = run_name(args.lang, variant)
        ckpt = f"{work}/{langdir}/7_checkpoints/{run}_{args.checkpoint}.pt"
        if not Path(ckpt).exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        model, mcfg, ck = load_model(ckpt, device)
        check_tokenizer(sp, mcfg, args.lang)
        attentions = collect_attention(model, idx)
        stats[label] = {**summarise(attentions), "step": ck["step"],
                        "checkpoint": Path(ckpt).name}

        show_heads = list(range(min(args.heads, mcfg.n_heads)))
        for where, layer in (("early", 0), ("late", mcfg.n_layers - 1)):
            w = attentions[layer][0].float().cpu().numpy()
            path = figs / f"reasoning_attention_{where}_{args.lang}_{label}.png"
            V.plot_attention_heatmap(w, f"{args.lang} {label}", layer,
                                     show_heads, tokens, str(path))
            kv(f"{label}, {where} layer", path)
        del model, attentions
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    pre, ft = stats["pretrained"], stats["finetuned"]
    kv("maximum possible entropy", f"{pre['max_entropy']:.3f} nats")

    print("\nper-layer means, pretrained -> finetuned")
    table(["layer", "entropy", "entropy change", "distance", "distance change"],
          [[p["layer"], f"{p['entropy']:.3f}",
            f"{f_['entropy'] - p['entropy']:+.3f}",
            f"{p['mean_distance']:.2f}",
            f"{f_['mean_distance'] - p['mean_distance']:+.2f}"]
           for p, f_ in zip(pre["layer_means"], ft["layer_means"])])

    # Which individual heads moved most. Head specialisation showing up as a
    # handful of large shifts against a flat majority is the interesting case;
    # a uniform drift across all heads is not.
    moved = sorted(
        [{"layer": a["layer"], "head": a["head"],
          "entropy_change": b["entropy"] - a["entropy"],
          "distance_change": b["mean_distance"] - a["mean_distance"]}
         for a, b in zip(pre["heads"], ft["heads"])],
        key=lambda r: abs(r["entropy_change"]), reverse=True)

    print("\nheads whose entropy moved most")
    table(["layer", "head", "entropy change", "distance change"],
          [[r["layer"], r["head"], f"{r['entropy_change']:+.3f}",
            f"{r['distance_change']:+.2f}"] for r in moved[:8]])

    n_heads = len(moved)
    summary = {
        "mean_entropy_pretrained": sum(h["entropy"] for h in pre["heads"]) / n_heads,
        "mean_entropy_finetuned": sum(h["entropy"] for h in ft["heads"]) / n_heads,
        "mean_distance_pretrained": sum(h["mean_distance"] for h in pre["heads"]) / n_heads,
        "mean_distance_finetuned": sum(h["mean_distance"] for h in ft["heads"]) / n_heads,
        "heads_more_focused": sum(1 for r in moved if r["entropy_change"] < 0),
        "heads_more_diffuse": sum(1 for r in moved if r["entropy_change"] > 0),
        "heads_more_local": sum(1 for r in moved if r["distance_change"] < 0),
        "heads_more_long_range": sum(1 for r in moved if r["distance_change"] > 0),
    }
    print("\nacross all heads")
    kv("mean entropy", f"{summary['mean_entropy_pretrained']:.3f} -> "
                       f"{summary['mean_entropy_finetuned']:.3f} nats")
    kv("mean attention distance",
       f"{summary['mean_distance_pretrained']:.2f} -> "
       f"{summary['mean_distance_finetuned']:.2f} positions")
    kv("heads that sharpened", f"{summary['heads_more_focused']} of {n_heads}")
    kv("heads that became more local",
       f"{summary['heads_more_local']} of {n_heads}")

    V.plot_attention_shift(pre["layer_means"], ft["layer_means"], args.lang,
                           str(figs / f"attention_shift_entropy_{args.lang}.png"),
                           "entropy", "Entropy (nats)")
    V.plot_attention_shift(pre["layer_means"], ft["layer_means"], args.lang,
                           str(figs / f"attention_shift_distance_{args.lang}.png"),
                           "mean_distance", "Mean attention distance")

    tabs = Path(args.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"attention_compare_{args.lang}.json").write_text(json.dumps({
        "lang": args.lang,
        "split": args.split,
        "prompts": {"n": idx.size(0), "seq_len": idx.size(1),
                    "families": sorted({r["template"] for r in used})},
        "summary": summary,
        "pretrained": pre,
        "finetuned": ft,
        "head_changes": moved,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    kv("comparison", tabs / f"attention_compare_{args.lang}.json")


if __name__ == "__main__":
    main()
