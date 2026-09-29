"""Stage 21: intrinsic language-model evaluation.

Cross-entropy, perplexity and bits-per-byte on the held-out splits, plus the
loss curves from training. Perplexity is not comparable across the two models
because their tokenizers differ; bits-per-byte is, which is why both are here.

Usage:
    python scripts/21_evaluate.py --lang tel
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
from common.eval.lm import bits_per_byte, cross_entropy_and_tokens, perplexity
from common.train.data import load_splits, split_bytes
from scripts_util import load_model, run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap evaluation batches; default is the whole split")
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
    splits = load_splits(work, langdir, mcfg.max_seq_len)
    rep = model.parameter_report()

    section(f"Evaluating {run}")
    kv("checkpoint", f"{run}_{args.checkpoint}.pt, step {ck['step']:,}")
    kv("parameters", f"{rep['trainable']:,}")
    kv("device", device)

    results = {"lang": args.lang, "run": run, "step": ck["step"],
               "parameters": rep["trainable"], "vocab_size": mcfg.vocab_size,
               "splits": {}}
    rows = []
    for split in ("val", "test"):
        if splits.get(split) is None:
            continue
        ce, n_tokens = cross_entropy_and_tokens(
            model, splits[split], args.batch_size, device, args.max_batches)
        n_bytes = split_bytes(work, langdir, split)
        ppl = perplexity(ce)
        bpb = bits_per_byte(ce, n_tokens, n_bytes)
        results["splits"][split] = {
            "cross_entropy_nats": ce, "perplexity": ppl,
            "bits_per_byte": bpb, "tokens_scored": n_tokens,
            "bytes_of_text": n_bytes,
        }
        rows.append([split, f"{ce:.4f}", f"{ppl:.2f}", f"{bpb:.4f}",
                     f"{n_tokens:,}"])

    print()
    table(["split", "cross-entropy", "perplexity", "bits/byte", "tokens"], rows)

    # Curves from the training history stored in the checkpoint.
    history = ck.get("history", [])
    figs = Path(args.out) / "figures"
    if history:
        V.plot_loss_curves(history, run,
                           str(figs / f"loss_{run}.png"))
        V.plot_lr_schedule(history, run,
                           str(figs / f"lr_{run}.png"))
        results["history"] = history
        print()
        kv("loss curve", figs / f"loss_{run}.png")

    tabs = Path(args.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"lm_eval_{run}.json").write_text(
        json.dumps(results, indent=2))
    kv("metrics", tabs / f"lm_eval_{run}.json")


if __name__ == "__main__":
    main()
