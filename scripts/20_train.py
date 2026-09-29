"""Stage 20: pretrain one model.

Resumable by default: if a checkpoint exists it continues from the recorded
step, which is what makes this survive a Colab session ending. The two models
are trained independently and neither is initialised from the other.

Usage:
    python scripts/20_train.py --lang tel
    python scripts/20_train.py --lang tel --max-steps 2000    # short run
    python scripts/20_train.py --lang tel --fresh             # ignore checkpoints
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, millions, section
from common.model.config import load_configs, save_configs
from common.model.transformer import DecoderLM
from common.train.data import load_splits
from common.train.loop import Trainer
from scripts_util import run_name, vocab_size_of


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="start over, ignoring any checkpoint")
    # Overrides for fitting a smaller GPU, or for a fast smoke run. The YAML
    # stays the record of the real run; these are for when it does not fit.
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-config", default=None,
                    help="defaults to {language}/configs/model.yaml")
    ap.add_argument("--variant", default="",
                    help="suffix for a non-standard run, e.g. nopos")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = run_name(args.lang, args.variant)
    V = vocab_size_of(langdir, args.lang)
    mcfg, tcfg = load_configs(
        args.model_config or f"{langdir}/configs/model.yaml", vocab_size=V)
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.grad_accum:
        tcfg.grad_accum_steps = args.grad_accum
    if args.eval_every:
        tcfg.eval_every = args.eval_every
    if args.seq_len:
        mcfg.max_seq_len = args.seq_len

    torch.manual_seed(tcfg.seed)
    splits = load_splits(work, langdir, mcfg.max_seq_len, tcfg.seed)
    model = DecoderLM(mcfg).to(device)
    rep = model.parameter_report()

    ckpt_dir = f"{work}/{langdir}/7_checkpoints"
    trainer = Trainer(model, splits, mcfg, tcfg, ckpt_dir, device, run)
    resumed = False if args.fresh else trainer.resume("last")

    section(f"Pretraining {run}")
    kv("device", f"{device}"
       + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    kv("precision", str(trainer.dtype).replace("torch.", ""))
    kv("parameters", f"{rep['trainable']:,}")
    kv("positional embeddings",
       "learned absolute" if mcfg.use_positional_embeddings else "none (ablation)")
    kv("vocabulary", f"{V:,}")
    kv("training tokens", f"{len(splits['train']):,}")
    kv("validation tokens", f"{len(splits['val']):,}" if splits["val"] else "none")
    kv("tokens per step", f"{tcfg.tokens_per_step * mcfg.max_seq_len:,}")
    kv("steps", f"{tcfg.max_steps:,}")
    kv("total tokens to see",
       millions(tcfg.max_steps * tcfg.tokens_per_step * mcfg.max_seq_len))
    kv("checkpoints", ckpt_dir)
    kv("resumed", f"from step {trainer.step:,}" if resumed else "no, fresh start")

    save_configs(mcfg, tcfg, f"{ckpt_dir}/config_{run}.yaml")

    t0 = time.time()
    history = trainer.train()
    mins = (time.time() - t0) / 60

    section(f"Training finished, {run}")
    kv("wall clock", f"{mins:.1f} min")
    kv("steps completed", f"{trainer.step:,}")
    if history:
        kv("best validation loss", f"{trainer.best_val:.4f}")
        kv("best validation perplexity", f"{__import__('math').exp(min(trainer.best_val, 20)):.2f}")
    kv("checkpoints written", f"{run}_best.pt, {run}_last.pt")


if __name__ == "__main__":
    main()
