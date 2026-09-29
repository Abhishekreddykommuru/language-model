"""Stage 31: finetune one pretrained model on its own reasoning dataset.

Starts from that language's pretrained checkpoint, keeps its tokenizer and
vocabulary, and writes finetuned checkpoints in the same resumable format as
pretraining -- so an interrupted session continues rather than starting over.

Nothing crosses between languages: each run reads only its own checkpoint, its
own tokenizer and its own generated data.

Usage:
    python scripts/31_finetune.py --lang tel
    python scripts/31_finetune.py --lang tel --max-steps 200    # smoke run
    python scripts/31_finetune.py --lang nep --fresh            # ignore state
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, warn
from common.model.config import TrainConfig, save_configs
from common.eval.generate import load_tokenizer
from common.reasoning.data import ReasoningDataset
from common.train.loop import Trainer
from scripts_util import check_tokenizer, load_model, run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--data-dir", default=None,
                    help="defaults to {work}/{language}/9_reasoning")
    ap.add_argument("--config", default=None,
                    help="defaults to {language}/configs/finetune.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="restart finetuning from the pretrained checkpoint")
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = run_name(args.lang, "ft")

    raw = yaml.safe_load(
        Path(args.config or f"{langdir}/configs/finetune.yaml").read_text())
    ft = raw.get("finetune", {})
    tcfg = TrainConfig(**raw.get("train", {}))
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.batch_size:
        tcfg.batch_size = args.batch_size

    ckpt_dir = f"{work}/{langdir}/7_checkpoints"
    pretrained = f"{ckpt_dir}/{args.lang}_{ft.get('init_from', 'best')}.pt"
    if not Path(pretrained).exists():
        raise SystemExit(f"no pretrained checkpoint at {pretrained}; "
                         f"run scripts/20_train.py first")

    model, mcfg, pre = load_model(pretrained, device)
    sp = load_tokenizer(langdir, args.lang)
    check_tokenizer(sp, mcfg, args.lang)

    # Dropout is a property of the modules, so changing it means walking them;
    # the weights and the architecture are otherwise untouched.
    if "dropout" in ft:
        mcfg.dropout = mcfg.attn_dropout = float(ft["dropout"])
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = float(ft["dropout"])

    data_dir = Path(args.data_dir or f"{work}/{langdir}/9_reasoning")
    if not (data_dir / "train.jsonl").exists():
        raise SystemExit(f"no reasoning data at {data_dir}; "
                         f"run scripts/30_make_reasoning.py --lang {args.lang}")
    answer_only = bool(ft.get("answer_only", True))
    splits = {name: ReasoningDataset(str(data_dir / f"{name}.jsonl"), sp,
                                     mcfg.max_seq_len, answer_only, tcfg.seed)
              for name in ("train", "val")}

    torch.manual_seed(tcfg.seed)
    trainer = Trainer(model, splits, mcfg, tcfg, ckpt_dir, device, run)
    trainer.tokens_per_window = splits["train"].tokens / len(splits["train"])
    resumed = False if args.fresh else trainer.resume("last")

    section(f"Reasoning finetuning, {args.lang}")
    kv("starting from", f"{Path(pretrained).name}, step {pre['step']:,}")
    kv("device", f"{device}"
       + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    kv("precision", str(trainer.dtype).replace("torch.", ""))
    kv("parameters", f"{model.parameter_report()['trainable']:,}")
    kv("vocabulary", f"{mcfg.vocab_size:,} pieces, unchanged")
    kv("dropout", mcfg.dropout)
    kv("training examples", f"{len(splits['train']):,}")
    kv("validation examples", f"{len(splits['val']):,}")
    kv("mean example length",
       f"{splits['train'].tokens / len(splits['train']):.1f} tokens")
    kv("loss computed on", "answer tokens only" if answer_only else "all tokens")
    kv("learning rate", f"{tcfg.learning_rate:g} to {tcfg.min_learning_rate:g}")
    kv("steps", f"{tcfg.max_steps:,}")
    kv("examples per step", f"{tcfg.tokens_per_step:,}")
    kv("passes over the training set",
       f"{tcfg.max_steps * tcfg.tokens_per_step / len(splits['train']):.1f}")
    kv("resumed", f"from step {trainer.step:,}" if resumed else "no, fresh start")

    for name, ds in splits.items():
        if ds.dropped_too_long:
            warn(f"{name}: {ds.dropped_too_long} examples longer than "
                 f"{mcfg.max_seq_len} tokens were dropped")
        if ds.boundary_mismatches:
            warn(f"{name}: {ds.boundary_mismatches} examples tokenise across "
                 f"the prompt boundary and are scored on all their tokens")

    save_configs(mcfg, tcfg, f"{ckpt_dir}/config_{run}.yaml")

    t0 = time.time()
    history = trainer.train()
    mins = (time.time() - t0) / 60

    section(f"Finetuning finished, {args.lang}")
    kv("wall clock", f"{mins:.1f} min")
    kv("steps completed", f"{trainer.step:,}")
    if history:
        kv("best validation loss", f"{trainer.best_val:.4f}")
        kv("first validation loss", f"{history[0]['val_loss']:.4f}")
    kv("checkpoints written", f"{run}_best.pt, {run}_last.pt")

    tabs = Path(args.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"finetune_{args.lang}.json").write_text(json.dumps({
        "lang": args.lang,
        "run": run,
        "started_from": {"checkpoint": Path(pretrained).name,
                         "step": pre["step"]},
        "hyperparameters": {**{k: v for k, v in vars(tcfg).items()},
                            "answer_only": answer_only,
                            "dropout": mcfg.dropout},
        "data": {name: {"examples": len(ds), "tokens": ds.tokens,
                        "dropped_too_long": ds.dropped_too_long,
                        "boundary_mismatches": ds.boundary_mismatches}
                 for name, ds in splits.items()},
        "steps": trainer.step,
        "best_val_loss": trainer.best_val,
        "minutes": mins,
        "history": history,
    }, indent=2))
    kv("log", tabs / f"finetune_{args.lang}.json")


if __name__ == "__main__":
    main()
