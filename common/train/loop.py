"""Training loop: AdamW, cosine schedule with warmup, resumable checkpoints.

Checkpoints carry model weights, optimizer state, scheduler position, the step
counter and the full config, which is what makes a run survive a Colab session
ending mid-training.
"""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then cosine decay to `min_learning_rate`.

    Warmup exists because AdamW's second-moment estimate is badly scaled for the
    first few hundred steps; starting at the full rate there is the usual cause
    of an early loss spike.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate
                                             - cfg.min_learning_rate)


def build_optimizer(model, cfg):
    """AdamW with decay on matrices only.

    Weight decay on LayerNorm gains, biases and embeddings pulls parameters that
    have no business shrinking towards zero; the standard split is to decay
    tensors of 2 or more dimensions and leave the rest alone.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate,
                             betas=(cfg.beta1, cfg.beta2), eps=1e-8)


def resolve_precision(requested: str, device: str) -> torch.dtype:
    """bf16 where supported, fp16 on older GPUs, fp32 on CPU."""
    if device == "cpu":
        return torch.float32
    if requested == "fp32":
        return torch.float32
    if requested == "bf16" or (requested == "auto"
                               and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, data, batch_size: int, device: str,
             max_batches: int) -> float:
    """Mean cross-entropy over deterministic windows of a split."""
    model.eval()
    total, n = 0.0, 0
    for x, y in data.sequential_batches(batch_size, device, max_batches):
        _, loss, _ = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


class Trainer:
    """Owns the loop, the checkpoints and the history."""

    def __init__(self, model, splits, model_cfg, train_cfg, ckpt_dir: str,
                 device: str, lang: str):
        self.model = model
        self.splits = splits
        self.mcfg = model_cfg
        self.cfg = train_cfg
        self.device = device
        self.lang = lang
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_optimizer(model, train_cfg)
        self.dtype = resolve_precision(train_cfg.precision, device)
        self.scaler = torch.amp.GradScaler(
            enabled=(self.dtype == torch.float16))
        self.step = 0
        self.best_val = float("inf")
        self.history = []
        # Pretraining windows are all max_seq_len long. Finetuning examples are
        # shorter and variable, so the token counter is told the real width
        # rather than assuming a full window.
        self.tokens_per_window = model_cfg.max_seq_len

    # ---------------- checkpointing ----------------

    def save(self, tag: str = "last"):
        """Write a resumable checkpoint.

        Written to a temporary file and renamed, so a session dying mid-write
        leaves the previous checkpoint intact rather than a truncated one.
        """
        payload = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val": self.best_val,
            "history": self.history,
            "model_config": asdict(self.mcfg),
            "train_config": asdict(self.cfg),
            "lang": self.lang,
        }
        dst = self.ckpt_dir / f"{self.lang}_{tag}.pt"
        tmp = dst.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(dst)
        (self.ckpt_dir / f"history_{self.lang}.json").write_text(
            json.dumps(self.history, indent=2))

    def resume(self, tag: str = "last") -> bool:
        """Restore weights, optimizer, scaler, step and history."""
        path = self.ckpt_dir / f"{self.lang}_{tag}.pt"
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])
        self.step = ck["step"]
        self.best_val = ck.get("best_val", float("inf"))
        self.history = ck.get("history", [])
        return True

    # ---------------- the loop ----------------

    def train(self):
        cfg = self.cfg
        model = self.model
        model.train()
        t0 = time.time()
        tokens_seen = self.step * cfg.tokens_per_step * self.tokens_per_window

        while self.step < cfg.max_steps:
            lr = lr_at(self.step, cfg)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            running = 0.0
            # Gradient accumulation: an effective batch of
            # batch_size x grad_accum_steps without the memory of one.
            for _ in range(cfg.grad_accum_steps):
                x, y = self.splits["train"].batch(cfg.batch_size, self.device)
                with torch.autocast(device_type=self.device.split(":")[0],
                                    dtype=self.dtype,
                                    enabled=(self.dtype != torch.float32)):
                    _, loss, _ = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                running += loss.item()

            # Unscale before clipping, or the threshold applies to scaled
            # gradients and means nothing.
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            tokens_seen += cfg.tokens_per_step * self.tokens_per_window

            if self.step % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"  step {self.step:>6}/{cfg.max_steps}  "
                      f"loss {running:.4f}  lr {lr:.2e}  "
                      f"grad {grad_norm:.2f}  "
                      f"{tokens_seen / 1e6:.1f}M tokens  "
                      f"{elapsed / 60:.1f} min", flush=True)

            if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                val = evaluate(model, self.splits["val"], cfg.batch_size,
                               self.device, cfg.eval_batches)
                ppl = math.exp(min(val, 20))
                self.history.append({"step": self.step, "train_loss": running,
                                     "val_loss": val, "val_ppl": ppl, "lr": lr,
                                     "tokens": tokens_seen})
                print(f"  step {self.step:>6}  val loss {val:.4f}  "
                      f"val ppl {ppl:.2f}", flush=True)
                if val < self.best_val:
                    self.best_val = val
                    self.save("best")
                    print(f"         new best, checkpoint written", flush=True)

            if self.step % cfg.checkpoint_every == 0:
                self.save("last")

        self.save("last")
        return self.history
