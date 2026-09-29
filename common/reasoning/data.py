"""Batching for reasoning finetuning.

Pretraining streams fixed-length windows out of one flat token array; finetuning
works on short, variable-length examples that must not run into each other. So
this module exposes the same `batch` / `sequential_batches` interface the
training loop already expects, but each row is one example, padded to the
longest in the batch.

Loss is taken on the answer tokens only by default. The prompt is given, not
predicted, and including it lets a model lower the loss by getting fluent at
restating the question -- which is not what is being finetuned for. Prompt
positions are set to -100, the index `cross_entropy` ignores.

Padding goes on the right and needs no attention mask: attention is causal, so a
pad can only be seen by positions after it, and every target from the padding
onwards is already ignored. Left-padding would not be safe, which is why
generation batches examples by equal prompt length instead.
"""
import json
from pathlib import Path

import numpy as np
import torch

IGNORE = -100


class ReasoningDataset:
    """Tokenised examples from one JSONL split.

    Args:
        path: a split written by scripts/30_make_reasoning.py.
        sp: the language's SentencePiece processor, unchanged from Phase 1.
        max_seq_len: the model's context window; longer examples are dropped.
        answer_only: mask the prompt out of the loss.
    """

    def __init__(self, path: str, sp, max_seq_len: int,
                 answer_only: bool = True, seed: int = 1337):
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.answer_only = answer_only
        self.rng = np.random.default_rng(seed)
        self.pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
        self.bos_id = sp.bos_id() if sp.bos_id() >= 0 else self.pad_id
        self.eos_id = sp.eos_id() if sp.eos_id() >= 0 else self.pad_id

        self.rows = [json.loads(line)
                     for line in Path(path).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.examples = []
        self.boundary_mismatches = 0
        self.dropped_too_long = 0
        for row in self.rows:
            enc = self.encode(row)
            if enc is None:
                self.dropped_too_long += 1
                continue
            self.examples.append(enc)
        if not self.examples:
            raise ValueError(f"{path} produced no usable examples")
        self.seq_len = max(len(e["x"]) for e in self.examples)

    def encode(self, row: dict):
        """One example as (input ids, target ids), or None if it will not fit.

        The prompt boundary is found by tokenising the prompt separately and
        checking it is a prefix of the full encoding. SentencePiece can merge
        across the join, and where it does the check fails and the example
        falls back to loss on every token rather than being silently mislabelled.
        """
        full = self.sp.encode(row["text"])
        prompt = self.sp.encode(row["prompt"])
        n_prompt = len(prompt)
        if full[:n_prompt] != prompt:
            self.boundary_mismatches += 1
            n_prompt = 0

        x = [self.bos_id] + full
        y = full + [self.eos_id]
        if len(x) > self.max_seq_len:
            return None
        if self.answer_only and n_prompt:
            y = [IGNORE] * n_prompt + y[n_prompt:]
        return {"x": x, "y": y, "id": row["id"], "template": row["template"]}

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def tokens(self) -> int:
        return sum(len(e["x"]) for e in self.examples)

    def _collate(self, chosen: list, device: str) -> tuple:
        width = max(len(e["x"]) for e in chosen)
        x = np.full((len(chosen), width), self.pad_id, dtype=np.int64)
        y = np.full((len(chosen), width), IGNORE, dtype=np.int64)
        for i, e in enumerate(chosen):
            x[i, :len(e["x"])] = e["x"]
            y[i, :len(e["y"])] = e["y"]
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))

    def batch(self, batch_size: int, device: str) -> tuple:
        """A random batch, sampled with replacement.

        Sampling rather than sweeping epochs keeps this interchangeable with the
        pretraining data layer, so the same training loop drives both.
        """
        idx = self.rng.integers(0, len(self.examples), size=batch_size)
        return self._collate([self.examples[i] for i in idx], device)

    def sequential_batches(self, batch_size: int, device: str,
                           max_batches: int = None):
        """Every example once, in order, for deterministic evaluation."""
        n = len(self.examples)
        if max_batches is not None:
            n = min(n, max_batches * batch_size)
        for i in range(0, n, batch_size):
            chunk = self.examples[i:i + batch_size]
            if chunk:
                yield self._collate(chunk, device)
