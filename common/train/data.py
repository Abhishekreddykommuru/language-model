"""Batching over the uint16 token arrays written by Phase 1 stage 09.

`train.bin` is a flat array of token ids and `train.idx` holds document start
offsets. The array is memory-mapped rather than loaded: a 500M-token corpus is
1 GB as uint16, which fits in RAM, but memory-mapping keeps startup instant and
leaves room for the model.
"""
from pathlib import Path

import numpy as np
import torch


class TokenArray:
    """A memory-mapped split, sampled as fixed-length windows."""

    def __init__(self, path: str, seq_len: int, seed: int = 1337):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Phase 1 stage 09 writes it; check the Drive "
                f"path or re-run scripts/09_tokenize.py.")
        self.tokens = np.memmap(p, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"{p} holds only {len(self.tokens)} tokens, too "
                             f"few for a {seq_len}-token window")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(self, batch_size: int, device: str) -> tuple:
        """A random batch of windows.

        Windows are sampled uniformly at random rather than swept in order, so
        every step sees a fresh mix of documents and the loss curve is not an
        artefact of corpus ordering. Inputs and targets are the same window
        offset by one; the shift into next-token prediction happens in the loss.
        """
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1,
                                   size=batch_size)
        x = np.stack([self.tokens[s:s + self.seq_len] for s in starts])
        y = np.stack([self.tokens[s + 1:s + 1 + self.seq_len] for s in starts])
        # int64 because cross_entropy wants long targets; uint16 has no torch
        # equivalent so the copy is unavoidable.
        xt = torch.from_numpy(x.astype(np.int64))
        yt = torch.from_numpy(y.astype(np.int64))
        if device.startswith("cuda"):
            return (xt.pin_memory().to(device, non_blocking=True),
                    yt.pin_memory().to(device, non_blocking=True))
        return xt.to(device), yt.to(device)

    def sequential_batches(self, batch_size: int, device: str,
                           max_batches: int = None):
        """Non-overlapping windows in order, for deterministic evaluation.

        Evaluation must not resample: perplexity computed on random windows
        moves between runs and is not comparable across checkpoints.
        """
        stride = self.seq_len
        n_windows = (len(self.tokens) - 1) // stride
        if max_batches is not None:
            n_windows = min(n_windows, max_batches * batch_size)
        for i in range(0, n_windows, batch_size):
            starts = [j * stride for j in range(i, min(i + batch_size, n_windows))]
            if not starts:
                break
            x = np.stack([self.tokens[s:s + self.seq_len] for s in starts])
            y = np.stack([self.tokens[s + 1:s + 1 + self.seq_len] for s in starts])
            yield (torch.from_numpy(x.astype(np.int64)).to(device),
                   torch.from_numpy(y.astype(np.int64)).to(device))


def load_splits(work: str, lang_dir: str, seq_len: int, seed: int = 1337) -> dict:
    """Open train/val/test for one language."""
    base = f"{work}/{lang_dir}/6_tokenized"
    out = {}
    for split in ("train", "val", "test"):
        try:
            out[split] = TokenArray(f"{base}/{split}.bin", seq_len, seed)
        except FileNotFoundError:
            out[split] = None
    if out["train"] is None:
        raise FileNotFoundError(f"no train.bin under {base}")
    return out


def split_bytes(work: str, lang_dir: str, split: str) -> int:
    """UTF-8 byte count of a split, recorded by stage 09.

    Bits-per-byte needs the byte count of the original text, not of the token
    array, and recomputing it from a regenerated corpus is error-prone.
    """
    import json

    p = Path(f"{work}/{lang_dir}/6_tokenized/token_report.json")
    if not p.exists():
        return 0
    return json.loads(p.read_text()).get(split, {}).get("bytes_of_text", 0)
