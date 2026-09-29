"""Intrinsic language-model metrics: cross-entropy, perplexity, bits-per-byte."""
import math

import torch


@torch.no_grad()
def cross_entropy_and_tokens(model, data, batch_size: int, device: str,
                             max_batches: int = None) -> tuple:
    """Mean cross-entropy in nats, and how many target tokens it covers.

    Batches are weighted by their token count rather than averaged over batches,
    so a short final batch cannot skew the result.
    """
    model.eval()
    total_nats, total_tokens = 0.0, 0
    for x, y in data.sequential_batches(batch_size, device, max_batches):
        _, loss, _ = model(x, y)
        n = x.size(0) * x.size(1)
        total_nats += loss.item() * n
        total_tokens += n
    return total_nats / max(total_tokens, 1), total_tokens


def perplexity(cross_entropy: float) -> float:
    """exp(cross-entropy). The effective number of equally likely choices."""
    return math.exp(min(cross_entropy, 20))


def bits_per_byte(cross_entropy: float, n_tokens: int, n_bytes: int) -> float:
    """Convert per-token cross-entropy into bits per UTF-8 byte.

        bpb = cross_entropy_nats * tokens * log2(e) / bytes

    Perplexity is not comparable between the two models: they use different
    tokenizers, so a "token" means something different in each. Bits-per-byte
    normalises by the underlying text instead, which is the same unit in both
    languages and is what makes an H-vs-L comparison meaningful.
    """
    if not n_bytes:
        return float("nan")
    return cross_entropy * n_tokens * math.log2(math.e) / n_bytes
