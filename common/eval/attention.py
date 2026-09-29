"""What the attention heads learned: entropy and mean attention distance.

Computation only. `common/viz.py` draws the heatmaps from the arrays returned
here, keeping visualisation and computation in separate functions.
"""
import torch


@torch.no_grad()
def collect_attention(model, idx: torch.Tensor) -> list:
    """Run one batch and return each layer's attention weights.

    Returns a list of length n_layers, each (B, n_heads, T, T).
    """
    model.eval()
    _, _, attentions = model(idx, return_attention=True)
    return attentions


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy per head, in nats, averaged over batch and query position.

    Args:
        weights: (B, h, T, T) attention probabilities.

    A head attending equally to all t positions available to it has entropy
    ln(t); a head attending to exactly one position has entropy 0. So high
    entropy means diffuse, low means sharply focused. Because of causal masking
    the number of available keys grows with position, which is why the maximum
    attainable entropy is reported alongside.
    """
    w = weights.float().clamp_min(1e-12)
    ent = -(w * w.log()).sum(dim=-1)          # (B, h, T)
    return ent.mean(dim=(0, 2))               # (h,)


def max_entropy_for_length(T: int, device=None) -> float:
    """Entropy of a uniform causal distribution, averaged over query positions.

    The ceiling entropy can be compared against: position t may attend to t+1
    keys, so uniform attention there has entropy ln(t+1).
    """
    t = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    return t.log().mean().item()


def mean_attention_distance(weights: torch.Tensor) -> torch.Tensor:
    """Average distance, in positions, between a query and the keys it attends.

    Args:
        weights: (B, h, T, T).

    distance = sum_j w_ij * (i - j), averaged over queries and batch. A head
    that only looks at the previous token scores near 1; a head that spreads
    attention over the whole prefix grows with sequence length. This is what
    separates local or positional heads from long-range content-based ones.
    """
    B, h, T, _ = weights.shape
    i = torch.arange(T, device=weights.device).view(1, 1, T, 1)
    j = torch.arange(T, device=weights.device).view(1, 1, 1, T)
    dist = (i - j).clamp_min(0).float()
    per_query = (weights.float() * dist).sum(dim=-1)     # (B, h, T)
    return per_query.mean(dim=(0, 2))                    # (h,)


def summarise(attentions: list) -> dict:
    """Per-layer, per-head entropy and mean distance for a whole model."""
    T = attentions[0].shape[-1]
    rows = []
    for layer, w in enumerate(attentions):
        ent = attention_entropy(w)
        dist = mean_attention_distance(w)
        for head in range(w.shape[1]):
            rows.append({
                "layer": layer,
                "head": head,
                "entropy": float(ent[head]),
                "mean_distance": float(dist[head]),
            })
    return {
        "seq_len": int(T),
        "max_entropy": max_entropy_for_length(T),
        "heads": rows,
        "layer_means": [
            {"layer": l,
             "entropy": sum(r["entropy"] for r in rows if r["layer"] == l)
                        / max(sum(1 for r in rows if r["layer"] == l), 1),
             "mean_distance": sum(r["mean_distance"] for r in rows if r["layer"] == l)
                              / max(sum(1 for r in rows if r["layer"] == l), 1)}
            for l in sorted({r["layer"] for r in rows})
        ],
    }
