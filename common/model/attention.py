"""Multi-head causal self-attention, written out in full.

No `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`: only Linear,
Dropout and tensor operations, so every shape in the forward pass is visible.

Shapes, with B batch, T sequence length, d model dimension, h heads:

    x            (B, T, d)
    Q, K, V      (B, T, d)      -> (B, h, T, d_head) after reshape+transpose
    scores       (B, h, T, T)
    weights      (B, h, T, T)   after mask and softmax
    context      (B, h, T, d_head) -> (B, T, d) after transpose+reshape
    output       (B, T, d)
"""
import math

import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal self-attention over h heads.

    The causal mask is built here rather than taken from a library: an
    upper-triangular block of -inf added to the scores before softmax, so
    position t can attend only to positions <= t. `scripts/19_sanity.py`
    verifies that empirically by perturbing a future token and checking the
    logits at earlier positions do not move.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        # One Linear per projection rather than a fused qkv matrix: slightly
        # slower, much easier to explain and to inspect head by head.
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Registered as a buffer so it moves with .to(device) and is saved with
        # nothing (persistent=False keeps it out of the state dict).
        mask = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len,
                                     dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, h, T, d_head).

        The reshape splits the model dimension into heads; the transpose puts
        the head axis before time so the matmuls below are batched per head.
        """
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, T, d_head) -> (B, T, d). The inverse of _split_heads.

        `contiguous` is required because transpose only changes strides, and
        view needs a contiguous buffer.
        """
        B, h, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * dh)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Returns (output, weights). `weights` is None unless requested.

        The attention weights are what the Phase 2 attention analysis needs, so
        the forward pass can hand them back rather than being re-run through a
        hook.
        """
        B, T, _ = x.shape

        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # Scaled dot-product. The 1/sqrt(d_head) scaling keeps the variance of
        # the scores at ~1 as d_head grows: q.k is a sum of d_head products of
        # unit-variance terms, so its variance grows linearly with d_head, and
        # without the scaling softmax saturates and gradients vanish.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal masking: -inf on the strict upper triangle, so those positions
        # contribute exactly zero after softmax. Sliced to T so shorter
        # sequences than max_seq_len work unchanged.
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = self.attn_dropout(weights) @ v

        out = self.resid_dropout(self.W_o(self._merge_heads(attended)))
        return (out, weights) if return_weights else (out, None)
