"""Feed-forward sublayer and the transformer block that wraps both sublayers."""
import torch
import torch.nn as nn

from common.model.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise MLP: d_model -> d_ff -> d_model, with GELU between.

    Position-wise means the same weights apply at every position independently;
    attention is the only place information moves along the sequence. The inner
    dimension is wider than the model dimension (4x here) because this is where
    most of the model's capacity to store patterns lives.
    """

    def __init__(self, cfg):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Attention sublayer then feed-forward sublayer, each with a residual.

    Pre-norm (`norm_first=True`) normalises the input to each sublayer and adds
    the result to the unnormalised residual:

        x = x + Attn(LN(x));  x = x + FFN(LN(x))

    That leaves an unbroken identity path from the first layer to the last, so
    gradients reach early layers without passing through a normalisation at
    every step. Post-norm normalises after the addition instead, which is the
    original formulation and needs a more careful warmup to stay stable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm_first = cfg.norm_first
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        if self.norm_first:
            attn_out, weights = self.attn(self.ln_1(x), return_weights)
            x = x + attn_out
            x = x + self.ffn(self.ln_2(x))
        else:
            attn_out, weights = self.attn(x, return_weights)
            x = self.ln_1(x + attn_out)
            x = self.ln_2(x + self.ffn(x))
        return x, weights
