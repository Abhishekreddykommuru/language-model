"""Decoder-only transformer language model.

    tokens -> token embedding + positional embedding -> dropout
           -> N transformer blocks
           -> final LayerNorm
           -> linear projection to vocabulary
           -> cross-entropy against the next token

Positional information uses **learned absolute embeddings**: a second embedding
table indexed by position, added to the token embedding. Self-attention is
permutation invariant, so without this the model cannot tell "the dog bit the
man" from "the man bit the dog". The consequence is a hard ceiling at
`max_seq_len` positions -- there is no embedding row for position 512 in a
512-position model, so the context length cannot be extended after training
without adding and training new rows. Sinusoidal or rotary encodings would
extrapolate further; learned absolute was chosen because it is the simplest
thing to explain end to end and the context length is fixed for this project.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLM(nn.Module):
    """~25M-parameter causal language model built from primitive layers only."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # The ablation drops the table entirely rather than zeroing it, so the
        # ablated model has no position parameters to train at all.
        self.position_embedding = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model)
            if cfg.use_positional_embeddings else None)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        from common.model.block import TransformerBlock
        self.blocks = nn.ModuleList([TransformerBlock(cfg)
                                     for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the output projection reuses the input embedding matrix.
        # It saves vocab_size x d_model parameters -- 6.1M of a 24M model at a
        # 16k vocabulary -- and ties the representation of a token as input to
        # its representation as a prediction target, which is a reasonable prior
        # and consistently helps at this scale.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scale the residual projections by 1/sqrt(2N). Each layer adds two
        # residual contributions, so without this the variance of the residual
        # stream grows with depth and deep stacks start unstable.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.W_o.weight, mean=0.0,
                                  std=0.02 * scale)
            torch.nn.init.normal_(block.ffn.fc_out.weight, mean=0.0,
                                  std=0.02 * scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                return_attention: bool = False):
        """Run the model.

        Args:
            idx: (B, T) token ids.
            targets: (B, T) next-token ids, already shifted so
                targets[i] follows idx[i]. None for inference.
            return_attention: also return each layer's attention weights.

        Returns:
            (logits, loss, attentions). `loss` is None without targets;
            `attentions` is None unless requested.
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}; learned positional embeddings have no "
                f"row beyond that")

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.embed_dropout(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, weights = block(x, return_weights=return_attention)
            if return_attention:
                attentions.append(weights)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # `targets` must already be the next tokens: targets[i] is the token
            # that follows idx[i]. The data layer produces that shift when it
            # cuts windows, so shifting again here would train the model to
            # predict two positions ahead -- a bug that is invisible at
            # initialisation, since both alignments give a loss of ln(V).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, attentions

    def parameter_report(self) -> dict:
        """Measured parameter counts, checked against the config's arithmetic."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.token_embedding.weight.numel()
        if self.position_embedding is not None:
            embedding += self.position_embedding.weight.numel()
        blocks = sum(p.numel() for b in self.blocks for p in b.parameters())
        predicted = self.cfg.n_params()["total"]
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "blocks": blocks,
            "embedding_share": embedding / total,
            "predicted_by_config": predicted,
            "matches_config": total == predicted,
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None,
                 eos_id: int = None) -> torch.Tensor:
        """Autoregressive sampling.

        `temperature` divides the logits before softmax: below 1 sharpens the
        distribution towards the mode, above 1 flattens it. Temperature 0 is
        treated as greedy decoding, taking the argmax outright.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to the context window; positions beyond max_seq_len have no
            # embedding.
            window = idx[:, -self.cfg.max_seq_len:]
            logits, _, _ = self(window)
            logits = logits[:, -1, :]

            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
