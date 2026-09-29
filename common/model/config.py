"""Model and training configuration.

One dataclass per concern, both loaded from a single YAML per language so a run
is reproducible from the file alone. The exact parameter count is derived from
the config rather than hardcoded, so changing a width or depth cannot leave a
stale number in the report.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Architecture. Everything the forward pass needs, and nothing else."""

    vocab_size: int
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    tie_embeddings: bool = True
    #: Pre-norm places LayerNorm inside the residual branch, which keeps a clean
    #: identity path from input to output and is what makes deep stacks trainable
    #: without a warmup-sensitive learning rate. Post-norm is offered for the
    #: comparison in the report.
    norm_first: bool = True
    #: Off only for the bonus ablation, which retrains one language without any
    #: position information to show what the positional table is carrying.
    use_positional_embeddings: bool = True

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must divide by n_heads {self.n_heads}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> dict:
        """Exact parameter count, broken down by component.

        Derived from the config rather than measured, so it can be reported
        before a model is built. `DecoderLM.parameter_report` checks it against
        the real modules.
        """
        d, V, L = self.d_model, self.vocab_size, self.n_layers
        token_emb = V * d
        pos_emb = self.max_seq_len * d if self.use_positional_embeddings else 0
        attn = 4 * (d * d + d)                       # W_Q, W_K, W_V, W_O
        ffn = (d * self.d_ff + self.d_ff) + (self.d_ff * d + d)
        norms = 2 * (2 * d)                          # two LayerNorms per block
        block = attn + ffn + norms
        head = 0 if self.tie_embeddings else V * d + V
        total = token_emb + pos_emb + L * block + 2 * d + head
        return {
            "token_embedding": token_emb,
            "positional_embedding": pos_emb,
            "per_block": block,
            "blocks_total": L * block,
            "final_norm": 2 * d,
            "output_head": head,
            "total": total,
            "embedding_share": (token_emb + pos_emb) / total,
        }


@dataclass
class TrainConfig:
    """Optimisation and schedule."""

    batch_size: int = 16
    grad_accum_steps: int = 8
    max_steps: int = 20000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 50
    checkpoint_every: int = 1000
    log_every: int = 20
    seed: int = 1337
    #: bf16 where the GPU supports it, fp16 otherwise, fp32 on CPU.
    precision: str = "auto"
    compile_model: bool = False

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum_steps


def load_configs(path: str, vocab_size: int = None) -> tuple:
    """Read one YAML into (ModelConfig, TrainConfig).

    `vocab_size` overrides the file, so the tokenizer stays the single source of
    truth for how many pieces exist.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model = dict(raw.get("model", {}))
    if vocab_size is not None:
        model["vocab_size"] = vocab_size
    return ModelConfig(**model), TrainConfig(**raw.get("train", {}))


def save_configs(model_cfg: ModelConfig, train_cfg: TrainConfig, path: str):
    """Write both configs beside a checkpoint, so a run is reproducible."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump({"model": asdict(model_cfg), "train": asdict(train_cfg)},
                       sort_keys=False),
        encoding="utf-8")
