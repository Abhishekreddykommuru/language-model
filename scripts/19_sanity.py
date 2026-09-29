"""Checks to run before committing hours of GPU time.

Three things, each of which catches a class of bug that is otherwise invisible
until the loss curve looks wrong:

  1. Parameter count matches the config's arithmetic, and lands near 25M.
  2. Loss before training is ln(vocab_size). At initialisation the logits carry
     no information, so the softmax is near-uniform and every token has
     probability 1/V; cross-entropy is then -ln(1/V) = ln(V). A materially
     different starting loss means the initialisation, the loss reduction or
     the target shift is wrong.
  3. The causal mask holds. Changing token t+1 must not change the logits at
     position t. This is the only empirical way to be sure the mask is applied
     where it is claimed.
  4. Targets are aligned. A tiny model is trained for a few hundred steps on a
     sequence whose next token is a deterministic function of the current one,
     and must then predict it. This catches an off-by-one between the data
     layer and the loss, which nothing else here would: a model predicting two
     tokens ahead still starts at ln(V) and still trains down.

Usage:
    python scripts/19_sanity.py --lang tel
"""
import argparse
import math
import sys

import torch

sys.path.insert(0, ".")

from common.io.paths import lang_dir
from common.log import kv, section, table
from common.model.config import load_configs
from common.model.transformer import DecoderLM


def vocab_size_of(langdir: str, lang: str) -> int:
    """Read the real vocabulary size from the Phase 1 tokenizer."""
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(
            model_file=f"{langdir}/tokenizer/{lang}.model")
        return sp.get_piece_size()
    except Exception:
        import json
        from pathlib import Path

        p = Path(f"{langdir}/tokenizer/chosen.json")
        if p.exists():
            return json.loads(p.read_text())["chosen"]["vocab_size"]
        raise


def check_initial_loss(model, vocab_size: int, device: str) -> dict:
    """Loss on random tokens from an untrained model.

    eval() matters: dropout is stochastic, and with it active the measured loss
    carries noise that has nothing to do with the initialisation.
    """
    model.eval()
    torch.manual_seed(0)
    seq = torch.randint(0, vocab_size, (4, 129), device=device)
    x, y = seq[:, :-1], seq[:, 1:]          # y[i] is the token following x[i]
    with torch.no_grad():
        _, loss, _ = model(x, y)
    expected = math.log(vocab_size)
    return {"observed": loss.item(), "expected": expected,
            "ratio": loss.item() / expected}


def check_causal_mask(model, vocab_size: int, device: str) -> dict:
    """Perturb a future token; the logits before it must not move.

    eval() is essential here rather than merely tidy. With dropout active the
    two forward passes differ by random masking alone, and the comparison
    measures dropout noise instead of information flow.
    """
    model.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, vocab_size, (1, 32), device=device)
    t = 16

    with torch.no_grad():
        base, _, _ = model(idx)
        altered = idx.clone()
        # Change the token at t+1 to something different.
        altered[0, t + 1] = (altered[0, t + 1] + 1) % vocab_size
        after, _, _ = model(altered)

    before = (base[0, :t + 1] - after[0, :t + 1]).abs().max().item()
    at_and_after = (base[0, t + 1:] - after[0, t + 1:]).abs().max().item()
    return {"max_change_up_to_t": before,
            "max_change_from_t_plus_1": at_and_after,
            "causal": before < 1e-5 and at_and_after > 0}


def check_alignment(device: str) -> dict:
    """Train a small model on a predictable sequence; it must learn t -> t+1.

    The sequence is 0, 1, 2, ... mod V, so the next token is fully determined
    by the current one. A correctly aligned model drives the loss to nearly
    zero and reproduces the shifted sequence exactly.
    """
    from common.model.config import ModelConfig

    V = 64
    cfg = ModelConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=32, dropout=0.0, attn_dropout=0.0)
    torch.manual_seed(0)
    m = DecoderLM(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    seq = torch.arange(0, 33, device=device).remainder(V).unsqueeze(0).repeat(16, 1)
    x, y = seq[:, :-1], seq[:, 1:]
    for _ in range(300):
        _, loss, _ = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        logits, final, _ = m(x, y)
        accuracy = (logits.argmax(-1) == y).float().mean().item()
    return {"final_loss": final.item(), "accuracy": accuracy,
            "example_in": x[0, :6].tolist(),
            "example_pred": logits.argmax(-1)[0, :6].tolist(),
            "aligned": accuracy > 0.95}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    langdir = lang_dir(args.lang)
    V = vocab_size_of(langdir, args.lang)
    mcfg, tcfg = load_configs(f"{langdir}/configs/model.yaml", vocab_size=V)
    model = DecoderLM(mcfg).to(args.device)

    section(f"Model configuration, {args.lang}")
    kv("vocabulary", f"{V:,}")
    kv("d_model / heads / d_head", f"{mcfg.d_model} / {mcfg.n_heads} / {mcfg.d_head}")
    kv("layers", mcfg.n_layers)
    kv("feed-forward width", f"{mcfg.d_ff} ({mcfg.d_ff // mcfg.d_model}x d_model)")
    kv("context length", mcfg.max_seq_len)
    kv("normalisation", "pre-norm" if mcfg.norm_first else "post-norm")
    kv("embeddings tied", mcfg.tie_embeddings)

    rep = model.parameter_report()
    section("Parameter count")
    table(["component", "parameters", "share"],
          [["token + positional embedding", f"{rep['embedding']:,}",
            f"{rep['embedding_share']:.1%}"],
           ["transformer blocks", f"{rep['blocks']:,}",
            f"{rep['blocks'] / rep['total']:.1%}"],
           ["total trainable", f"{rep['trainable']:,}", ""]])
    print()
    kv("matches config arithmetic", rep["matches_config"])

    section("Loss before training")
    li = check_initial_loss(model, V, args.device)
    kv("observed", f"{li['observed']:.4f}")
    kv("expected, ln(V)", f"{li['expected']:.4f}")
    kv("ratio", f"{li['ratio']:.4f}"
                f"   {'ok' if 0.95 < li['ratio'] < 1.05 else 'CHECK'}")

    section("Causal masking")
    cm = check_causal_mask(model, V, args.device)
    kv("max logit change at positions <= t", f"{cm['max_change_up_to_t']:.3e}")
    kv("max logit change at positions > t", f"{cm['max_change_from_t_plus_1']:.3e}")
    kv("verdict", "causal" if cm["causal"] else "NOT CAUSAL")
    print("\n  A future token was altered. Logits before it are unchanged, so no\n"
          "  position can see ahead; logits at and after it do change, which\n"
          "  confirms the perturbation was real.")

    section("Next-token alignment")
    al = check_alignment(args.device)
    kv("loss after 300 steps", f"{al['final_loss']:.4f}")
    kv("next-token accuracy", f"{al['accuracy']:.3f}")
    kv("input", al["example_in"])
    kv("predicted", al["example_pred"])
    kv("verdict", "aligned" if al["aligned"] else "MISALIGNED")
    print("\n  A model trained on 0,1,2,... must predict the following token.\n"
          "  If the data layer and the loss disagreed by one position, this\n"
          "  would predict two ahead while every other check still passed.")

    if not (rep["matches_config"] and cm["causal"] and al["aligned"]
            and 0.95 < li["ratio"] < 1.05):
        raise SystemExit("\nsanity checks failed; do not start training")


if __name__ == "__main__":
    main()
