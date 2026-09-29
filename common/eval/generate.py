"""Sampling continuations from held-out prefixes, and decoding them to text."""
import numpy as np
import torch


def load_tokenizer(lang_dir: str, lang: str):
    """The Phase 1 SentencePiece model for this language."""
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=f"{lang_dir}/tokenizer/{lang}.model")


def held_out_prefixes(data, n_prompts: int, prefix_len: int,
                      continuation_len: int, seed: int = 1337) -> list:
    """Sample (prefix, reference continuation) pairs from a held-out split.

    Both come from the same window, so the reference is exactly what followed
    the prefix in real text. That is the fairest available target for BLEU and
    friends, while still being only one of many acceptable continuations.
    """
    rng = np.random.default_rng(seed)
    total = prefix_len + continuation_len
    starts = rng.integers(0, len(data.tokens) - total - 1, size=n_prompts)
    out = []
    for s in starts:
        window = np.asarray(data.tokens[s:s + total]).astype(np.int64)
        out.append((window[:prefix_len].tolist(),
                    window[prefix_len:].tolist()))
    return out


@torch.no_grad()
def generate_batch(model, prefixes: list, max_new_tokens: int,
                   temperature: float, device: str, top_k: int = None) -> list:
    """Continue each prefix. temperature=0 means greedy decoding.

    Prefixes are padded to a common length on the left so one forward pass
    covers the batch; the padding is stripped from the returned continuations.
    """
    lengths = {len(p) for p in prefixes}
    if len(lengths) != 1:
        raise ValueError("all prefixes must share a length")
    idx = torch.tensor(prefixes, dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return out[:, idx.size(1):].tolist()


def decode_all(sp, sequences: list) -> list:
    """Token ids to text, dropping anything past the vocabulary."""
    texts = []
    for seq in sequences:
        ids = [int(t) for t in seq if 0 <= int(t) < sp.get_piece_size()]
        texts.append(sp.decode(ids))
    return texts
