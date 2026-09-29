"""Tokenizer evaluation: fertility, UNK statistics, frequency stats, vocab sweep.

Vocabulary size is chosen from fertility and unknown-token rate measured on
held-out text, so the whole sweep is the deliverable, not just the final number.
"""
from collections import Counter

from common.tokenizer.train import SPM_ALGOS

#: Fixed probe words for the tokenization-examples table. Chosen once and never
#: changed, so comparisons across runs stay meaningful.
PROBE_WORDS = {
    "tel": ["ప్రభుత్వం", "విద్యార్థి", "హైదరాబాద్", "చేస్తున్నారు", "కంప్యూటర్",
            "2026", "తెలంగాణ", "పుస్తకాలు", "అభివృద్ధి", "వాళ్ళు",
            "సినిమా", "రాష్ట్రంలోని", "శాస్త్రవేత్తలు", "మంచిది", "అంతర్జాతీయ"],
    "nep": ["सरकार", "विद्यार्थी", "काठमाडौं", "गरिरहेका", "कम्प्युटर",
            "2026", "नेपाली", "पुस्तकहरू", "विकास", "उनीहरू",
            "चलचित्र", "राज्यको", "वैज्ञानिकहरू", "राम्रो", "अन्तर्राष्ट्रिय"],
}


class TokenizerWrapper:
    """Uniform interface over SentencePiece and HuggingFace tokenizers."""

    def __init__(self, path: str, algorithm: str):
        self.algorithm = algorithm
        if algorithm in SPM_ALGOS:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=path)
            self.unk_id = self.sp.unk_id()
            self.size = self.sp.get_piece_size()
        else:
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(path)
            self.unk_id = self.tok.token_to_id("[UNK]")
            self.size = self.tok.get_vocab_size()

    def encode(self, text: str) -> list:
        if self.algorithm in SPM_ALGOS:
            return self.sp.encode(text)
        return self.tok.encode(text).ids

    def encode_pieces(self, text: str) -> list:
        if self.algorithm in SPM_ALGOS:
            return self.sp.encode(text, out_type=str)
        return self.tok.encode(text).tokens

    def id_to_piece(self, i: int) -> str:
        """The surface string of one token id, for frequency tables."""
        if self.algorithm in SPM_ALGOS:
            return self.sp.id_to_piece(i)
        return self.tok.id_to_token(i) or ""

    @staticmethod
    def is_byte_fallback(piece: str) -> bool:
        """SentencePiece byte-fallback pieces look like '<0xE0>'."""
        return piece.startswith("<0x") and piece.endswith(">")


def evaluate(tok: TokenizerWrapper, held_out_lines: list) -> dict:
    """Compute every reported metric on held-out text.

        chars_per_token     average characters per token
        bytes_per_token     for Phase 2 bits-per-byte and corpus planning
        tokens_per_word     fertility, which drives the vocabulary choice
        unk_rate            zero for byte-fallback SentencePiece
        byte_fallback_rate  the meaningful version of unk_rate there
        vocab_used_frac     share of the vocabulary seen in held-out text
        dead_vocab_frac     share never seen, i.e. wasted embedding parameters
    """
    n_tok = n_unk = n_bf = 0
    n_chars = n_bytes = n_words = 0
    freq = Counter()

    for line in held_out_lines:
        if not line.strip():
            continue
        ids = tok.encode(line)
        pieces = tok.encode_pieces(line)
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == tok.unk_id)
        n_bf += sum(1 for p in pieces if tok.is_byte_fallback(p))
        n_chars += len(line)
        n_bytes += len(line.encode("utf-8"))
        n_words += len(line.split())
        freq.update(ids)

    n_tok = max(n_tok, 1)
    return {
        "vocab_size": tok.size,
        "n_tokens": n_tok,
        "chars_per_token": n_chars / n_tok,
        "bytes_per_token": n_bytes / n_tok,
        "tokens_per_word": n_tok / max(n_words, 1),
        "unk_rate": n_unk / n_tok,
        "byte_fallback_rate": n_bf / n_tok,
        "vocab_used_frac": len(freq) / max(tok.size, 1),
        "dead_vocab_frac": 1.0 - len(freq) / max(tok.size, 1),
        "top_50": [(int(i), int(c)) for i, c in freq.most_common(50)],
    }


def phase2_param_estimate(vocab_size: int, d_model: int = 384,
                          n_layers: int = 6) -> dict:
    """Phase 2 parameter count implied by a vocabulary size.

    For a decoder with tied embeddings and d_ff = 4*d_model,

        params ~= V*d + L*(4*d^2 + 2*d*d_ff) = V*d + L*12*d^2

    Phase 2 targets ~25M parameters, so at d_model 384 a 32k vocabulary is
    already 54% of the model: vocabulary size is not free.
    """
    emb = vocab_size * d_model
    body = n_layers * 12 * d_model * d_model
    total = emb + body
    return {"embedding": emb, "transformer": body, "total": total,
            "embedding_share": emb / total}


def segmentation_examples(tok: TokenizerWrapper, words: list) -> list:
    """How each probe word is segmented, for the examples table."""
    out = []
    for w in words:
        pieces = tok.encode_pieces(w)
        out.append({"word": w, "pieces": pieces, "n_pieces": len(pieces)})
    return out


def select_best(results: list, max_embedding_share: float = 0.50,
                within_frac: float = 0.95, max_vocab: int = 65536) -> dict:
    """The smallest vocabulary whose chars/token is within 5% of the best
    observed, that still leaves half the Phase 2 parameter budget for the
    transformer body and keeps token ids inside uint16."""
    if not results:
        raise ValueError("no sweep results to select from")
    best = max(r["chars_per_token"] for r in results)
    eligible = [
        r for r in results
        if r["chars_per_token"] >= within_frac * best
        and r["phase2"]["embedding_share"] <= max_embedding_share
        and r["vocab_size"] <= max_vocab
    ]
    if not eligible:
        # Fall back to the parameter constraint alone rather than failing.
        eligible = [r for r in results
                    if r["phase2"]["embedding_share"] <= max_embedding_share] or results
    return min(eligible, key=lambda r: (r["vocab_size"], -r["chars_per_token"]))
