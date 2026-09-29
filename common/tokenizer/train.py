"""Train all three tokenizer algorithms from scratch, per language.

No pretrained tokenizer is used anywhere. Each language gets its own training
corpus (its train split only), vocabulary and model files.

    BPE        merge the most frequent adjacent pair
    WordPiece  merge by freq(pair) / (freq(a) * freq(b))
    Unigram    prune a large candidate vocabulary by likelihood loss
"""
import os
from pathlib import Path

import sentencepiece as spm

#: Algorithms implemented by SentencePiece rather than HuggingFace tokenizers.
SPM_ALGOS = ("unigram", "bpe")


def train_sentencepiece(corpus_txt: str, out_prefix: str, vocab_size: int,
                        model_type: str, cfg: dict) -> str:
    """Train a SentencePiece unigram or BPE model, returning the .model path.

    The settings that matter, all from tokenizer.yaml: character_coverage 0.9995
    for a large-inventory Indic script; byte_fallback so there are no true UNKs;
    split_digits so digits stay atomic; normalization_rule_name identity, since
    the corpus is normalised once already and a second normaliser inside the
    .model file would transform text again at inference; input_sentence_size to
    bound RAM; and max_sentence_length raised from the 4192-byte default, which
    would otherwise truncate long documents silently.
    """
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    sp = cfg["special_tokens"]
    spm.SentencePieceTrainer.train(
        input=corpus_txt,
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=cfg["character_coverage"],
        byte_fallback=cfg["byte_fallback"],
        split_digits=cfg["split_digits"],
        add_dummy_prefix=cfg["add_dummy_prefix"],
        remove_extra_whitespaces=cfg["remove_extra_whitespaces"],
        allow_whitespace_only_pieces=cfg["allow_whitespace_only_pieces"],
        max_sentencepiece_length=cfg["max_sentencepiece_length"],
        normalization_rule_name=cfg["normalization_rule_name"],
        input_sentence_size=cfg["input_sentence_size"],
        shuffle_input_sentence=cfg["shuffle_input_sentence"],
        max_sentence_length=cfg["max_sentence_length"],
        unk_id=sp["unk"], bos_id=sp["bos"], eos_id=sp["eos"], pad_id=sp["pad"],
        user_defined_symbols=list(cfg.get("user_defined_symbols", [])),
        vocabulary_output_piece_score=True,
        num_threads=os.cpu_count() or 2,
    )
    return out_prefix + ".model"


def train_wordpiece(corpus_txt: str, out_path: str, vocab_size: int,
                    cfg: dict) -> str:
    """Train a WordPiece tokenizer, returning the tokenizer.json path.

    WordPiece scores a candidate pair as freq(pair)/(freq(a)*freq(b)), so it
    prefers pairs whose parts are individually rare. It also emits [UNK] for a
    whole word when any part is unmatchable, where byte-fallback SentencePiece
    never emits UNK at all -- which is what makes the UNK comparison in
    evaluate.py worth reporting.
    """
    from tokenizers import Tokenizer, decoders, models, normalizers
    from tokenizers import pre_tokenizers, trainers

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Identity normaliser, for the same reason SentencePiece uses one: the
    # corpus is normalised once, in one place.
    tok.normalizer = normalizers.Sequence([])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Digits(individual_digits=cfg["split_digits"]),
    ])
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
                       + list(cfg.get("user_defined_symbols", [])),
        continuing_subword_prefix="##",
        min_frequency=2,
    )

    def batches(path: str, size: int = 10_000):
        buf = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                buf.append(line.rstrip("\n"))
                if len(buf) >= size:
                    yield buf
                    buf = []
        if buf:
            yield buf

    tok.train_from_iterator(batches(corpus_txt), trainer=trainer)
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.save(out_path)
    return out_path


def train(algorithm: str, corpus_txt: str, out_base: str, vocab_size: int,
          cfg: dict) -> str:
    """Dispatch to the right trainer. Returns the model path."""
    if algorithm in SPM_ALGOS:
        return train_sentencepiece(corpus_txt, out_base, vocab_size,
                                   algorithm, cfg)
    if algorithm == "wordpiece":
        return train_wordpiece(corpus_txt, out_base + ".json", vocab_size, cfg)
    raise ValueError(f"unknown algorithm: {algorithm}")


def export_vocab_txt(model_path: str, out_txt: str, algorithm: str):
    """Write the vocabulary as plain text, one piece and score per line."""
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    if algorithm in SPM_ALGOS:
        sp = spm.SentencePieceProcessor(model_file=model_path)
        with open(out_txt, "w", encoding="utf-8") as fh:
            for i in range(sp.get_piece_size()):
                fh.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i):.6f}\n")
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(model_path)
        with open(out_txt, "w", encoding="utf-8") as fh:
            for piece, idx in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
                fh.write(f"{piece}\t{idx}\n")
