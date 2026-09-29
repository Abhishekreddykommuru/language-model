"""Small helpers shared by the Phase 2 scripts."""
import json
from pathlib import Path


def run_name(lang: str, variant: str = "") -> str:
    """What a run is called on disk.

    The standard run is just the language code, so Phase 2 filenames are
    unchanged. A variant -- the finetuned model, or the bonus ablation -- gets a
    suffix, which keeps every checkpoint, figure and table for that run beside
    the others without overwriting them.
    """
    return f"{lang}_{variant}" if variant else lang


def check_tokenizer(sp, model_cfg, lang: str):
    """Refuse to run when the tokenizer and the checkpoint disagree.

    A checkpoint carries the vocabulary size it was trained with. If the
    tokenizer in the repository is a different one -- a different sweep, a
    different vocabulary size -- then token ids no longer mean what the weights
    expect, and the failure is an out-of-range embedding lookup a hundred lines
    later. Checking here says what is actually wrong.
    """
    pieces = sp.get_piece_size()
    if pieces != model_cfg.vocab_size:
        raise SystemExit(
            f"tokenizer and checkpoint disagree for {lang}: the tokenizer has "
            f"{pieces:,} pieces, the checkpoint was trained with "
            f"{model_cfg.vocab_size:,}. Copy the tokenizer that produced this "
            f"checkpoint into {lang}/tokenizer/ before continuing.")


def vocab_size_of(langdir: str, lang: str) -> int:
    """The real vocabulary size, from the Phase 1 tokenizer.

    The tokenizer is the single source of truth; reading it from a config would
    let the two drift apart and produce a model whose output layer does not
    match its own vocabulary.
    """
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(
            model_file=f"{langdir}/tokenizer/{lang}.model")
        return sp.get_piece_size()
    except Exception:
        p = Path(f"{langdir}/tokenizer/chosen.json")
        if p.exists():
            return json.loads(p.read_text())["chosen"]["vocab_size"]
        raise


def load_model(ckpt_path: str, device: str):
    """Rebuild a model from a checkpoint, config included.

    Returns (model, model_config, checkpoint_dict).
    """
    import torch

    from common.model.config import ModelConfig
    from common.model.transformer import DecoderLM

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = DecoderLM(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mcfg, ck
