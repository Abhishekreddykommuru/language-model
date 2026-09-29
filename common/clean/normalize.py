"""Unicode and Indic-specific normalisation.

Applied exactly once, before every other stage. The tokenizers then train with
normalization_rule_name=identity, so the project has a single normaliser: were
SentencePiece allowed to normalise too, a second and different one would be
baked into the .model file and would transform text at inference time.
"""
import re
import unicodedata

# --- invisible / structural characters -------------------------------------
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
BIDI_RE = re.compile("[‎‏‪-‮⁦-⁩]")
ZERO_WIDTH_RE = re.compile("[﻿­]")  # BOM, soft hyphen
ZWJ = "‍"
ZWNJ = "‌"
SPACES_RE = re.compile("[   -   　]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTINEWLINE_RE = re.compile(r"\n{3,}")

# --- Devanagari nukta ------------------------------------------------------
# Precomposed nukta letters are decomposed to base + U+093C so that every word
# has exactly one byte representation. Without this, "क़लम" spelled two ways
# splits token statistics and defeats deduplication.
NUKTA_MAP = {
    "ऩ": "ऩ",  # ऩ
    "ऱ": "ऱ",  # ऱ
    "ऴ": "ऴ",  # ऴ
    "क़": "क़",  # क़
    "ख़": "ख़",  # ख़
    "ग़": "ग़",  # ग़
    "ज़": "ज़",  # ज़
    "ड़": "ड़",  # ड़
    "ढ़": "ढ़",  # ढ़
    "फ़": "फ़",  # फ़
    "य़": "य़",  # य़
}
NUKTA_RE = re.compile("[" + "".join(NUKTA_MAP) + "]")

# --- digits ----------------------------------------------------------------
# Native digits are mapped to ASCII. This unifies numeric statistics and makes
# Phase 3 (comparative numeric reasoning) materially easier.
DIGIT_MAP = {}
for _zero in (0x0966, 0x0C66):  # Devanagari, Telugu
    for _d in range(10):
        DIGIT_MAP[chr(_zero + _d)] = str(_d)
DIGIT_RE = re.compile("[" + "".join(DIGIT_MAP) + "]")

# --- punctuation -----------------------------------------------------------
QUOTE_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}
QUOTE_RE = re.compile("[" + "".join(QUOTE_MAP) + "]")

# Danda and double danda are genuine sentence terminators in Indic scripts and
# must survive normalisation: the common "keep lines ending in punctuation"
# heuristic is meaningless for Indic text unless these are counted.
TERMINATORS = ".!?।॥"

SCRIPT_RANGES = {
    "tel": (0x0C00, 0x0C7F),  # Telugu block, used by no other language
    "nep": (0x0900, 0x097F),  # Devanagari, shared with hi/mr/mai/bho/sa
}


def normalize(text: str, lang: str) -> str:
    """Normalise one document. Deterministic and idempotent.

    Order matters: steps 4-7 can leave decomposed sequences, so NFC is applied
    again at the end.
    """
    # 1. strip invisible junk
    text = CONTROL_RE.sub("", text)
    text = BIDI_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)

    # 2. unify every space character to U+0020
    text = SPACES_RE.sub(" ", text)

    # 3. canonical composition
    text = unicodedata.normalize("NFC", text)

    # 4. Devanagari nukta: one consistent spelling
    if lang == "nep":
        text = NUKTA_RE.sub(lambda m: NUKTA_MAP[m.group()], text)

    # 5. ZWJ/ZWNJ are inserted inconsistently by content management systems and
    #    create invisible token variants that also break deduplication.
    text = text.replace(ZWJ, "").replace(ZWNJ, "")

    # 6. native digits to ASCII
    text = DIGIT_RE.sub(lambda m: DIGIT_MAP[m.group()], text)

    # 7. punctuation (danda deliberately untouched)
    text = QUOTE_RE.sub(lambda m: QUOTE_MAP[m.group()], text)

    # 8. whitespace layout
    text = MULTISPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTINEWLINE_RE.sub("\n\n", text)

    # 9. re-apply NFC
    return unicodedata.normalize("NFC", text).strip()


def script_ratio(text: str, lo: int, hi: int) -> float:
    """Fraction of LETTER characters falling inside a Unicode range.

    Digits, punctuation and whitespace are excluded from the denominator so that
    a numeric table is not mistaken for the wrong language.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def latin_ratio(text: str) -> float:
    """Fraction of letters that are Latin -- a code-mixing indicator."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 0x250) / len(letters)
