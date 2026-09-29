"""Generation-quality metrics: BLEU, chrF++, ROUGE-L, diversity, repetition.

BLEU and chrF++ come from sacrebleu, which is the reference implementation and
handles tokenisation consistently; ROUGE-L and the diversity statistics are
short enough to implement directly and avoid another dependency.

A caveat that belongs in the report rather than buried here: all three overlap
metrics compare one generated continuation against one reference continuation.
Open-ended language modelling has no single correct continuation, so a fluent
generation that simply goes somewhere else scores near zero. They are reported
because relative differences between decoding temperatures are still
informative, but the absolute values should not be read as generation quality.
"""
from collections import Counter


def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length, the basis of ROUGE-L."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hypotheses: list, references: list, beta: float = 1.2) -> dict:
    """Corpus ROUGE-L, averaged over sentence-level F-scores.

    Beta weights recall over precision, following the original ROUGE paper.
    Word-level, which is the usual choice, though for a morphologically rich
    language it is harsh: a correct stem with a different case ending counts as
    a miss.
    """
    ps, rs, fs = [], [], []
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        lcs = _lcs_length(h, r)
        p = lcs / len(h) if h else 0.0
        rc = lcs / len(r) if r else 0.0
        if p and rc:
            f = ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)
        else:
            f = 0.0
        ps.append(p); rs.append(rc); fs.append(f)
    n = max(len(fs), 1)
    return {"rouge_l_precision": sum(ps) / n,
            "rouge_l_recall": sum(rs) / n,
            "rouge_l_f": sum(fs) / n}


def bleu_and_chrf(hypotheses: list, references: list) -> dict:
    """Corpus BLEU-4 and chrF++ via sacrebleu.

    chrF++ is character n-gram based, which suits Telugu and Nepali far better
    than word-level BLEU: both are agglutinative enough that a single wrong
    suffix destroys a word match while leaving most characters correct.
    """
    try:
        import sacrebleu
    except ImportError:
        return {"bleu": float("nan"), "chrf_pp": float("nan"),
                "note": "sacrebleu not installed"}
    refs = [references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs, tokenize="intl")
    chrf = sacrebleu.corpus_chrf(hypotheses, refs, word_order=2)   # chrF++
    return {"bleu": bleu.score, "bleu_order": 4,
            "bleu_signature": str(bleu.format(score_only=False))[:120],
            "chrf_pp": chrf.score}


def distinct_n(texts: list, n: int) -> float:
    """Unique n-grams over total n-grams, across all generations.

    Distinct-1 and Distinct-2 measure lexical diversity. A model that has
    collapsed onto a few safe phrases scores low even while its perplexity looks
    respectable.
    """
    total, unique = 0, set()
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repetition_rate(texts: list, n: int = 4) -> float:
    """Fraction of n-grams that are repeats within their own generation.

    Degenerate repetition is the classic failure of a small model decoded
    greedily, and it is invisible to perplexity.
    """
    repeated, total = 0, 0
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        counts = Counter(grams)
        repeated += sum(c - 1 for c in counts.values() if c > 1)
        total += len(grams)
    return repeated / total if total else 0.0


def diversity_report(texts: list) -> dict:
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "repetition_4gram": repetition_rate(texts, 4),
        "mean_words": sum(len(t.split()) for t in texts) / max(len(texts), 1),
    }


def all_metrics(hypotheses: list, references: list) -> dict:
    """Every text metric computed here, in one dict."""
    out = {}
    out.update(bleu_and_chrf(hypotheses, references))
    out.update(rouge_l(hypotheses, references))
    out.update(diversity_report(hypotheses))
    return out
