"""Heuristic quality filters: stage 3 of the Yi pipeline.

Thresholds live in each language's clean.yaml and differ between the two on
purpose. Telugu has data to spare and filters hard. Nepali filters gently and
runs in repair mode, where a document failing only a repetition test has the
offending lines removed instead of being discarded.
"""
from collections import Counter

from common.clean.normalize import TERMINATORS

BULLET_START = ("*", "-", "•", "‣", "·", "◦", ">")
SYMBOLS = "#…{}[]|<>"


def _ngrams(seq: list, n: int) -> list:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def compute_metrics(text: str) -> dict:
    """Every document statistic the filters need. Pure, so the statistics
    report can reuse it."""
    lines = [l for l in text.split("\n") if l.strip()]
    paras = [p for p in text.split("\n\n") if p.strip()]
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {"n_words": 0}

    n_symbols = sum(text.count(c) for c in SYMBOLS)
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    term_lines = sum(1 for l in lines if l.rstrip()[-1:] in TERMINATORS)

    line_counts = Counter(lines)
    para_counts = Counter(paras)
    dup_lines = sum(c - 1 for c in line_counts.values() if c > 1)
    dup_paras = sum(c - 1 for c in para_counts.values() if c > 1)

    bigrams = _ngrams(words, 2)
    top_bigram = Counter(bigrams).most_common(1)[0][1] if bigrams else 0
    grams5 = _ngrams(words, 5)
    dup5 = sum(c for c in Counter(grams5).values() if c > 1)

    return {
        "n_words": n_words,
        "n_lines": len(lines),
        "mean_word_len": sum(len(w) for w in words) / n_words,
        "symbol_word_ratio": n_symbols / n_words,
        "alpha_word_ratio": alpha_words / n_words,
        "terminal_punct_ratio": term_lines / max(len(lines), 1),
        "bullet_line_ratio": sum(1 for l in lines
                                 if l.lstrip()[:1] in BULLET_START) / max(len(lines), 1),
        "ellipsis_line_ratio": sum(1 for l in lines
                                   if l.rstrip().endswith("...")) / max(len(lines), 1),
        "dup_line_ratio": dup_lines / max(len(lines), 1),
        "dup_para_ratio": dup_paras / max(len(paras), 1),
        "top_2gram_ratio": top_bigram / max(len(bigrams), 1),
        "dup_5gram_ratio": dup5 / max(len(grams5), 1),
    }


class QualityFilter:
    """Document-level quality rules, with an optional repair mode."""

    #: Failures caused by repeated content, which deduplicating lines may fix.
    #: Duplicated lines also inflate the n-gram metrics, so all four have to be
    #: repairable or repair mode would never fire.
    REPAIRABLE = {"dup_lines", "dup_paras", "top_2gram", "dup_5gram"}

    def __init__(self, cfg: dict, stopwords: set = None):
        """Build a filter from the `quality` block of clean.yaml."""
        self.c = cfg
        self.stopwords = stopwords or set()
        self.repair = cfg.get("repair_mode", False)
        self.reasons = Counter()

    def _stopword_hits(self, text: str) -> int:
        if not self.stopwords:
            return 99                   # test disabled without a stopword list
        return sum(1 for w in set(text.split()) if w in self.stopwords)

    #: Line-structure rules only mean anything once a document has several
    #: lines. A single-paragraph news article is one line, so if it happens not
    #: to end in a full stop its terminal-punctuation ratio is 0 and it would be
    #: discarded. Auditing real Telugu text, that alone was 70% of rejections.
    MIN_LINES_FOR_STRUCTURE = 5

    def _structural_failures(self, m: dict) -> list:
        """Names of every structural rule this document fails."""
        c = self.c
        checks = [
            (m["dup_line_ratio"] > c["max_dup_line_ratio"], "dup_lines"),
            (m["dup_para_ratio"] > c["max_dup_para_ratio"], "dup_paras"),
            (m["top_2gram_ratio"] > c["max_top_2gram_ratio"], "top_2gram"),
            (m["dup_5gram_ratio"] > c["max_dup_5gram_ratio"], "dup_5gram"),
        ]
        if m.get("n_lines", 0) >= self.MIN_LINES_FOR_STRUCTURE:
            checks += [
                (m["bullet_line_ratio"] > c["max_bullet_line_ratio"], "bullet_lines"),
                (m["ellipsis_line_ratio"] > c["max_ellipsis_line_ratio"],
                 "ellipsis_lines"),
                (m["terminal_punct_ratio"] < c["min_terminal_punct_ratio"],
                 "no_terminal_punct"),
            ]
        return [name for failed, name in checks if failed]

    @staticmethod
    def _dedupe_lines(text: str) -> str:
        """Keep the first occurrence of each line, in order."""
        seen, out = set(), []
        for line in text.split("\n"):
            key = line.strip()
            if key and key in seen:
                continue
            seen.add(key)
            out.append(line)
        return "\n".join(out)

    def apply(self, text: str, boilerplate_lines: set = None) -> tuple:
        """Filter or repair one document.

        Returns (kept, text, reason, metrics). When kept is True the text may
        have been repaired, i.e. had lines removed.
        """
        c = self.c
        was_repaired = False

        if boilerplate_lines:
            lines = text.split("\n")
            keep = [l for l in lines if l.strip() not in boilerplate_lines]
            if len(keep) < len(lines):
                text = "\n".join(keep)
                was_repaired = True

        m = compute_metrics(text)
        if m["n_words"] == 0:
            self.reasons["empty"] += 1
            return False, text, "empty", m

        # Hard rejects: no repair can save these.
        hard = [
            (m["n_words"] < c["min_words"], "too_short"),
            (m["n_words"] > c["max_words"], "too_long"),
            (not c["min_mean_word_len"] <= m["mean_word_len"] <= c["max_mean_word_len"],
             "mean_word_len"),
            (m["symbol_word_ratio"] > c["max_symbol_word_ratio"], "symbol_ratio"),
            (m["alpha_word_ratio"] < c["min_alpha_word_ratio"], "alpha_ratio"),
            (self._stopword_hits(text) < c["min_stopword_hits"], "stopwords"),
        ]
        for failed, reason in hard:
            if failed:
                self.reasons[reason] += 1
                return False, text, reason, m

        # Structural failures: repairable in low-resource mode.
        structural = self._structural_failures(m)

        if structural:
            if self.repair and set(structural) <= self.REPAIRABLE:
                repaired = self._dedupe_lines(text)
                m2 = compute_metrics(repaired)
                # Re-check every structural rule, not just the one that failed:
                # removing duplicate lines moves all the repetition metrics.
                if m2.get("n_words", 0) >= c["min_words"] \
                        and not self._structural_failures(m2):
                    self.reasons["repaired"] += 1
                    return True, repaired, "repaired", m2
            self.reasons[structural[0]] += 1
            return False, text, structural[0], m

        self.reasons["repaired" if was_repaired else "kept"] += 1
        return True, text, "repaired" if was_repaired else "kept", m
