"""HTML to article text, plus a site-wide boilerplate line detector.

Boilerplate is the main cause of degenerate repetition in a small model.
Trafilatura does the extraction; a line-frequency filter removes what it misses.
"""
from collections import Counter

import trafilatura


def extract_article(html: str, url: str = "") -> dict:
    """Return {"text", "title", "date"}, or {} if nothing was extracted.

    favor_precision drops doubtful blocks: a missed paragraph costs little, a
    retained navigation menu repeats on every page of the site.
    """
    text = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text:
        return {}
    meta = {}
    try:
        md = trafilatura.extract_metadata(html)
        if md:
            meta = {"title": md.title or "", "date": md.date or ""}
    except Exception:
        pass
    return {"text": text, **meta}


class BoilerplateDetector:
    """Finds lines repeated across a site's documents so they can be stripped.

    A short line appearing in more than `doc_freq` of a site's documents is
    furniture ("Follow us on", "తాజా వార్తలు", "सम्बन्धित समाचार"), not content.
    Two passes: count() over the corpus, then boilerplate_lines().
    """

    #: Lines longer than this are treated as prose, never as furniture.
    MAX_FURNITURE_CHARS = 200

    def __init__(self, doc_freq: float = 0.001, min_docs: int = 500):
        self.doc_freq = doc_freq
        self.min_docs = min_docs
        self.counts = Counter()
        self.n_docs = 0

    def count(self, text: str):
        self.n_docs += 1
        for line in {l.strip() for l in text.split("\n") if l.strip()}:
            if len(line) < self.MAX_FURNITURE_CHARS:
                self.counts[line] += 1

    def boilerplate_lines(self) -> set:
        if self.n_docs < self.min_docs:
            return set()
        cutoff = max(3, int(self.n_docs * self.doc_freq))
        return {line for line, c in self.counts.items() if c >= cutoff}
