"""Corpus statistics. Computation only; every function returns data and
common/viz.py does the plotting."""
import unicodedata
from collections import Counter, defaultdict

from common.io.shards import read_dir


def corpus_stats(shard_dir: str) -> dict:
    """Aggregate document, character and byte counts by source and class."""
    by_source = defaultdict(lambda: {"docs": 0, "chars": 0, "bytes": 0})
    by_class = defaultdict(lambda: {"docs": 0, "chars": 0, "bytes": 0})
    by_method = defaultdict(lambda: {"docs": 0, "chars": 0, "bytes": 0})
    lengths = []

    for d in read_dir(shard_dir):
        n_chars = d.get("n_chars") or len(d.get("text", ""))
        n_bytes = d.get("n_bytes") or len(d.get("text", "").encode("utf-8"))
        for bucket, key in ((by_source, d.get("source", "?")),
                            (by_class, d.get("source_class", "?")),
                            (by_method, d.get("method", "?"))):
            bucket[key]["docs"] += 1
            bucket[key]["chars"] += n_chars
            bucket[key]["bytes"] += n_bytes
        lengths.append(n_chars)

    return {"by_source": dict(by_source), "by_class": dict(by_class),
            "by_method": dict(by_method), "doc_lengths": lengths}


def manual_fraction(tokens_by_class: dict) -> dict:
    """Manual versus downloaded, by real token count.

    This is what the >=20% requirement is judged on, so it comes from tokenizer
    output rather than a bytes-per-token estimate.
    """
    manual = tokens_by_class.get("manual", 0)
    downloaded = tokens_by_class.get("downloaded", 0)
    total = manual + downloaded
    return {
        "manual_tokens": manual,
        "downloaded_tokens": downloaded,
        "total_tokens": total,
        "manual_fraction": (manual / total) if total else 0.0,
        "meets_requirement": bool(total) and (manual / total) >= 0.20,
    }


def downloaded_cap_for_manual(manual_tokens: int, min_fraction: float = 0.20) -> int:
    """Most downloaded tokens that still leaves manual >= min_fraction.

    At min_fraction 0.20 this is 4x the manual token count. Stage 07 treats it
    as a floor to respect where it can, not a hard ceiling; see 07_finalize.py.
    """
    if min_fraction <= 0 or min_fraction >= 1:
        raise ValueError("min_fraction must be in (0, 1)")
    return int(manual_tokens * (1.0 - min_fraction) / min_fraction)


def attrition_table(counters: list) -> list:
    """Documents surviving each pipeline stage, as counts and percentages."""
    rows, prev = [], None
    first = counters[0]["docs"] if counters and counters[0]["docs"] else 1
    for c in counters:
        row = dict(c)
        row["pct_of_raw"] = c["docs"] / first
        row["pct_of_prev"] = (c["docs"] / prev["docs"]) if prev and prev["docs"] else 1.0
        rows.append(row)
        prev = c
    return rows


def script_composition(shard_dir: str, sample: int = 20_000) -> dict:
    """Character-level script mix: evidence the corpus is monolingual."""
    counts = Counter()
    for i, d in enumerate(read_dir(shard_dir)):
        if i >= sample:
            break
        for ch in d.get("text", "")[:2000]:
            counts[unicodedata.name(ch, "?").split()[0]] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.most_common(12)}


def zipf_data(freq: dict) -> tuple:
    """Return (ranks, frequencies) sorted for a log-log Zipf plot."""
    vals = sorted(freq.values(), reverse=True)
    return list(range(1, len(vals) + 1)), vals
