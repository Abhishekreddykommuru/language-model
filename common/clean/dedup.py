"""Deduplication: stage 4 of the Yi pipeline.

Parameters: 3-word shingles, a k=100 MinHash signature, and b=20 bands of r=5,
giving an LSH threshold of 0.55. At that setting two documents of similarity 0.8
are compared with probability 0.9996, and of similarity 0.3 with probability
0.047.
"""
import hashlib
from typing import Iterable, Iterator

from datasketch import MinHash, MinHashLSH


def shingles(text: str, n: int = 3) -> set:
    """Overlapping runs of n consecutive words.

    "the old quarter around the Charminar" with n=3 gives
    {"the old quarter", "old quarter around", "quarter around the", ...}
    """
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def exact_hash(text: str) -> str:
    """SHA-256 of whitespace-collapsed text; catches byte-identical reposts."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def make_minhash(text: str, num_perm: int = 100, n: int = 3) -> MinHash:
    """Build a document's k-permutation MinHash signature."""
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, n):
        m.update(sh.encode("utf-8"))
    return m


def lsh_probability(similarity: float, bands: int = 20, rows: int = 5) -> float:
    """P(two documents share at least one band) = 1 - (1 - s**r)**b."""
    return 1.0 - (1.0 - similarity ** rows) ** bands


class Deduplicator:
    """Exact then near-duplicate deduplication, with a manual-wins policy.

    With keep_policy 'prefer_manual', a manually collected document beats a
    downloaded duplicate, which protects the manual share.
    """

    def __init__(self, cfg: dict):
        self.n = cfg.get("shingle_words", 3)
        self.num_perm = cfg.get("num_perm", 100)
        self.bands = cfg.get("bands", 20)
        self.rows = cfg.get("rows_per_band", 5)
        self.threshold = cfg.get("jaccard_threshold", 0.70)
        self.prefer_manual = cfg.get("keep_policy") == "prefer_manual"
        self.lsh = MinHashLSH(num_perm=self.num_perm, params=(self.bands, self.rows))
        self.seen_exact: dict = {}
        self.sigs: dict = {}
        self.classes: dict = {}
        self.stats = {"exact_dups": 0, "near_dups": 0, "kept": 0,
                      "manual_preferred": 0}

    def add(self, doc_id: str, text: str, source_class: str) -> tuple:
        """Try to add a document. Returns (kept, reason, duplicate_of)."""
        h = exact_hash(text)
        if h in self.seen_exact:
            other = self.seen_exact[h]
            self.stats["exact_dups"] += 1
            if self.prefer_manual and source_class == "manual" \
                    and self.classes.get(other) == "downloaded":
                self.seen_exact[h] = doc_id
                self.classes[doc_id] = source_class
                self.stats["manual_preferred"] += 1
                return True, "exact_dup_replaced_downloaded", other
            return False, "exact_dup", other

        m = make_minhash(text, self.num_perm, self.n)
        for cand in self.lsh.query(m):
            if m.jaccard(self.sigs[cand]) >= self.threshold:
                self.stats["near_dups"] += 1
                if self.prefer_manual and source_class == "manual" \
                        and self.classes.get(cand) == "downloaded":
                    self.stats["manual_preferred"] += 1
                    return True, "near_dup_replaced_downloaded", cand
                return False, "near_dup", cand

        self.lsh.insert(doc_id, m)
        self.sigs[doc_id] = m
        self.seen_exact[h] = doc_id
        self.classes[doc_id] = source_class
        self.stats["kept"] += 1
        return True, "kept", None


class Decontaminator:
    """Removes training documents that duplicate held-out validation/test text.

    Evaluation text that also appears in training makes perplexity look
    excellent and mean nothing. The held-out split is carved out before
    cleaning, indexed here, and every training document checked against it.
    """

    def __init__(self, cfg: dict):
        self.n = cfg.get("shingle_words", 3)
        self.num_perm = cfg.get("num_perm", 100)
        self.bands = cfg.get("bands", 20)
        self.rows = cfg.get("rows_per_band", 5)
        self.threshold = cfg.get("jaccard_threshold", 0.70)
        self.lsh = MinHashLSH(num_perm=self.num_perm, params=(self.bands, self.rows))
        self.sigs: dict = {}
        self.n_held = 0
        self.n_removed = 0

    def index_heldout(self, docs: Iterable[dict]):
        """Index the held-out documents. Call before filtering train."""
        for d in docs:
            m = make_minhash(d["text"], self.num_perm, self.n)
            self.lsh.insert(d["doc_id"], m)
            self.sigs[d["doc_id"]] = m
            self.n_held += 1

    def is_contaminated(self, text: str) -> tuple:
        """Return (contaminated, held_out_doc_id_or_None)."""
        m = make_minhash(text, self.num_perm, self.n)
        for cand in self.lsh.query(m):
            if m.jaccard(self.sigs[cand]) >= self.threshold:
                self.n_removed += 1
                return True, cand
        return False, None


class MinHashClusterer:
    """Memory-bounded near-duplicate clustering over a whole corpus.

    A single in-memory MinHashLSH over millions of documents does not fit in
    Colab's RAM, and sharding by one band's hash is wrong: near-duplicates are
    guaranteed to agree on at least one band but not on any particular one, so
    sharding on band 0 splits duplicate pairs apart and misses them.

    Three passes instead: cache every signature on disk, then group by each band
    in turn and union the collisions, then keep one representative per cluster.
    Memory is bounded by one band's grouping dict plus the union-find array.
    """

    def __init__(self, cfg: dict, cache_path: str):
        self.n = cfg.get("shingle_words", 3)
        self.num_perm = cfg.get("num_perm", 100)
        self.bands = cfg.get("bands", 20)
        self.rows = cfg.get("rows_per_band", 5)
        self.prefer_manual = cfg.get("keep_policy") == "prefer_manual"
        self.cache_path = cache_path
        self.doc_ids: list = []
        self.classes: list = []
        self.lengths: list = []
        self.sources: list = []
        self._sigs: list = []
        self._parent = None

    # ---- pass 1 -----------------------------------------------------------
    def add(self, doc_id: str, text: str, source_class: str,
            source: str = "?"):
        m = make_minhash(text, self.num_perm, self.n)
        self._sigs.append(m.hashvalues.astype("uint64"))
        self.doc_ids.append(doc_id)
        self.classes.append(source_class)
        self.lengths.append(len(text))
        self.sources.append(source)

    def finalize_signatures(self):
        """Write the signature matrix to disk and free the Python list."""
        import numpy as np

        if not self._sigs:
            self._matrix = np.zeros((0, self.num_perm), dtype="uint64")
            return
        matrix = np.vstack(self._sigs)
        np.save(self.cache_path, matrix)
        self._sigs = []
        self._matrix = np.load(self.cache_path, mmap_mode="r")

    # ---- union-find -------------------------------------------------------
    def _find(self, i: int) -> int:
        parent = self._parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]       # path halving
            i = parent[i]
        return int(i)

    def _union(self, a: int, b: int):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    # ---- pass 2 -----------------------------------------------------------
    def cluster(self) -> dict:
        """Union all documents that collide in any band. Returns statistics."""
        import numpy as np

        n = len(self.doc_ids)
        self._parent = np.arange(n, dtype="int64")
        if n == 0:
            return {"documents": 0, "clusters": 0, "duplicates": 0}

        for band in range(self.bands):
            lo, hi = band * self.rows, (band + 1) * self.rows
            groups: dict = {}
            for i in range(n):
                key = self._matrix[i, lo:hi].tobytes()
                first = groups.get(key)
                if first is None:
                    groups[key] = i
                else:
                    self._union(first, i)
            del groups                          # one band resident at a time
            print(f"    band {band + 1}/{self.bands} clustered", flush=True)

        roots = {self._find(i) for i in range(n)}
        return {"documents": n, "clusters": len(roots),
                "duplicates": n - len(roots)}

    # ---- pass 3 -----------------------------------------------------------
    def _better(self, i: int, current: int) -> bool:
        """Is document i a better cluster representative than `current`?

        Manual beats downloaded; otherwise the longer document wins. Outlets
        truncate republished wire stories differently, so keeping the longest
        surviving version preserves the most text.
        """
        if self.prefer_manual and self.classes[i] != self.classes[current]:
            return self.classes[i] == "manual"
        return self.lengths[i] > self.lengths[current]

    def representatives(self) -> set:
        """Choose one doc_id per cluster: manual first, then longest."""
        best: dict = {}
        for i, doc_id in enumerate(self.doc_ids):
            root = self._find(i)
            current = best.get(root)
            if current is None or self._better(i, current):
                best[root] = i
        return {self.doc_ids[i] for i in best.values()}

    def duplication_report(self) -> dict:
        """Who duplicated whom, by source.

        The same wire story republished across dozens of sites is the classic
        failure mode, so the rate is worth measuring rather than assuming.
        """
        from collections import Counter, defaultdict

        winners = {}
        for i in range(len(self.doc_ids)):
            root = self._find(i)
            cur = winners.get(root)
            if cur is None or self._better(i, cur):
                winners[root] = i

        kept = Counter()
        dropped = Counter()
        lost_to = defaultdict(Counter)
        for i, src in enumerate(self.sources):
            root = self._find(i)
            if winners[root] == i:
                kept[src] += 1
            else:
                dropped[src] += 1
                lost_to[src][self.sources[winners[root]]] += 1

        return {
            "kept_by_source": dict(kept),
            "dropped_by_source": dict(dropped),
            "duplicate_of_source": {k: dict(v) for k, v in lost_to.items()},
        }
