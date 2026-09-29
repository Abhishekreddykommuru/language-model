"""BPE, WordPiece and Unigram implemented from first principles.

Libraries are permitted for Phase 1 -- the rule is no *pretrained* tokenizers --
so these exist to show the algorithms are understood, and to be checked against
SentencePiece on the same sample. Production runs use the library path, chosen
per algorithm by the `engine` setting in the tokenizer configs, because these
pure-Python versions are about two orders of magnitude slower.

    BPE        merge the most frequent adjacent pair, num_merges times
    WordPiece  merge the highest freq(pair) / (freq(a) * freq(b)), which
               prefers pairs whose parts are individually rare
    Unigram    seed a large vocabulary and prune the pieces whose removal costs
               the least likelihood, by EM with Viterbi segmentation
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

#: Marks a word-initial position, following the SentencePiece convention. Using
#: the same marker means our vocabularies are directly comparable with theirs.
WORD_START = "▁"

#: Marks a continuation piece, following the WordPiece/BERT convention.
CONTINUING_PREFIX = "##"


def word_frequencies(lines, lowercase: bool = False) -> Counter:
    """Count whitespace-delimited words across a corpus.

    All three algorithms operate on word types weighted by frequency rather than
    on the raw character stream, which is what makes them tractable: a corpus of
    millions of tokens usually has only a few hundred thousand distinct words.
    """
    freq = Counter()
    for line in lines:
        if lowercase:
            line = line.lower()
        for word in line.split():
            freq[word] += 1
    return freq


class ScratchBPE:
    """Byte-Pair Encoding.

    Training repeatedly merges the most frequent adjacent symbol pair. The number
    of merges is the real hyperparameter; vocabulary size is a consequence of it:

        vocab_size = len(alphabet) + len(special_tokens) + num_merges

    so we derive num_merges from the requested vocab_size and record BOTH in the
    saved model, since the merge list is what actually defines the tokenizer.
    """

    def __init__(self, vocab_size: int, special_tokens: list = None,
                 max_piece_length: int = 16):
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens or ["<unk>", "<s>", "</s>", "<pad>"])
        self.max_piece_length = max_piece_length
        self.merges: list = []          # ordered list of (a, b) pairs
        self.vocab: dict = {}           # piece -> id
        self.alphabet: list = []
        self.num_merges = 0

    @staticmethod
    def _split_word(word: str) -> tuple:
        """Represent a word as a tuple of symbols, marking the word start."""
        return (WORD_START + word[0],) + tuple(word[1:]) if word else ()

    @staticmethod
    def _count_pairs(splits: dict, freq: Counter) -> Counter:
        """Count adjacent symbol pairs, weighted by word frequency."""
        pairs = Counter()
        for word, symbols in splits.items():
            f = freq[word]
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += f
        return pairs

    @staticmethod
    def _apply_merge(splits: dict, pair: tuple) -> dict:
        """Replace every occurrence of `pair` with the merged symbol."""
        a, b = pair
        merged = a + b
        out = {}
        for word, symbols in splits.items():
            if len(symbols) < 2:
                out[word] = symbols
                continue
            new, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new.append(merged)
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            out[word] = tuple(new)
        return out

    def train(self, lines, verbose: bool = True) -> "ScratchBPE":
        """Learn the merge list from a corpus."""
        freq = word_frequencies(lines)
        splits = {w: self._split_word(w) for w in freq}

        self.alphabet = sorted({s for symbols in splits.values() for s in symbols})
        self.num_merges = max(
            0, self.vocab_size - len(self.alphabet) - len(self.special_tokens))
        if verbose:
            print(f"    BPE: {len(freq):,} word types, "
                  f"alphabet={len(self.alphabet)}, num_merges={self.num_merges:,}")

        for step in range(self.num_merges):
            pairs = self._count_pairs(splits, freq)
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:               # nothing left worth merging
                break
            if len(a) + len(b) > self.max_piece_length:
                # Skip over-long candidates rather than stopping: a single long
                # pair should not end training early.
                for (ca, cb), _ in pairs.most_common():
                    if len(ca) + len(cb) <= self.max_piece_length:
                        a, b = ca, cb
                        break
                else:
                    break
            splits = self._apply_merge(splits, (a, b))
            self.merges.append((a, b))
            if verbose and (step + 1) % 2000 == 0:
                print(f"      merge {step + 1:,}/{self.num_merges:,}", flush=True)

        pieces = self.special_tokens + self.alphabet + \
            [a + b for a, b in self.merges]
        seen, ordered = set(), []
        for p in pieces:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        self.vocab = {p: i for i, p in enumerate(ordered)}
        return self

    def encode_word(self, word: str) -> list:
        """Segment one word by replaying the merge list in learned order."""
        symbols = list(self._split_word(word))
        for a, b in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i:i + 2] = [a + b]
                else:
                    i += 1
        return symbols

    def encode(self, text: str) -> list:
        out = []
        for word in text.split():
            out.extend(self.encode_word(word))
        return out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"algorithm": "bpe", "vocab_size": len(self.vocab),
                   "num_merges": self.num_merges,
                   "alphabet_size": len(self.alphabet),
                   "special_tokens": self.special_tokens,
                   "merges": [list(m) for m in self.merges],
                   "vocab": self.vocab},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)


class ScratchWordPiece:
    """WordPiece: BPE's machinery with a different merge criterion.

        score(a, b) = freq(a, b) / (freq(a) * freq(b))

    Dividing by the parts' own frequencies merges a pair only when its halves
    co-occur far more than their individual frequencies predict, which
    suppresses merges that are frequent only because their components are.
    """

    def __init__(self, vocab_size: int, special_tokens: list = None,
                 max_piece_length: int = 16, min_frequency: int = None):
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens or ["[UNK]", "[BOS]", "[EOS]", "[PAD]"])
        self.max_piece_length = max_piece_length
        #: The critical hyperparameter here. A pair occurring once whose parts
        #: occur only there scores 1.0, the formula's maximum, so without a
        #: frequency floor the whole merge budget goes to the rarest strings in
        #: the corpus. Measured on 50,455 words of Telugu web text: at 2 the
        #: first merges are ##**, ##•ఱ, TUF and chars/token stays at 1.16; at
        #: 100 they are 20, ఒక, కొ, వెళ and chars/token reaches 2.03; at 300 the
        #: vocabulary cannot be filled. None derives it from corpus size.
        self.min_frequency = min_frequency
        self.vocab: dict = {}
        self.alphabet: list = []
        self.merged_pieces: list = []
        self.num_merges = 0

    @staticmethod
    def _split_word(word: str) -> tuple:
        """First character bare, every later character prefixed with '##'."""
        if not word:
            return ()
        return (word[0],) + tuple(CONTINUING_PREFIX + c for c in word[1:])

    @staticmethod
    def _merge_symbols(a: str, b: str) -> str:
        return a + (b[len(CONTINUING_PREFIX):] if b.startswith(CONTINUING_PREFIX) else b)

    def train(self, lines, verbose: bool = True) -> "ScratchWordPiece":
        freq = word_frequencies(lines)
        splits = {w: self._split_word(w) for w in freq}
        self.alphabet = sorted({s for symbols in splits.values() for s in symbols})
        self.num_merges = max(
            0, self.vocab_size - len(self.alphabet) - len(self.special_tokens))

        if self.min_frequency is None:
            # Roughly 0.2% of corpus size, which reproduced the measured sweet
            # spot above (50,455 words -> 100), floored at 2 and capped so that
            # a very large corpus does not raise the bar past useful pairs.
            total_words = sum(freq.values())
            self.min_frequency = max(2, min(5000, total_words // 500))
        if verbose:
            print(f"    WordPiece: {len(freq):,} word types, "
                  f"alphabet={len(self.alphabet)}, num_merges={self.num_merges:,}, "
                  f"min_frequency={self.min_frequency}")

        for step in range(self.num_merges):
            sym_freq = Counter()
            pair_freq = Counter()
            for word, symbols in splits.items():
                f = freq[word]
                for s in symbols:
                    sym_freq[s] += f
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i + 1])] += f
            if not pair_freq:
                break

            best, best_score = None, -1.0
            for (a, b), f in pair_freq.items():
                if f < self.min_frequency:
                    continue
                if len(self._merge_symbols(a, b)) > self.max_piece_length:
                    continue
                score = f / (sym_freq[a] * sym_freq[b])
                if score > best_score:
                    best, best_score = (a, b), score
            if best is None:
                break

            a, b = best
            merged = self._merge_symbols(a, b)
            self.merged_pieces.append(merged)
            new_splits = {}
            for word, symbols in splits.items():
                out, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                new_splits[word] = tuple(out)
            splits = new_splits
            if verbose and (step + 1) % 1000 == 0:
                print(f"      merge {step + 1:,}/{self.num_merges:,}", flush=True)

        # Vocabulary = specials + alphabet + every piece created by a merge, in
        # creation order. Building it from the merge history rather than from the
        # final splits guarantees that intermediate pieces stay available to the
        # greedy longest-match encoder.
        pieces = self.special_tokens + self.alphabet + self.merged_pieces
        seen, ordered = set(), []
        for p in pieces:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        self.vocab = {p: i for i, p in enumerate(ordered)}
        return self

    def encode_word(self, word: str) -> list:
        """Greedy longest-match from the left.

        If any position cannot be matched the WHOLE word becomes [UNK]. That is
        genuine WordPiece behaviour and differs from BPE, which falls back to
        individual characters -- it is why the two report different UNK rates.
        """
        tokens, start = [], 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = CONTINUING_PREFIX + candidate
                if candidate in self.vocab:
                    piece = candidate
                    break
                end -= 1
            if piece is None:
                return [self.special_tokens[0]]
            tokens.append(piece)
            start = end
        return tokens

    def encode(self, text: str) -> list:
        out = []
        for word in text.split():
            out.extend(self.encode_word(word))
        return out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"algorithm": "wordpiece", "vocab_size": len(self.vocab),
                   "num_merges": self.num_merges,
                   "alphabet_size": len(self.alphabet),
                   "continuing_prefix": CONTINUING_PREFIX,
                   "special_tokens": self.special_tokens,
                   "vocab": self.vocab},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)


class ScratchUnigram:
    """Unigram language-model tokenizer.

    Where BPE and WordPiece grow a vocabulary by merging, Unigram seeds a large
    one and shrinks it. Each piece carries a probability, and a word's
    segmentation is the one maximising the product of its pieces, found by
    Viterbi. Each round: E-step Viterbi-segments every word and accumulates
    expected counts, M-step re-estimates the probabilities, then the cheapest
    pieces by likelihood contribution are pruned, until the target size.

    Single characters are never pruned, so every string stays segmentable and
    the UNK rate stays at zero.
    """

    def __init__(self, vocab_size: int, special_tokens: list = None,
                 max_piece_length: int = 16, seed_size: int = 200_000,
                 shrinking_factor: float = 0.75, em_iterations: int = 2):
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens or ["<unk>", "<s>", "</s>", "<pad>"])
        self.max_piece_length = max_piece_length
        self.seed_size = seed_size
        self.shrinking_factor = shrinking_factor
        self.em_iterations = em_iterations
        self.piece_logp: dict = {}
        self.vocab: dict = {}

    def _seed_vocabulary(self, freq: Counter) -> dict:
        """Frequent substrings, plus every single character (never pruned)."""
        sub = Counter()
        chars = set()
        for word, f in freq.items():
            marked = WORD_START + word
            chars.update(marked)
            n = len(marked)
            for i in range(n):
                for j in range(i + 1, min(i + self.max_piece_length, n) + 1):
                    sub[marked[i:j]] += f
        for c in chars:
            sub[c] += 1  # ensure presence
        keep = dict(sub.most_common(self.seed_size))
        for c in chars:
            keep.setdefault(c, 1)
        total = sum(keep.values())
        return {p: math.log(c / total) for p, c in keep.items()}

    def _viterbi(self, word: str) -> list:
        """Best segmentation of one word under the current piece log-probs."""
        marked = WORD_START + word
        n = len(marked)
        NEG = -1e10
        best = [NEG] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_piece_length), i):
                piece = marked[j:i]
                lp = self.piece_logp.get(piece)
                if lp is None or best[j] <= NEG:
                    continue
                score = best[j] + lp
                if score > best[i]:
                    best[i] = score
                    back[i] = j
        if best[n] <= NEG:
            return list(marked)     # unreachable in practice: chars are kept
        pieces, i = [], n
        while i > 0:
            j = back[i]
            pieces.append(marked[j:i])
            i = j
        return pieces[::-1]

    def train(self, lines, verbose: bool = True) -> "ScratchUnigram":
        freq = word_frequencies(lines)
        self.piece_logp = self._seed_vocabulary(freq)
        single_chars = {p for p in self.piece_logp if len(p) == 1}
        target = max(1, self.vocab_size - len(self.special_tokens))
        if verbose:
            print(f"    Unigram: {len(freq):,} word types, "
                  f"seed vocabulary={len(self.piece_logp):,}, target={target:,}")

        round_no = 0
        while len(self.piece_logp) > target:
            round_no += 1
            # --- EM ---
            for _ in range(self.em_iterations):
                counts = Counter()
                for word, f in freq.items():
                    for piece in self._viterbi(word):
                        counts[piece] += f
                total = sum(counts.values()) or 1
                self.piece_logp = {
                    p: math.log(max(counts.get(p, 0), 1e-9) / total)
                    for p in self.piece_logp
                }

            # --- prune ---
            # A piece's value is (expected frequency x its log-probability): the
            # total likelihood contributed. Cheapest contributors go first.
            counts = Counter()
            for word, f in freq.items():
                for piece in self._viterbi(word):
                    counts[piece] += f
            scored = [(counts.get(p, 0) * -lp, p)
                      for p, lp in self.piece_logp.items()
                      if p not in single_chars]
            scored.sort()
            n_keep = max(target, int(len(self.piece_logp) * self.shrinking_factor))
            n_drop = max(0, len(self.piece_logp) - n_keep)
            for _, piece in scored[:n_drop]:
                self.piece_logp.pop(piece, None)
            if verbose:
                print(f"      prune round {round_no}: "
                      f"{len(self.piece_logp):,} pieces remain", flush=True)
            if n_drop == 0:
                break

        ordered = self.special_tokens + [
            p for p, _ in sorted(self.piece_logp.items(), key=lambda kv: -kv[1])]
        self.vocab = {p: i for i, p in enumerate(ordered)}
        return self

    def encode(self, text: str) -> list:
        out = []
        for word in text.split():
            out.extend(self._viterbi(word))
        return out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"algorithm": "unigram", "vocab_size": len(self.vocab),
                   "seed_size": self.seed_size,
                   "shrinking_factor": self.shrinking_factor,
                   "em_iterations": self.em_iterations,
                   "special_tokens": self.special_tokens,
                   "piece_logp": self.piece_logp,
                   "vocab": self.vocab},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)


ALGORITHMS = {"bpe": ScratchBPE, "wordpiece": ScratchWordPiece,
              "unigram": ScratchUnigram}


def train_scratch(algorithm: str, lines, vocab_size: int, cfg: dict,
                  out_path: str, verbose: bool = True):
    """Train one from-scratch tokenizer and save it. Returns the trained object."""
    kwargs = {"vocab_size": vocab_size,
              "max_piece_length": cfg.get("max_piece_length", 16)}
    if algorithm == "wordpiece":
        kwargs["min_frequency"] = cfg.get("min_frequency")
    if algorithm == "unigram":
        kwargs.update(seed_size=cfg.get("seed_size", 200_000),
                      shrinking_factor=cfg.get("shrinking_factor", 0.75),
                      em_iterations=cfg.get("em_iterations", 2))
    model = ALGORITHMS[algorithm](**kwargs)
    model.train(lines, verbose=verbose)
    model.save(out_path)
    return model
