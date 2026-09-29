"""Language identification: stage 2 of the Yi pipeline, and the first filter.

The difficulty is wildly asymmetric, which is itself a finding worth reporting.
Telugu has an exclusive Unicode block, so a script test is nearly perfect.
Nepali shares Devanagari with Hindi, Marathi, Maithili, Bhojpuri and Sanskrit,
so it gets an ensemble: a script gate, GlotLID, and a function-word
discriminator for the uncertain middle. GlotLID is less confident on genuine
Nepali precisely because the script is shared, so gating on confidence alone
would delete real data that a low-resource corpus cannot spare.
"""
from pathlib import Path

from common.clean.normalize import SCRIPT_RANGES, latin_ratio, script_ratio

# Neighbouring languages that share Devanagari. A confident vote for one of
# these is a hard reject for the Nepali corpus.
DEVANAGARI_NEIGHBOURS = {"hin", "mar", "mai", "bho", "san", "awa", "doi", "kok"}


class LanguageIdentifier:
    """Script gate + GlotLID + optional function-word discriminator."""

    def __init__(self, cfg: dict):
        """Build an identifier from the `lid` block of clean.yaml."""
        self.cfg = cfg
        self.expected = cfg["expected_label"]
        self.min_conf = cfg["min_confidence"]
        self.min_script = cfg["min_script_ratio"]
        self.max_latin = cfg["max_latin_ratio"]
        self.use_fw = cfg.get("use_function_words", False)
        self.nep_markers = set(cfg.get("nepali_markers", []))
        self.hin_markers = set(cfg.get("hindi_markers", []))
        self.min_adv = cfg.get("min_marker_advantage", 1.5)
        self.model = None
        for path in self._candidate_paths(cfg.get("model")):
            try:
                import fasttext

                # A truncated download is the usual cause of "cannot be
                # opened": the model is 1.7 GB and local disk is wiped between
                # sessions.
                size = Path(path).stat().st_size if Path(path).exists() else 0
                if size < 1_000_000_000:
                    if size:
                        print(f"  GlotLID at {path} is {size / 1e6:.0f} MB, "
                              f"expected ~1700 MB; ignoring truncated file")
                    continue
                self.model = fasttext.load_model(str(path))
                print(f"  GlotLID loaded from {path}")
                break
            except Exception as exc:
                print(f"  GlotLID at {path} unusable ({exc})")
        self._probe_predict()
        if self.model is None:
            print("  ! GlotLID unavailable, falling back to script"
                  + (" and function words" if self.use_fw else " only"))

    @staticmethod
    def _candidate_paths(configured: str) -> list:
        """Where to look for the model, most specific first.

        The Drive copy matters because local disk is wiped between sessions, and
        re-downloading 1.7 GB each time is not worth it.
        """
        from common.io.paths import work_dir

        out = []
        if configured:
            out.append(configured)
        out.append("models/glotlid.bin")
        try:
            out.append(f"{work_dir()}/_models/glotlid.bin")
        except Exception:
            pass
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def _probe_predict(self):
        """Decide once whether fasttext's Python wrapper is usable.

        fasttext's predict() ends in np.array(probs, copy=False). Under NumPy 2
        that means "never copy", and it raises instead of falling back, so every
        call fails. Rather than pin NumPy down for the whole project, detect it
        here and go to the C++ layer directly, which returns plain tuples.
        """
        self._raw_predict = False
        if self.model is None:
            return
        try:
            self.model.predict("test", k=1)
            return
        except ValueError:
            pass
        # Confirm the fallback works before committing to it, rather than
        # raising once per document for the whole corpus.
        try:
            self.model.f.predict("test\n", 1, 0.0, "strict")
            self._raw_predict = True
            print("  fasttext predict() is incompatible with this NumPy; "
                  "using its C++ layer directly")
        except Exception as exc:
            print(f"  ! GlotLID is unusable ({type(exc).__name__}: {exc}); "
                  f"continuing without it")
            self.model = None

    def _glotlid(self, text: str) -> tuple:
        """Return (label, confidence). GlotLID labels look like 'npi_Deva'."""
        if self.model is None:
            return "", 0.0
        sample = text.replace("\n", " ")[:2000]
        if self._raw_predict:
            # The Python wrapper appends the newline the C++ reader expects.
            preds = self.model.f.predict(sample + "\n", 1, 0.0, "strict")
            if not preds:
                return "", 0.0
            prob, label = preds[0]
            return label.replace("__label__", ""), float(prob)
        labels, probs = self.model.predict(sample, k=1)
        return labels[0].replace("__label__", ""), float(probs[0])

    def _marker_advantage(self, text: str) -> float:
        """Nepali marker hits over Hindi marker hits, +1 smoothed.

        छ / छन् / हो are Nepali copulas, गर्ने / गरेको Nepali verb forms, -हरू
        the plural and -को / -ले case markers. The Hindi counterparts
        (है / हैं / नहीं / और) do not occur in Nepali.
        """
        words = text.split()
        nep = sum(1 for w in words if w in self.nep_markers)
        hin = sum(1 for w in words if w in self.hin_markers)
        return (nep + 1) / (hin + 1)

    def accept(self, text: str, lang: str) -> tuple:
        """Decide whether a document is in the target language.

        Returns (accepted, reason, detail).
        """
        lo, hi = SCRIPT_RANGES[lang]
        sr = script_ratio(text, lo, hi)
        lr = latin_ratio(text)
        detail = {"script_ratio": round(sr, 4), "latin_ratio": round(lr, 4)}

        # Gate 1: script. For Telugu this is nearly the whole decision.
        if sr < self.min_script:
            return False, "low_script_ratio", detail
        if lr > self.max_latin:
            return False, "high_latin_ratio", detail

        # Gate 2: GlotLID.
        label, conf = self._glotlid(text)
        detail.update({"lid_label": label, "lid_conf": round(conf, 4)})

        if self.model is None:
            if self.use_fw:
                adv = self._marker_advantage(text)
                detail["marker_advantage"] = round(adv, 3)
                ok = adv >= self.min_adv
                return ok, "function_word_only" if ok else "function_word_reject", detail
            return True, "script_only", detail

        if label == self.expected and conf >= self.min_conf:
            return True, "lid_confident", detail

        if lang == "nep" and label.split("_")[0] in DEVANAGARI_NEIGHBOURS \
                and conf >= 0.70:
            return False, f"confident_other_language:{label}", detail

        # Gate 3: function-word discriminator rescues the uncertain middle.
        if self.use_fw:
            adv = self._marker_advantage(text)
            detail["marker_advantage"] = round(adv, 3)
            if adv >= self.min_adv:
                return True, "function_word_rescue", detail
            return False, "function_word_reject", detail

        return False, "lid_unconfident", detail
