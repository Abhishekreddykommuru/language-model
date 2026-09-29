"""Stage 6: pull the deliverables off Drive and write the report.

Graders read the branch, not the Drive folder, so anything that should be
marked has to exist as a file in the repository. This copies the tokenizer and
vocabulary files back from Drive, writes report/phase1_report.md from the
numbers the pipeline actually produced, and fills in the README.

Usage:
    python scripts/12_package.py
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, millions, section

LANGS = [("tel", "Telugu", "Model H, higher-resource"),
         ("nep", "Nepali", "Model L, lower-resource")]

#: Split manifests are copied into the repo only below this size. They are the
#: split definitions, which the spec asks for, but a multi-million-line TSV does
#: not belong in git.
MAX_MANIFEST_MB = 20


def load(path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def copy_tokenizer(work: str, lang: str) -> list:
    """Copy tokenizer model, vocabulary and sweep results into the repo."""
    src = Path(f"{work}/{lang_dir(lang)}/tokenizer")
    dst = Path(f"{lang_dir(lang)}/tokenizer")
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in (f"{lang}.model", f"{lang}.json", f"{lang}.vocab.txt",
                    "chosen.json", "sweep_results.json",
                    "scratch_validation.json"):
        f = src / pattern
        if f.exists():
            shutil.copyfile(f, dst / f.name)
            copied.append(f"{dst / f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    return copied


def copy_manifest(work: str, lang: str) -> str:
    """Copy the split manifest, gzipped, if it is small enough to belong in git."""
    src = Path(f"{work}/{lang_dir(lang)}/2_splits/split_manifest.tsv")
    if not src.exists():
        return ""
    dst = Path(f"{lang_dir(lang)}/splits/split_manifest.tsv.gz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    mb = dst.stat().st_size / 1e6
    if mb > MAX_MANIFEST_MB:
        dst.unlink()
        return f"manifest is {mb:.0f} MB, left on Drive"
    return f"{dst} ({mb:.1f} MB)"


def corpus_section(work: str, lang: str, name: str, tier: str) -> list:
    """The per-language part of the report, built from the pipeline's own logs."""
    rep = load(f"{work}/{lang_dir(lang)}/6_tokenized/token_report.json")
    fin = load(f"{work}/_logs/finalize_{lang}.json")
    ded = load(f"{work}/_logs/dedup_{lang}.json")
    cln = load(f"{work}/_logs/clean_{lang}_train.json")
    chosen = load(f"{work}/{lang_dir(lang)}/tokenizer/chosen.json")
    sweep = load(f"{work}/{lang_dir(lang)}/tokenizer/sweep_results.json")
    cfg = yaml.safe_load(
        open(f"{lang_dir(lang)}/configs/sources.yaml", encoding="utf-8"))

    out = [f"## {name} ({tier})", ""]
    if not rep.get("train"):
        return out + ["_Stage 09 has not run for this language yet._", ""]

    tr = rep["train"]
    cls = tr.get("tokens_by_class", {})
    man, dl = cls.get("manual", 0), cls.get("downloaded", 0)
    total = man + dl
    frac = rep.get("manual_fraction_train", 0.0)

    out += [
        "### Corpus size and the manual fraction", "",
        "Counted with the trained tokenizer, not estimated from bytes.", "",
        "| | tokens | share |", "|---|---:|---:|",
        f"| Manual (own crawl, OCR, PDFs) | {man:,} | **{frac:.1%}** |",
        f"| Downloaded (public corpora) | {dl:,} | {1 - frac:.1%} |",
        f"| **Total, train split** | **{total:,}** | |", "",
        f"Target was ~500M tokens with at least 20% manual. "
        f"This corpus reaches **{millions(total)} tokens** at "
        f"**{frac:.1%} manual**.", "",
        "| split | documents | tokens | bytes of text |", "|---|---:|---:|---:|",
    ]
    for sp in ("train", "val", "test"):
        if sp in rep:
            r = rep[sp]
            out.append(f"| {sp} | {r['documents']:,} | {r['tokens']:,} | "
                       f"{r['bytes_of_text']:,} |")
    out += ["", "### Where the tokens came from", "",
            "| source | tokens | share |", "|---|---:|---:|"]
    for src, n in sorted(tr.get("tokens_by_source", {}).items(),
                         key=lambda kv: -kv[1]):
        out.append(f"| {src} | {n:,} | {n / max(total, 1):.1%} |")

    out += ["", "### Cleaning", "",
            "![attrition](figures/attrition_%s.png)" % lang, ""]
    if cln:
        c = cln.get("counters", {})
        out += [
            f"Language filtering and the quality rules kept "
            f"**{cln.get('survival', 0):.1%}** of the documents entering "
            f"stage 05 ({c.get('out', 0):,} of {c.get('in', 0):,}).", "",
        ]
        rows = [(k, v) for k, v in c.items()
                if k.startswith(("lid_", "quality_"))
                and k not in ("lid_pass", "lid_rejected", "quality_rejected")]
        if rows:
            out += ["| rejection reason | documents |", "|---|---:|"]
            for k, v in sorted(rows, key=lambda kv: -kv[1])[:12]:
                out.append(f"| {k} | {v:,} |")
        out.append("")
    if ded and ded.get("method") == "exact_only":
        out += [
            "### Deduplication", "",
            f"**Exact deduplication only.** {ded.get('exact_dup', 0):,} "
            f"byte-identical documents were removed, along with "
            f"{ded.get('decontaminated', 0):,} training documents whose text "
            f"also appears in the held-out splits.", "",
            "MinHash near-duplicate removal was **not** applied to this "
            "language: the full pass did not fit the available compute budget. "
            "The implementation is the same one used for the other language "
            "(3-word shingles, k=100, b=20 bands of r=5, LSH threshold 0.55) "
            "and is in `common/clean/dedup.py`; only the run is missing.", "",
        ]
    elif ded:
        out += [
            "### Deduplication", "",
            f"3-word shingles, k=100 MinHash, b=20 bands of r=5, so an LSH "
            f"threshold of 0.55. Removed {ded.get('exact_dup', 0):,} exact "
            f"duplicates and {ded.get('duplicate_dropped', 0):,} "
            f"near-duplicates, and dropped {ded.get('decontaminated', 0):,} "
            f"training documents that overlapped the held-out splits.", "",
        ]
        dups = ded.get("duplication_by_source", {}).get("dropped_by_source", {})
        kept = ded.get("duplication_by_source", {}).get("kept_by_source", {})
        if dups:
            out += ["| source | dropped as duplicate | of | rate |",
                    "|---|---:|---:|---:|"]
            for src, n in sorted(dups.items(), key=lambda kv: -kv[1])[:10]:
                tot = kept.get(src, 0) + n
                out.append(f"| {src} | {n:,} | {tot:,} | {n / max(tot, 1):.1%} |")
            out.append("")

    if fin:
        out += ["### Corpus sizing", "",
                f"Stage 07 aims for {millions(fin.get('target_tokens', 0))} "
                f"tokens and relaxes the 20% manual cap only as far as reaching "
                f"that requires. Counters: `{fin.get('counters', {})}`.", ""]

    ch = chosen.get("chosen", {})
    fm = chosen.get("final_metrics", {})
    out += [
        "### Tokenizer", "",
        (f"The sweep covered "
         f"{', '.join(sorted({r['algorithm'] for r in sweep}))} at "
         f"{', '.join(f'{v:,}' for v in sorted({r['vocab_size'] for r in sweep}))}"
         f", each evaluated on held-out text; "
         f"**{ch.get('algorithm', '?')}** at **{ch.get('vocab_size', 0):,}** "
         f"was selected."
         if sweep else "Tokenizer selection details are in `chosen.json`."), "",
        "![fertility](figures/fertility_%s.png)" % lang, "",
        "| metric | value |", "|---|---:|",
        f"| Vocabulary size | {rep.get('vocab_size', 0):,} |",
        f"| Average characters per token | {fm.get('chars_per_token', 0):.3f} |",
        f"| Bytes per token | {fm.get('bytes_per_token', 0):.3f} |",
        f"| Fertility, tokens per word | {fm.get('tokens_per_word', 0):.3f} |",
        f"| Unknown-token rate | {fm.get('unk_rate', 0):.6f} |",
        f"| Byte-fallback rate | {fm.get('byte_fallback_rate', 0):.6f} |",
        f"| Vocabulary seen in held-out text | {fm.get('vocab_used_frac', 0):.1%} |",
        "",
    ]
    if sweep:
        out += ["Full sweep:", "",
                "| algorithm | vocab | chars/token | tokens/word | unk rate | "
                "byte fallback | Phase 2 embedding share |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sweep:
            out.append(
                f"| {r['algorithm']} | {r['vocab_size']:,} | "
                f"{r['chars_per_token']:.3f} | {r['tokens_per_word']:.3f} | "
                f"{r['unk_rate']:.6f} | {r['byte_fallback_rate']:.6f} | "
                f"{r['phase2']['embedding_share']:.1%} |")
        out.append("")

    tf = rep.get("token_frequency", {})
    if tf:
        out += [
            "#### Token-frequency statistics", "",
            "![zipf](figures/zipf_%s.png)" % lang, "",
            f"{tf['distinct_tokens_used']:,} of {rep.get('vocab_size', 0):,} "
            f"pieces appear in the training corpus "
            f"({tf['vocab_coverage']:.1%}); {tf['unused_vocab']:,} never occur "
            f"and are dead embedding parameters in Phase 2.", "",
            "| rank | piece | count | share |", "|---:|---|---:|---:|",
        ]
        for i, r in enumerate(tf.get("top_50", [])[:20], 1):
            out.append(f"| {i} | `{r['piece']}` | {r['count']:,} | "
                       f"{r['count'] / max(total, 1):.2%} |")
        out.append("")

    if chosen.get("examples"):
        out += ["#### Tokenization examples", "",
                "| word | pieces | count |", "|---|---|---:|"]
        for ex in chosen["examples"]:
            pieces = " ".join(f"`{p}`" for p in ex["pieces"])
            out.append(f"| {ex['word']} | {pieces} | {ex['n_pieces']} |")
        out.append("")

    out += ["### Sources", "", "| source | class | licence |", "|---|---|---|"]
    for d in cfg["downloaded"]:
        out.append(f"| {d['name']} | downloaded | {d.get('licence', '')} |")
    for s in cfg["crawl"]["sites"]:
        out.append(f"| {s['host']} | manual, own crawl | site terms, academic use |")
    out += ["| archive.org scanned books | manual, own OCR | item licence |",
            "| government PDFs | manual, text layer | government publication |",
            ""]
    excluded = cfg.get("excluded_hosts") or []
    if excluded:
        out += [f"Excluded on ethical grounds, since they declare "
                f"`Content-Signal: ai-train=no`: "
                + ", ".join(f"`{h}`" for h in excluded) + ".", ""]
    out += ["![sources](figures/sources_%s.png)" % lang, "",
            "![document lengths](figures/doclen_%s.png)" % lang, "",
            "![manual split](figures/manual_split_%s.png)" % lang, ""]
    return out


def write_report(work: str, out_path: str):
    doc = [
        "# Phase 1 — Data Collection and Tokenizer Construction", "",
        "Telugu as Model H and Nepali as Model L. The two corpora share no "
        "documents, no vocabulary and no tokenizer: every stage runs "
        "per-language, from separate configs, into separate directories.", "",
        "## Language choice", "",
        "**Telugu (Model H).** A non-English Indian language with abundant "
        "public text, and an exclusive Unicode block, which makes language "
        "identification nearly exact and keeps the corpus clean.", "",
        "**Nepali (Model L).** On the permitted lower-resource list. It shares "
        "the Devanagari script with Hindi, Marathi, Maithili and others, so "
        "language identification is genuinely hard and needed a purpose-built "
        "ensemble; that contrast with Telugu is the most interesting finding "
        "in this phase.", "",
        "## Pipeline", "",
        "```",
        "01 download      public corpora from HuggingFace",
        "02 crawl         our own polite crawler, robots.txt and Crawl-delay",
        "02b extract      HTML to article text, two-pass boilerplate removal",
        "03 OCR           scanned books from archive.org, Tesseract",
        "03b PDF text     text-layer government PDFs, pypdf",
        "04 split         val/test held out BEFORE cleaning",
        "05 clean         language id, Unicode normalisation, quality rules, PII",
        "06 dedup         exact + MinHash near-duplicate removal, decontamination",
        "07 finalize      blocklists, source-mix capping, corpus sizing",
        "08 tokenizers    all three algorithms, swept over vocabulary sizes",
        "09 tokenize      encode to uint16, exact token counts",
        "10 stats         tables and figures",
        "11 dataset card  written to the Drive root",
        "```", "",
        "Validation and test are carved out at stage 04, before any cleaning, "
        "so no cleaning decision can leak across the split boundary. Stage 06 "
        "then removes any training document that duplicates held-out text.", "",
        "Deduplication runs over every source at once rather than per source, "
        "so a crawled page that also sits inside a downloaded corpus is caught; "
        "where the two collide the manually collected copy is the one kept.", "",
    ]
    for code, name, tier in LANGS:
        doc += corpus_section(work, code, name, tier)

    doc += [
        "## Filter audit", "",
        "`scripts/audit_filters.py` prints rejected documents grouped by "
        "reason. Reading them already caught one destructive rule: "
        "`no_terminal_punct` was removing 70% of good Telugu articles, because "
        "a single-paragraph news article is one line and does not always end "
        "in a full stop. Line-structure rules now apply only to documents of "
        "five lines or more, and Telugu survival went from 64.6% to 92.2%.", "",
        "_Add your own judgements on the sampled rejections here._", "",
        "## Shortfall against 500M", "",
        "_Fill in from the numbers above._ The relevant causes: several Telugu "
        "hosts return nothing to datacenter IP ranges; five of seven Nepali "
        "hosts publish no sitemap at all and three declare a Crawl-delay of "
        "10-20 seconds, which was honoured. The spec allows a justified "
        "shortfall for the lower-resource language.", "",
        "## Ethics", "",
        "robots.txt is obeyed, including Crawl-delay. Hosts declaring "
        "`Content-Signal: ai-train=no` are excluded by name and listed above. "
        "Requests carry a descriptive User-Agent with a contact address and are "
        "rate-limited to about two per second per host. Only extracted text and "
        "its provenance are published; the raw crawled HTML is not "
        "redistributed. Email addresses and phone numbers are replaced with "
        "`<EMAIL>` and `<PHONE>` placeholders.", "",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(doc), encoding="utf-8")


def write_readme(work: str):
    rows = []
    for code, name, tier in LANGS:
        rep = load(f"{work}/{lang_dir(code)}/6_tokenized/token_report.json")
        if rep.get("train"):
            rows.append(f"| {name} | {rep['train']['tokens']:,} | "
                        f"{rep.get('manual_fraction_train', 0):.1%} | "
                        f"{rep.get('algorithm', '?')} | "
                        f"{rep.get('vocab_size', 0):,} |")
        else:
            rows.append(f"| {name} | not yet built | | | |")

    readme = [
        "# Monolingual Language Models — Telugu (Model H) and Nepali (Model L)", "",
        "Two fully independent monolingual corpora and tokenizers. No shared "
        "documents, vocabulary or tokenizer.", "",
        "| language | training tokens | manual share | tokenizer | vocabulary |",
        "|---|---:|---:|---|---:|",
        *rows, "",
        "The Phase 1 report, with every table and figure, is "
        "[`report/phase1_report.md`](report/phase1_report.md).", "",
        "## Google Drive links", "",
        "_Set each to “Anyone with the link → Viewer” and verify in a private "
        "window while signed out._", "",
        "| artifact | link |", "|---|---|",
        "| Telugu corpus (`telugu/5_final`) | _paste link_ |",
        "| Telugu tokenized (`telugu/6_tokenized`) | _paste link_ |",
        "| Nepali corpus (`nepali/5_final`) | _paste link_ |",
        "| Nepali tokenized (`nepali/6_tokenized`) | _paste link_ |", "",
        "The raw crawled HTML is deliberately **not** shared: it is "
        "copyrighted news content that we are not entitled to redistribute.", "",
        "## Repository layout", "",
        "```",
        "common/          shared library: collection, cleaning, tokenizers, stats",
        "scripts/         numbered pipeline stages, run in order",
        "telugu/          configs, tokenizer and vocabulary for Model H",
        "nepali/          configs, tokenizer and vocabulary for Model L",
        "report/          the Phase 1 report, figures and tables",
        "```", "",
        "## Reproducing", "",
        "```bash",
        "pip install -r requirements.txt      # or: bash setup_colab.sh",
        "python scripts/run_all.py            # every stage, both languages",
        "```", "",
        "Individual stages, and a progress report that is safe to run at any "
        "time:", "",
        "```bash",
        "python scripts/run_all.py --only 06,07 --lang tel",
        "python scripts/status.py",
        "```", "",
        "Every stage is checkpointed, so a lost session resumes rather than "
        "restarting. Stages 04 to 07 clear their own output first, so they are "
        "safe to re-run and always give identical counts.", "",
    ]
    Path("README.md").write_text("\n".join(readme), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default="report/phase1_report.md")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()

    section("Packaging the deliverables into the repository")
    for code, name, _tier in LANGS:
        for line in copy_tokenizer(work, code):
            kv(f"{code} tokenizer", line)
        manifest = copy_manifest(work, code)
        if manifest:
            kv(f"{code} split manifest", manifest)

    write_report(work, args.out)
    kv("report", args.out)
    if Path("README.md").exists():
        kv("readme", "left alone; delete it first to regenerate")
    else:
        write_readme(work)
        kv("readme", "README.md")
    print("\n  Remaining by hand: paste the Drive links into README.md, and "
          "write up\n  the filter-audit judgements in the report.")


if __name__ == "__main__":
    main()
