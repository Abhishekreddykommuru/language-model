"""Write DATASET_CARD.md at the Drive root, so the shared link explains itself.

Describes the layout, the exact sources, and the manual versus downloaded token
split as measured by the real tokenizer.

Usage:
    python scripts/11_dataset_card.py
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv

LANGS = [("tel", "Telugu", "Model H (higher-resource)"),
         ("nep", "Nepali", "Model L (lower-resource)")]


def token_report(work: str, lang: str) -> dict:
    p = Path(f"{work}/{lang_dir(lang)}/6_tokenized/token_report.json")
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ensure_drive_mounted()
    work = work_dir()
    out = [
        "# Phase 1 Dataset — Telugu (Model H) and Nepali (Model L)",
        "",
        "Monolingual corpora and tokenizers for Telugu and Nepali.",
        "The two languages share no documents, no vocabulary and no tokenizer.",
        "",
        "## Folder layout",
        "",
        "```",
        "<language>/",
        "  1_raw/                 everything collected, before any cleaning",
        "    downloaded/          public corpora, one folder per source",
        "    manual/              collected by us -- this is the >=20% share",
        "      web_crawl/<host>/    our own crawler, one folder per news site",
        "      ocr_books/           our own OCR of scanned books (archive.org)",
        "      pdf_documents/       text-layer PDFs, extracted with pypdf",
        "  2_splits/              train/val/test, held out BEFORE cleaning",
        "  3_clean/               after language ID + normalisation + filters",
        "  4_dedup/               after exact + MinHash near-duplicate removal",
        "  5_final/               <-- THE CORPUS. train/val/test",
        "  6_tokenized/           <-- uint16 token arrays + token_report.json",
        "  tokenizer/             .model, .vocab.txt, sweep results",
        "_state/                  pipeline checkpoints (not part of the dataset)",
        "_logs/                   per-stage counters",
        "_raw_html/               NOT SHARED: raw crawled HTML, not redistributable",
        "```",
        "",
        "Every file is `.jsonl.zst`: one JSON document per line, zstd-compressed.",
        "Each document carries full provenance:",
        "",
        "```json",
        '{"doc_id": "...", "lang": "tel", "text": "...",',
        ' "source_class": "manual|downloaded", "method": "self_crawl|own_ocr|',
        '  pdf_text|hf_download", "source": "...", "licence": "...", "url": "...",',
        ' "fetched_at": "...", "filters": [...], "n_chars": 0, "n_bytes": 0}',
        "```",
        "",
        "So the manual/downloaded split is verifiable from the data itself, not",
        "only from the folder names.",
        "",
    ]

    for code, name, tier in LANGS:
        cfg = yaml.safe_load(
            open(f"{lang_dir(code)}/configs/sources.yaml", encoding="utf-8"))
        rep = token_report(work, code)
        out += [f"## {name} — {tier}", ""]

        if rep.get("train"):
            tr = rep["train"]
            cls = tr.get("tokens_by_class", {})
            man, dl = cls.get("manual", 0), cls.get("downloaded", 0)
            total = man + dl
            frac = rep.get("manual_fraction_train", 0)
            out += [
                "| | tokens | share |",
                "|---|---|---|",
                f"| **Manual** (crawl + OCR + PDFs) | {man:,} | **{frac:.1%}** |",
                f"| Downloaded (public corpora) | {dl:,} | {1 - frac:.1%} |",
                f"| **Total (train)** | **{total:,}** | |",
                "",
                f"Requirement is >=20% manual: "
                f"**{'MET' if frac >= 0.20 else 'NOT MET'}**",
                "",
                "| split | documents | tokens | bytes of text |",
                "|---|---|---|---|",
            ]
            for sp in ("train", "val", "test"):
                if sp in rep:
                    r = rep[sp]
                    out.append(f"| {sp} | {r['documents']:,} | {r['tokens']:,} | "
                               f"{r['bytes_of_text']:,} |")
            out += ["", f"Tokenizer: **{rep.get('algorithm', '?')}**, "
                        f"vocabulary **{rep.get('vocab_size', '?'):,}**", ""]
            if tr.get("tokens_by_source"):
                out += ["| source | tokens |", "|---|---|"]
                for src, n in sorted(tr["tokens_by_source"].items(),
                                     key=lambda kv: -kv[1]):
                    out.append(f"| {src} | {n:,} |")
                out.append("")
        else:
            out += ["_Token counts appear here once stage 09 has run._", ""]

        out += ["### Sources", "", "| source | class | licence |", "|---|---|---|"]
        for d in cfg["downloaded"]:
            out.append(f"| {d['name']} | downloaded | {d.get('licence', '')} |")
        for s in cfg["crawl"]["sites"]:
            out.append(f"| {s['host']} | manual (web crawl) | site terms; "
                       f"academic research |")
        out.append("| archive.org scanned books | manual (our OCR) | "
                   "public domain / item licence |")
        out.append("| government PDFs | manual (text layer) | government "
                   "publication |")
        excluded = cfg.get("excluded_hosts") or []
        if excluded:
            out += ["", "**Deliberately excluded** (declare "
                        "`Content-Signal: ai-train=no`): "
                    + ", ".join(f"`{h}`" for h in excluded) + ".", ""]
        out.append("")

    out += [
        "## Reproducing",
        "",
        "All collection, cleaning and tokenizer code is in the GitHub repository",
        "on branch `phase-1`. The whole pipeline runs with:",
        "",
        "```bash",
        "python scripts/run_all.py",
        "```",
        "",
    ]

    path = Path(work) / "DATASET_CARD.md"
    path.write_text("\n".join(out), encoding="utf-8")
    kv("dataset card written to", path)


if __name__ == "__main__":
    main()
