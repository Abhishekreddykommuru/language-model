"""Stage 1: OCR scanned books from the Internet Archive (manual share).

One archive.org item is one checkpointed unit, so a lost session costs at most
one book. Quality-gated: pages that fail a script-purity and dictionary-hit test
are discarded rather than poisoning the corpus with OCR salad.

Usage:
    python scripts/03_ocr.py --lang tel --max-items 400
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.collect.ocr import (download_item_pdf, filter_ocr_pages,
                                ocr_pdf_tesseract, search_archive_org)
from common.io.checkpoint import Checkpoint
from common.io.paths import ensure_drive_mounted, lang_dir, scratch_dir, work_dir
from common.io.schema import Doc, make_doc_id
from common.io.shards import ShardWriter
from common.log import kv, section

MIN_WORDS_PER_BOOK = 200


def load_wordlist(path: str) -> set:
    """Word list built from our own clean web corpus, for scoring OCR pages."""
    p = Path(path)
    if not p.exists():
        print(f"  no word list at {path}, scoring on script purity alone")
        return set()
    return set(p.read_text(encoding="utf-8").split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--max-items", type=int, default=400)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/sources.yaml",
                              encoding="utf-8"))["ocr"]
    ck = Checkpoint(f"{work}/_state/ocr_{args.lang}")
    vocab = load_wordlist(f"{work}/_state/wordlist_{args.lang}.txt")
    workdir = f"{scratch_dir()}/ocr_work"

    identifiers = search_archive_org(cfg["archive_org_query"], cfg["max_items"])
    todo = [i for i in identifiers if not ck.done(i)][:args.max_items]
    section(f"OCR of scanned books, {args.lang}")
    kv("items matching the query", f"{len(identifiers):,}")
    kv("already done", f"{len(identifiers) - len([i for i in identifiers if not ck.done(i)]):,}")
    kv("to process this session", f"{len(todo):,}")
    print()

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_written = 0
    with ShardWriter(f"{work}/{lang_dir(args.lang)}/1_raw/manual/ocr_books",
                     "ocr_book") as writer:
        for i, ident in enumerate(todo, 1):
            try:
                pdf = download_item_pdf(ident, workdir)
                if not pdf:
                    ck.mark(ident, {"skipped_no_pdf": 1})
                    continue
                pages = ocr_pdf_tesseract(pdf, cfg["tesseract_lang"],
                                          cfg["max_pages_per_item"],
                                          cfg["min_dpi"])
                text, stats = filter_ocr_pages(pages, args.lang, vocab,
                                               cfg.get("min_page_score", 0.60))
                if len(text.split()) >= MIN_WORDS_PER_BOOK:
                    doc = Doc(
                        doc_id=make_doc_id(args.lang, "manual", "ocr", n_written),
                        lang=args.lang, text=text,
                        source_class="manual", method="own_ocr",
                        source=f"archive.org/{ident}",
                        licence="archive.org item licence / public domain",
                        url=f"https://archive.org/details/{ident}",
                        fetched_at=now,
                        extractor=f"tesseract-{cfg['tesseract_lang']}",
                    ).finalize()
                    writer.write(doc.to_dict())
                    n_written += 1
                ck.mark(ident, stats)
                Path(pdf).unlink(missing_ok=True)
                print(f"  [{i}/{len(todo)}] {ident}: kept "
                      f"{stats['pages_kept']}/{stats['pages_in']} pages, "
                      f"mean score {stats['mean_score']:.2f}", flush=True)
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {ident}: failed, {exc}", flush=True)
                ck.mark(ident, {"failed": 1})

    print()
    kv("OCR documents written", f"{n_written:,}")


if __name__ == "__main__":
    main()
