"""Stage 1: text-layer PDFs for the manual share, with no recognition error.

Government publications produced from word processors carry a real text layer,
which pypdf extracts directly. It is the highest tokens-per-hour manual source
available, and covers most of Nepali's budget where scanned books are scarce.

Build the URL list, then extract:
    python scripts/03b_pdf_text.py --lang nep --discover
    python scripts/03b_pdf_text.py --lang nep
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.collect.pdf_text import (download_pdf, extract_pdf_text,
                                     find_pdf_links, has_text_layer)
from common.io.checkpoint import Checkpoint
from common.io.paths import ensure_drive_mounted, lang_dir, scratch_dir, work_dir
from common.io.schema import Doc, make_doc_id
from common.io.shards import ShardWriter
from common.log import kv

MIN_WORDS = 200


def discover(cfg: dict, url_list: str):
    """Scrape .pdf links from the configured listing pages into url_list."""
    found = set()
    existing = Path(url_list)
    if existing.exists():
        found.update(l.strip() for l in existing.read_text().splitlines() if l.strip())
    for page in cfg.get("seed_listing_pages", []) or []:
        try:
            links = find_pdf_links(page)
            print(f"  {page}: {len(links)} PDF links")
            found.update(links)
        except Exception as exc:
            print(f"  {page}: failed, {exc}")
    Path(url_list).parent.mkdir(parents=True, exist_ok=True)
    Path(url_list).write_text("\n".join(sorted(found)), encoding="utf-8")
    print(f"  {len(found)} PDF URLs written to {url_list}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--discover", action="store_true",
                    help="scrape listing pages for PDF links, then exit")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/sources.yaml",
                              encoding="utf-8"))["pdf_text"]
    url_list = cfg["url_list"]

    if args.discover:
        discover(cfg, url_list)
        return

    def load_urls() -> list:
        if not Path(url_list).exists():
            return []
        return [u.strip() for u in
                Path(url_list).read_text(encoding="utf-8").splitlines() if u.strip()]

    urls = load_urls()

    # Build the list from the listing pages rather than aborting: run_all.py
    # invokes this stage without --discover, and failing hard here would kill an
    # otherwise healthy multi-hour pipeline.
    if not urls and cfg.get("seed_listing_pages"):
        print("  URL list is empty, discovering from the seed listing pages")
        discover(cfg, url_list)
        urls = load_urls()

    if not urls:
        # Nothing to do is not an error: Telugu has no seed pages configured, so
        # the stage is skipped and the crawl and OCR buckets carry the budget.
        seeds = cfg.get("seed_listing_pages") or []
        why = ("the listing pages returned no links; the site may be down or "
               "blocking us" if seeds else "no seed_listing_pages are configured")
        print(f"  skipping text-layer PDFs for {args.lang}: {url_list} is empty "
              f"and {why}")
        print(f"  to use this source later, put PDF URLs in {url_list}, one per "
              f"line, and re-run this stage")
        return

    ck = Checkpoint(f"{work}/_state/pdftext_{args.lang}")
    workdir = f"{scratch_dir()}/pdf_work"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_written = 0

    with ShardWriter(f"{work}/{lang_dir(args.lang)}/1_raw/manual/pdf_documents",
                     "pdf_doc") as writer:
        for i, url in enumerate(urls, 1):
            if ck.done(url):
                continue
            local = None
            try:
                local = download_pdf(url, f"{workdir}/{abs(hash(url)) % 10 ** 12}.pdf")
                if not has_text_layer(local):
                    ck.mark(url, {"no_text_layer": 1})
                    continue
                res = extract_pdf_text(local, args.lang)
                if len(res["text"].split()) >= MIN_WORDS:
                    doc = Doc(
                        doc_id=make_doc_id(args.lang, "manual", "pdftext", n_written),
                        lang=args.lang, text=res["text"],
                        source_class="manual", method="pdf_text",
                        source=url.split("/")[2],
                        licence="government publication",
                        url=url, fetched_at=now,
                        extractor="pypdf-textlayer",
                    ).finalize()
                    writer.write(doc.to_dict())
                    n_written += 1
                ck.mark(url, {"pages_kept": res.get("pages_kept", 0)})
                if i % 20 == 0:
                    print(f"  [{i}/{len(urls)}] {n_written:,} documents so far",
                          flush=True)
            except Exception as exc:
                ck.mark(url, {"failed": 1})
                print(f"  [{i}/{len(urls)}] {url}: {exc}", flush=True)
            finally:
                if local:
                    Path(local).unlink(missing_ok=True)

    print()
    kv("text-layer PDFs written", f"{n_written:,}")


if __name__ == "__main__":
    main()
