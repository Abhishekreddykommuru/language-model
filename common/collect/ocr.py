"""OCR of scanned books from the Internet Archive.

The Archive holds 12,887 Telugu text items but only 240 Nepali ones, so the two
sources.yaml files budget OCR very differently; Nepali leans on text-layer PDFs
instead. Tesseract is an OCR engine rather than a language model, so using it
introduces no pretrained language modelling into the pipeline.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

from common.clean.normalize import SCRIPT_RANGES, script_ratio

USER_AGENT = "lma-cl3410-student-corpus/1.0 (academic research)"


def search_archive_org(query: str, max_items: int = 1500) -> list:
    """Return Internet Archive identifiers matching a search query."""
    url = (
        "https://archive.org/advancedsearch.php?q="
        + urllib.parse.quote(query)
        + "&fl%5B%5D=identifier&rows=" + str(max_items)
        + "&page=1&output=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return [d["identifier"] for d in data["response"]["docs"]]


def download_item_pdf(identifier: str, dest_dir: str) -> str:
    """Return the local path of the item's PDF, or "" if it has none."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    out = Path(dest_dir) / f"{identifier}.pdf"
    if out.exists() and out.stat().st_size > 10_000:
        return str(out)

    req = urllib.request.Request(f"https://archive.org/metadata/{identifier}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    pdfs = [f["name"] for f in meta.get("files", [])
            if f["name"].lower().endswith(".pdf")]
    if not pdfs:
        return ""

    name = sorted(pdfs, key=len)[0]
    url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return str(out)


def ocr_pdf_tesseract(pdf_path: str, lang_code: str, max_pages: int = 400,
                      dpi: int = 300, batch: int = 10) -> list:
    """OCR a PDF page by page, returning one string per page.

    Pages are converted in small batches so a long book does not exhaust RAM.
    `lang_code` is a Tesseract model name: "tel" or "nep".
    """
    import pytesseract
    from pdf2image import convert_from_path

    pages = []
    for start in range(1, max_pages + 1, batch):
        try:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=start,
                                       last_page=min(start + batch - 1, max_pages))
        except Exception:
            break
        if not images:
            break
        for img in images:
            try:
                pages.append(pytesseract.image_to_string(img, lang=lang_code))
            except Exception:
                pages.append("")
        if len(pages) >= max_pages:
            break
    return pages


def page_quality(text: str, lang: str, vocab: set = None) -> float:
    """Score one OCR'd page in [0, 1].

    Script purity plus a hit rate against a word list built from our own clean
    web corpus. OCR garbage is character salad and fails both tests.
    """
    if len(text.strip()) < 50:
        return 0.0
    lo, hi = SCRIPT_RANGES[lang]
    sr = script_ratio(text, lo, hi)
    if not vocab:
        return sr
    words = [w for w in text.split() if len(w) > 2]
    hit = sum(1 for w in words if w in vocab) / max(len(words), 1)
    return 0.5 * sr + 0.5 * hit


def filter_ocr_pages(pages: list, lang: str, vocab: set = None,
                     min_score: float = 0.60) -> tuple:
    """Keep pages passing the quality gate. Returns (joined_text, stats)."""
    kept, scores = [], []
    for p in pages:
        s = page_quality(p, lang, vocab)
        scores.append(s)
        if s >= min_score:
            kept.append(p)
    return "\n\n".join(kept), {
        "pages_in": len(pages),
        "pages_kept": len(kept),
        "mean_score": round(sum(scores) / max(len(scores), 1), 4),
    }
