"""Text-layer PDFs: the highest tokens-per-hour manual source, and no OCR error.

Government PDFs published from word processors carry a real text layer, which
pypdf extracts at hundreds of pages per second. This covers most of Nepali's
manual budget, where scanned books are scarce but Acts and Rules are plentiful.
"""
import urllib.request
from pathlib import Path

from pypdf import PdfReader

from common.clean.normalize import SCRIPT_RANGES, script_ratio

USER_AGENT = "lma-cl3410-student-corpus/1.0 (academic research)"


def download_pdf(url: str, dest: str) -> str:
    """Download one PDF, skipping it if already present."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1000:
        return str(out)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as fh:
        fh.write(r.read())
    return str(out)


def has_text_layer(pdf_path: str, probe_pages: int = 3) -> bool:
    """True if the PDF contains extractable text, i.e. it is not a scan."""
    try:
        reader = PdfReader(pdf_path)
    except Exception:
        return False
    chars = 0
    for page in reader.pages[:probe_pages]:
        try:
            chars += len(page.extract_text() or "")
        except Exception:
            pass
    return chars > 200


def extract_pdf_text(pdf_path: str, lang: str,
                     min_script_ratio: float = 0.5) -> dict:
    """Extract text from a text-layer PDF, dropping wrong-script pages.

    Government publications are often bilingual, so mostly-English pages are
    discarded rather than polluting a monolingual corpus.
    """
    lo, hi = SCRIPT_RANGES[lang]
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        return {"text": "", "pages_total": 0, "pages_kept": 0, "error": str(exc)}

    kept, n_pages = [], 0
    for page in reader.pages:
        n_pages += 1
        try:
            t = page.extract_text() or ""
        except Exception:
            continue
        if len(t.strip()) < 100:
            continue
        if script_ratio(t, lo, hi) < min_script_ratio:
            continue
        kept.append(t)

    return {"text": "\n\n".join(kept), "pages_total": n_pages,
            "pages_kept": len(kept)}


def find_pdf_links(page_url: str, timeout: int = 60) -> list:
    """Scrape absolute .pdf links from a listing page."""
    import re
    from urllib.parse import urljoin

    req = urllib.request.Request(page_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="ignore")
    links = re.findall(r'href="([^"]+\.pdf)"', html, flags=re.IGNORECASE)
    return sorted({urljoin(page_url, l) for l in links})
