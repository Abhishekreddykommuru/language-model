"""Document schema. Every document carries its provenance to the final shard,
which is what makes the manual-vs-downloaded token split auditable."""
import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class Doc:
    """One document and where it came from.

    doc_id       stable and sortable, e.g. "tel-man-crawl-000123456"
    source_class "manual" or "downloaded"
    method       hf_download | self_crawl | own_ocr | pdf_text
    source       dataset name or site host
    filters      stages this document has passed through
    """

    doc_id: str
    lang: str
    text: str
    source_class: str
    method: str
    source: str
    licence: str = ""
    url: str = ""
    fetched_at: str = ""
    extractor: str = ""
    filters: list = field(default_factory=list)
    n_chars: int = 0
    n_bytes: int = 0
    n_tokens: int = 0
    sha256: str = ""

    def finalize(self) -> "Doc":
        """Fill the derived fields. Call before writing to a shard."""
        self.n_chars = len(self.text)
        self.n_bytes = len(self.text.encode("utf-8"))
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Doc":
        known = set(Doc.__dataclass_fields__)
        return Doc(**{k: v for k, v in d.items() if k in known})


def make_doc_id(lang: str, source_class: str, method: str, n: int) -> str:
    tag = "man" if source_class == "manual" else "dl"
    return f"{lang}-{tag}-{method}-{n:09d}"
