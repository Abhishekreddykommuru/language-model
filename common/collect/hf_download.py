"""Download the public-corpus share from HuggingFace.

Three access patterns, all resumable and all capped by a token target so we stop
as soon as we have enough: streaming splits, individual parquet files, and raw
.txt files read over HTTP. Everything lands on local disk first and reaches
Drive only as a finished shard.
"""
import time
from pathlib import Path

from common.io.schema import Doc, make_doc_id
from common.io.shards import ShardWriter

#: Measured bytes per token at a 32k unigram vocabulary. Used only to decide
#: when a download target has been met, before any tokenizer exists.
BYTES_PER_TOKEN = {"tel": 12.69, "nep": 12.89}

MIN_DOC_CHARS = 200


def estimate_tokens(text: str, lang: str) -> float:
    return len(text.encode("utf-8")) / BYTES_PER_TOKEN[lang]


def _make_doc(spec: dict, lang: str, text: str, n: int, now: str, url: str) -> dict:
    return Doc(
        doc_id=make_doc_id(lang, "downloaded", spec["name"], n),
        lang=lang,
        text=text.strip(),
        source_class="downloaded",
        method="hf_download",
        source=spec["name"],
        licence=spec.get("licence", ""),
        url=url,
        fetched_at=now,
        extractor="hf_datasets",
    ).finalize().to_dict()


def download_streaming(spec: dict, lang: str, out_dir: str) -> dict:
    """Stream one dataset config until the token target is met."""
    from datasets import load_dataset

    target = spec["target_raw_tokens"]
    field = spec.get("text_field", "text")
    ds = load_dataset(spec["repo"], spec.get("config"), split="train", streaming=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    got, n_docs = 0.0, 0
    with ShardWriter(out_dir, spec["name"]) as writer:
        for row in ds:
            text = (row.get(field) or "").strip()
            if len(text) < MIN_DOC_CHARS:
                continue
            writer.write(_make_doc(spec, lang, text, n_docs, now,
                                   f"{spec['repo']}/{spec.get('config', '')}"))
            n_docs += 1
            got += estimate_tokens(text, lang)
            if n_docs % 50_000 == 0:
                print(f"    {spec['name']}: {n_docs:,} docs, "
                      f"~{got / 1e6:.1f}M tokens", flush=True)
            if got >= target:
                break
    return {"docs": n_docs, "est_tokens": int(got)}


def download_parquet_files(spec: dict, lang: str, out_dir: str) -> dict:
    """Download a repo's parquet files one at a time, until the target is met.

    Each file is an independent unit, so a lost session costs at most one file,
    and the local parquet is deleted straight after conversion to reclaim disk.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    skip = set(spec.get("skip_files", []))
    files = sorted(
        f for f in api.list_repo_files(spec["repo"], repo_type="dataset")
        if f.startswith(spec["files_prefix"])
        and f.endswith(".parquet")
        and f.split("/")[-1] not in skip
    )

    target = spec["target_raw_tokens"]
    field = spec.get("text_field", "text")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    got, n_docs = 0.0, 0

    with ShardWriter(out_dir, spec["name"]) as writer:
        for fname in files:
            if got >= target:
                break
            print(f"    downloading {fname}", flush=True)
            local = hf_hub_download(spec["repo"], fname, repo_type="dataset")
            try:
                table = pq.read_table(local, columns=[field])
                for text in table.column(field).to_pylist():
                    if not text or len(text) < MIN_DOC_CHARS:
                        continue
                    writer.write(_make_doc(spec, lang, text, n_docs, now,
                                           f"{spec['repo']}/{fname}"))
                    n_docs += 1
                    got += estimate_tokens(text, lang)
                    if got >= target:
                        break
            finally:
                Path(local).unlink(missing_ok=True)
            print(f"    {spec['name']}: {n_docs:,} docs, ~{got / 1e6:.1f}M tokens",
                  flush=True)
    return {"docs": n_docs, "est_tokens": int(got)}


def download_text_stream(spec: dict, lang: str, out_dir: str) -> dict:
    """Stream a raw .txt file from the Hub over HTTP, stopping at the target.

    Some corpora are not loadable datasets: IndicCorpV2 is a single 15.8 GB
    te.txt, so load_dataset() fails and downloading it whole to keep 400 MB
    would be wasteful. One line is one document in these files.
    """
    import urllib.request

    target = spec["target_raw_tokens"]
    url = spec["url"]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    got, n_docs = 0.0, 0

    req = urllib.request.Request(url, headers={"User-Agent": "lma-cl3410/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, \
            ShardWriter(out_dir, spec["name"]) as writer:
        buf = b""
        while got < target:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                text = raw.decode("utf-8", errors="ignore").strip()
                if len(text) < MIN_DOC_CHARS:
                    continue
                writer.write(_make_doc(spec, lang, text, n_docs, now, url))
                n_docs += 1
                got += estimate_tokens(text, lang)
            if n_docs and n_docs % 100_000 == 0:
                print(f"    {spec['name']}: {n_docs:,} docs, "
                      f"~{got / 1e6:.1f}M tokens", flush=True)
    return {"docs": n_docs, "est_tokens": int(got)}


def run_source(spec: dict, lang: str, out_dir: str) -> dict:
    """Dispatch on the `kind` field of a sources.yaml entry."""
    kind = spec["kind"]
    if kind == "hf_streaming":
        return download_streaming(spec, lang, out_dir)
    if kind == "hf_parquet_files":
        return download_parquet_files(spec, lang, out_dir)
    if kind == "hf_text_stream":
        return download_text_stream(spec, lang, out_dir)
    raise ValueError(f"unknown source kind: {kind}")
