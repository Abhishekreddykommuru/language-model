"""Progress report: what has been collected, and how much is left.

Read-only, so it is safe to run at any time, including mid-crawl.

    python scripts/status.py              both languages
    python scripts/status.py --lang tel   one language
    python scripts/status.py --fast       frontier counts only

Per language it reports the crawl frontier, what has been collected by source
and by manual/downloaded, progress against the targets in sources.yaml, and how
the corpus is tracking against 500M tokens at a 20% manual share.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.collect.hf_download import BYTES_PER_TOKEN
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import read_dir
from common.log import human_bytes, kv, millions, section, table

STAGES = [("1_raw", "collected"), ("3_clean", "cleaned"),
          ("4_dedup", "deduplicated"), ("5_final", "final")]

#: Measured share of collected text surviving cleaning and deduplication.
SURVIVAL = 0.55


def scan_raw(root: Path, bpt: float) -> dict:
    """Walk 1_raw, grouping by source_class and source directory."""
    by_source = defaultdict(lambda: {"docs": 0, "bytes": 0})
    by_class = defaultdict(lambda: {"docs": 0, "bytes": 0})
    if not root.exists():
        return {"by_source": {}, "by_class": {}}

    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = cls_dir.name                      # downloaded | manual
        for src_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            # web_crawl nests one more level: web_crawl/<host>/
            leaves = [p for p in src_dir.iterdir() if p.is_dir()] \
                if src_dir.name == "web_crawl" else [src_dir]
            for leaf in leaves:
                label = (f"{src_dir.name}/{leaf.name}"
                         if src_dir.name == "web_crawl" else src_dir.name)
                for d in read_dir(str(leaf)):
                    nb = d.get("n_bytes") or len(d.get("text", "").encode())
                    by_source[label]["docs"] += 1
                    by_source[label]["bytes"] += nb
                    by_class[cls]["docs"] += 1
                    by_class[cls]["bytes"] += nb
    return {"by_source": dict(by_source), "by_class": dict(by_class)}


def frontier_status(db_path: Path) -> dict:
    """Per-host fetched / pending / failed counts from the crawl frontier."""
    if not db_path.exists():
        return {}
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for (host,) in db.execute("SELECT DISTINCT host FROM urls"):
        counts = dict(db.execute(
            "SELECT status, COUNT(*) FROM urls WHERE host=? GROUP BY status",
            (host,)))
        out[host] = {"fetched": counts.get(1, 0), "pending": counts.get(0, 0),
                     "failed": counts.get(2, 0)}
    db.close()
    return out


def stage_sizes(base: Path, bpt: float) -> list:
    """Documents and estimated tokens surviving each pipeline stage."""
    rows = []
    for folder, label in STAGES:
        d = base / folder
        if not d.exists():
            continue
        sub = d / "train" if (d / "train").exists() else d
        docs = nbytes = 0
        for rec in read_dir(str(sub)):
            docs += 1
            nbytes += rec.get("n_bytes") or len(rec.get("text", "").encode())
        rows.append({"stage": label, "docs": docs, "bytes": nbytes,
                     "tokens": nbytes / bpt})
    return rows


def report(lang: str, work: str, fast: bool):
    name = lang_dir(lang)
    bpt = BYTES_PER_TOKEN[lang]
    base = Path(work) / name
    cfg = yaml.safe_load(open(f"{name}/configs/sources.yaml", encoding="utf-8"))

    section(f"{cfg['lang_name']} ({lang})")

    fr = frontier_status(Path(work) / "_state" / f"frontier_{lang}.sqlite")
    if fr:
        rows = []
        for host, c in sorted(fr.items(), key=lambda kv: -kv[1]["fetched"]):
            if not c["fetched"] and not c["pending"]:
                note = "nothing discovered"
            elif not c["pending"]:
                note = "exhausted"
            else:
                note = ""
            rows.append([host, f"{c['fetched']:,}", f"{c['pending']:,}",
                         f"{c['failed']:,}", note])
        rows.append(["all hosts",
                     f"{sum(c['fetched'] for c in fr.values()):,}",
                     f"{sum(c['pending'] for c in fr.values()):,}",
                     f"{sum(c['failed'] for c in fr.values()):,}", ""])
        print("\ncrawl frontier")
        table(["host", "fetched", "pending", "failed", ""], rows,
              aligns="lrrrl")

    if fast:
        return

    raw = scan_raw(base / "1_raw", bpt)
    if raw["by_source"]:
        print("\ncollected so far")
        table(["source", "documents", "size", "tokens"],
              [[src, f"{v['docs']:,}", human_bytes(v["bytes"]),
                millions(v["bytes"] / bpt)]
               for src, v in sorted(raw["by_source"].items(),
                                    key=lambda kv: -kv[1]["bytes"])])

        man = raw["by_class"].get("manual", {"docs": 0, "bytes": 0})
        dl = raw["by_class"].get("downloaded", {"docs": 0, "bytes": 0})
        tot_b = man["bytes"] + dl["bytes"]
        frac = man["bytes"] / tot_b if tot_b else 0
        print()
        table(["", "documents", "size", "tokens", "share"],
              [["manual", f"{man['docs']:,}", human_bytes(man["bytes"]),
                millions(man["bytes"] / bpt), f"{frac:.1%}"],
               ["downloaded", f"{dl['docs']:,}", human_bytes(dl["bytes"]),
                millions(dl["bytes"] / bpt), f"{1 - frac:.1%}"],
               ["total", f"{man['docs'] + dl['docs']:,}", human_bytes(tot_b),
                millions(tot_b / bpt), ""]])

        # Progress against the per-bucket targets in sources.yaml.
        got = {"crawl": 0.0, "ocr": 0.0, "pdf": 0.0}
        for src, v in raw["by_source"].items():
            tk = v["bytes"] / bpt
            if src.startswith("web_crawl"):
                got["crawl"] += tk
            elif "ocr" in src:
                got["ocr"] += tk
            elif "pdf" in src:
                got["pdf"] += tk
        targets = (("web crawl", got["crawl"], cfg["crawl"]["target_raw_tokens"]),
                   ("OCR books", got["ocr"], cfg["ocr"]["target_raw_tokens"]),
                   ("text PDFs", got["pdf"], cfg["pdf_text"]["target_raw_tokens"]),
                   ("downloaded", dl["bytes"] / bpt,
                    sum(d["target_raw_tokens"] for d in cfg["downloaded"])))
        print("\nprogress against target")
        table(["bucket", "collected", "target", "", "done"],
              [[label, millions(have), millions(target),
                "#" * int(min(have / target if target else 0, 1) * 24),
                f"{(have / target if target else 0):.0%}"]
               for label, have, target in targets], aligns="lrrll")

        # What the corpus can reach, and what more crawling would buy.
        man_tok = man["bytes"] / bpt
        reach = tot_b / bpt * SURVIVAL
        print()
        kv("corpus if all is kept", f"{millions(reach)} tokens at "
                                    f"{man_tok * SURVIVAL / max(reach, 1):.1%} manual")
        kv("manual for 20% of 500M", f"{millions(100e6 / SURVIVAL)} raw, "
                                     f"against {millions(man_tok)} today")

        active = sum(1 for c in fr.values() if c["pending"] > 0) if fr else 0
        if active:
            # Two requests/second per host, ~490 words a page, ~70% extractable.
            rate = active * 2 * 3600 * 490 * 0.70
            print(f"\n  {active} host(s) still have pending URLs, worth about "
                  f"{millions(rate)} raw tokens an hour")
            table(["corpus target", "manual needed", "more crawl hours"],
                  [[millions(t), millions(t * 0.20),
                    f"{max(0.0, (t * 0.20 / SURVIVAL - got['crawl']) / rate):.1f}"]
                   for t in (250e6, 350e6, 500e6)])

    rows = stage_sizes(base, bpt)
    if len(rows) > 1:
        first = rows[0]["docs"] or 1
        print("\npipeline stages, train split")
        table(["stage", "documents", "size", "tokens", "survival"],
              [[r["stage"], f"{r['docs']:,}", human_bytes(r["bytes"]),
                millions(r["tokens"]), f"{r['docs'] / first:.1%}"]
               for r in rows])
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["tel", "nep"], default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="frontier counts only; skips reading every shard")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()

    # The Drive layout changed mid-project. Checkpoints written under the old
    # paths are invisible to the current code, so that work would be redone
    # silently rather than resumed.
    root = Path(work)
    old = [d for d in ("raw", "state", "clean", "dedup", "final", "splits",
                       "tokenized", "tokenizers", "logs", "raw_html")
           if (root / d).exists()]
    section("Pipeline state")
    kv("work directory", work)
    if old:
        print()
        print("  ! Folders from the old directory layout are present: "
              + ", ".join(old))
        print("    The current code does not read them, so that work would be")
        print("    repeated. To keep it, migrate first:")
        print(f"      mkdir -p {work}/_state")
        print(f"      cp -rn {work}/state/* {work}/_state/")
        print(f"      cp -rn {work}/raw_html/* {work}/_raw_html/")

    rows = []
    for lang in ("tel", "nep"):
        st = root / "_state"
        dl = st / f"download_{lang}"
        ocr = st / f"ocr_{lang}"
        fr_db = st / f"frontier_{lang}.sqlite"
        tar_dir = root / "_raw_html" / lang
        rows.append([
            lang,
            len(list(dl.glob("*._SUCCESS"))) if dl.exists() else 0,
            f"{fr_db.stat().st_size / 1e6:.0f} MB" if fr_db.exists() else "none",
            f"{len(list(tar_dir.glob('*.tar'))):,}" if tar_dir.exists() else "0",
            len(list(ocr.glob("*._SUCCESS"))) if ocr.exists() else 0,
        ])
    print()
    table(["language", "downloads done", "frontier", "html archives",
           "books OCR'd"], rows)

    for lang in ([args.lang] if args.lang else ["tel", "nep"]):
        try:
            report(lang, work, args.fast)
        except Exception as exc:
            print(f"  {lang}: could not report, {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
