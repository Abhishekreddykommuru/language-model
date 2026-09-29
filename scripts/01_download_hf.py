"""Stage 1: download the public-corpus share from HuggingFace.

Each dataset is a checkpointed unit, so re-running after a lost Colab session
skips whatever already completed.

Usage:
    python scripts/01_download_hf.py --lang tel
"""
import argparse
import sys

import yaml

sys.path.insert(0, ".")

from common.collect.hf_download import run_source
from common.io.checkpoint import run_resumable
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, millions, section


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None, help="output root (default: Drive)")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/sources.yaml",
                              encoding="utf-8"))
    state = f"{work}/_state/download_{args.lang}"
    specs = {s["name"]: s for s in cfg["downloaded"]}

    # One folder per corpus, so the shared Drive link shows at a glance which
    # text came from which source.
    def run_one(name):
        spec = specs[name]
        out_dir = f"{work}/{lang_dir(args.lang)}/1_raw/downloaded/{name}"
        return run_source(spec, args.lang, out_dir)

    totals = run_resumable(list(specs), state, run_one,
                           desc=f"download {args.lang}")

    section(f"Downloaded corpora, {args.lang}")
    kv("documents", f"{totals.get('docs', 0):,}")
    kv("estimated tokens", millions(totals.get("est_tokens", 0)))
    kv("written to", f"{work}/{lang_dir(args.lang)}/1_raw/downloaded/")


if __name__ == "__main__":
    main()
