"""Stages 2 and 3 of the Yi pipeline: language filtering, then rules.

Rejected documents are written out rather than discarded, so that
scripts/audit_filters.py can show what the thresholds actually removed. Reading
those is the only way to tell a working filter from a destructive one.

Usage:
    python scripts/05_clean.py --lang tel --split train
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.clean.lid import LanguageIdentifier
from common.clean.normalize import normalize
from common.clean.pii import anonymize
from common.clean.quality import QualityFilter
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import (ShardWriter, read_dir, require_shards,
                              reset_dir)
from common.log import kv, progress, section, table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/clean.yaml",
                              encoding="utf-8"))
    lid = LanguageIdentifier(cfg["lid"])
    profiles = cfg.get("profiles", {"downloaded": "full", "manual": "full"})
    # 'light' for corpora their authors already cleaned, 'full' for our own
    # crawl and OCR. Normalisation, LID, dedup and decontamination run on
    # everything regardless.
    filters = {
        "full": QualityFilter(cfg["quality"]),
        "light": QualityFilter({**cfg["quality"], **cfg.get("light", {}),
                                "repair_mode": False}),
    }
    apply_pii = {"full": True,
                 "light": cfg.get("light", {}).get("apply_pii", False)}
    section(f"Cleaning {args.lang}, {args.split} split")
    kv("filter profiles", f"downloaded={profiles.get('downloaded')}, "
                          f"manual={profiles.get('manual')}")

    base = f"{work}/{lang_dir(args.lang)}"
    src = f"{base}/2_splits/{args.split}"
    dst = f"{base}/3_clean/{args.split}"
    rejected = f"{base}/3_clean/{args.split}_rejected"

    require_shards(src, "Stage 05", f"Run: python scripts/04_split.py "
                                    f"--lang {args.lang}")
    reset_dir(dst, rejected)

    counters = Counter()
    with ShardWriter(dst, args.split) as keep_w, \
            ShardWriter(rejected, "rejected") as rej_w:
        for d in progress(read_dir(src), f"cleaning {args.lang} {args.split}"):
            counters["in"] += 1

            # Stage 2: language filtering.
            ok, reason, detail = lid.accept(d["text"], args.lang)
            if not ok:
                counters[f"lid_{reason}"] += 1
                counters["lid_rejected"] += 1
                d["reject_reason"] = reason
                d["reject_detail"] = detail
                rej_w.write(d)
                continue
            counters["lid_pass"] += 1

            # Stage 3: normalise, anonymise, then the heuristic filters.
            # Normalisation runs on every document whatever the profile: mixing
            # normalised and raw text would give the tokenizer two spellings of
            # each word and break exact deduplication.
            text = normalize(d["text"], args.lang)

            profile = profiles.get(d.get("source_class", "downloaded"), "full")
            counters[f"profile_{profile}"] += 1

            if apply_pii[profile]:
                text, pii = anonymize(text)
                counters["pii_email"] += pii["email"]
                counters["pii_phone"] += pii["phone"]

            kept, text, reason, _metrics = filters[profile].apply(text)
            if not kept:
                counters[f"quality_{reason}"] += 1
                counters["quality_rejected"] += 1
                d["reject_reason"] = reason
                rej_w.write(d)
                continue

            if reason == "repaired":
                counters["repaired"] += 1
            counters["out"] += 1
            d["text"] = text
            d["n_chars"] = len(text)
            d["n_bytes"] = len(text.encode("utf-8"))
            d["filters"] = ["lid", "normalize", f"quality:{profile}"] + \
                (["pii"] if apply_pii[profile] else [])
            keep_w.write(d)

            if counters["in"] % 100_000 == 0:
                print(f"  {counters['in']:,} in -> {counters['out']:,} kept",
                      flush=True)

    survival = counters["out"] / max(counters["in"], 1)
    rejects = sorted(((k, v) for k, v in counters.items()
                      if k.startswith(("lid_", "quality_"))
                      and k not in ("lid_pass", "lid_rejected",
                                    "quality_rejected")),
                     key=lambda kv: -kv[1])

    section(f"Cleaning results, {args.lang} {args.split}")
    kv("documents in", f"{counters['in']:,}")
    kv("kept", f"{counters['out']:,}  ({survival:.1%})")
    kv("rejected by language id", f"{counters['lid_rejected']:,}")
    kv("rejected by quality", f"{counters['quality_rejected']:,}")
    kv("repaired, not dropped", f"{counters['repaired']:,}")
    if counters["pii_email"] or counters["pii_phone"]:
        kv("pii replaced", f"{counters['pii_email']:,} emails, "
                           f"{counters['pii_phone']:,} phone numbers")
    if rejects:
        print()
        table(["rejection reason", "documents", "share of input"],
              [[k, f"{v:,}", f"{v / max(counters['in'], 1):.2%}"]
               for k, v in rejects[:12]])

    log_dir = Path(f"{work}/_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"clean_{args.lang}_{args.split}.json", "w") as fh:
        json.dump({"counters": dict(counters), "survival": survival}, fh, indent=2)


if __name__ == "__main__":
    main()
