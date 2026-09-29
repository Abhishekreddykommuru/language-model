"""Encode every split to a flat uint16 array, for the exact token counts.

These are the real numbers the report quotes, from tokenizer output rather than
the bytes-per-token estimate used while collecting. A flat token array plus
document offsets memory-maps directly in Phase 2; uint16 needs vocab <= 65536,
which the selection rule already enforces, and 500M tokens is exactly 1.0 GB.
Per-split byte counts are recorded here because Phase 2 needs them for
bits-per-byte and recomputing them later is error-prone.

Usage:
    python scripts/09_tokenize.py --lang tel
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.io.shards import read_dir
from common.log import kv, millions, progress, section, table
from common.tokenizer.evaluate import TokenizerWrapper

EOS_ID = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    tokdir = f"{work}/{lang_dir(args.lang)}/tokenizer"
    chosen = json.load(open(f"{tokdir}/chosen.json", encoding="utf-8"))
    tok = TokenizerWrapper(chosen["model"], chosen["chosen"]["algorithm"])
    if tok.size > 65536:
        raise SystemExit(f"vocab {tok.size} too large for uint16 storage")

    out = Path(f"{work}/{lang_dir(args.lang)}/6_tokenized")
    out.mkdir(parents=True, exist_ok=True)
    report = {"tokenizer": chosen["model"], "vocab_size": tok.size,
              "algorithm": chosen["chosen"]["algorithm"]}

    freq = Counter()
    for split in ("train", "val", "test"):
        chunks, offsets, pos = [], [], 0
        by_class = defaultdict(int)
        by_source = defaultdict(int)
        n_bytes = 0

        for d in progress(read_dir(f"{work}/{lang_dir(args.lang)}/5_final/{split}"),
                          f"encoding {split}"):
            ids = tok.encode(d["text"]) + [EOS_ID]
            chunks.append(np.asarray(ids, dtype=np.uint16))
            offsets.append(pos)
            pos += len(ids)
            by_class[d.get("source_class", "downloaded")] += len(ids)
            by_source[d.get("source", "?")] += len(ids)
            n_bytes += d.get("n_bytes") or len(d["text"].encode("utf-8"))
            if split == "train":
                freq.update(ids)
            if len(offsets) % 200_000 == 0:
                print(f"  {split}: {len(offsets):,} docs, {pos:,} tokens",
                      flush=True)

        arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
        arr.tofile(out / f"{split}.bin")
        np.asarray(offsets, dtype=np.int64).tofile(out / f"{split}.idx")

        report[split] = {
            "tokens": int(arr.size),
            "documents": len(offsets),
            "bytes_of_text": n_bytes,          # Phase 2 bits-per-byte needs this
            "tokens_by_class": dict(by_class),
            "tokens_by_source": dict(by_source),
        }
        print(f"  {split}: {arr.size:,} tokens across {len(offsets):,} documents",
              flush=True)

    # Token-frequency statistics, for the Zipf plot and the report. The counter
    # is bounded by the vocabulary, so counting every training token is cheap.
    report["token_frequency"] = {
        "distinct_tokens_used": len(freq),
        "vocab_coverage": len(freq) / max(tok.size, 1),
        "unused_vocab": tok.size - len(freq),
        "counts_sorted": [int(c) for _, c in freq.most_common()],
        "top_50": [{"id": int(i), "piece": tok.id_to_piece(int(i)), "count": int(c)}
                   for i, c in freq.most_common(50)],
    }

    if report["train"]["tokens"] == 0:
        raise SystemExit(
            f"The {args.lang} corpus is empty, so there is nothing to count.\n"
            f"  Check that stages 05 to 07 actually succeeded; a zero token "
            f"report is not a result.")

    train_classes = report["train"]["tokens_by_class"]
    manual = train_classes.get("manual", 0)
    downloaded = train_classes.get("downloaded", 0)
    total = manual + downloaded
    fraction = manual / total if total else 0.0
    report["manual_fraction_train"] = fraction
    report["meets_20pct"] = fraction >= 0.20

    with open(out / "token_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    tf = report["token_frequency"]
    section(f"Exact token counts, {args.lang}")
    table(["split", "documents", "tokens"],
          [[s, f"{report[s]['documents']:,}", f"{report[s]['tokens']:,}"]
           for s in ("train", "val", "test")])
    print()
    table(["train split", "tokens", "share"],
          [["manual", f"{manual:,}", f"{fraction:.1%}"],
           ["downloaded", f"{downloaded:,}", f"{1 - fraction:.1%}"],
           ["total", f"{total:,}", ""]])
    print()
    kv("corpus size", millions(report["train"]["tokens"]) + " training tokens")
    kv("manual share", f"{fraction:.2%}, "
                       f"{'at or above' if fraction >= 0.20 else 'below'} the "
                       f"20% floor")
    kv("vocabulary used", f"{tf['distinct_tokens_used']:,} of {tok.size:,} "
                          f"pieces, {tf['vocab_coverage']:.1%}")
    print("\nmost frequent pieces")
    table(["piece", "count", "share of corpus"],
          [[r["piece"], f"{r['count']:,}",
            f"{r['count'] / max(report['train']['tokens'], 1):.2%}"]
           for r in tf["top_50"][:10]])


if __name__ == "__main__":
    main()
