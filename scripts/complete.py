"""Build one language's corpus and tokenizer end to end, in a single command.

    python scripts/complete.py --lang tel
    python scripts/complete.py --lang nep

Designed to be run for the two languages in parallel, in separate notebooks:
they touch different directories and share nothing, so neither waits on the
other.

What it does, in order:

    01   top up the downloaded corpora from HuggingFace, to whatever the
         sources.yaml targets now ask for
    02b  turn any crawled HTML still sitting in archives into text
    04   hold out validation and test, before any cleaning
    05   language id, Unicode normalisation, quality filters, PII
    06   deduplicate and decontaminate
    07   size the corpus to the token target
    08   train and sweep tokenizers
    09   encode to uint16 and count tokens exactly
    10   figures and tables

Every stage is resumable and every stage that rewrites its own output clears it
first, so an interrupted run can simply be started again.
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section

#: (id, script, extra args, description, best-effort)
#: Collection stages are best-effort: a source can be down or a language can
#: have no crawled HTML, and neither should stop a pipeline that has hours of
#: work behind it.
STAGES = [
    ("01", "01_download_hf.py", [], "download public corpora", True),
    ("02b", "02b_extract.py", [], "extract crawled HTML to text", True),
    ("04", "04_split.py", [], "hold out val/test before cleaning", False),
    ("05", "05_clean.py", ["--split", "EACH"], "language id, normalise, filter", False),
    ("06", "06_dedup.py", [], "deduplicate and decontaminate", False),
    ("07", "07_finalize.py", [], "size the corpus", False),
    ("08", "08_train_tokenizers.py", [], "train and sweep tokenizers", False),
    ("09", "09_tokenize.py", [], "encode, count tokens exactly", False),
    ("10", "10_stats.py", [], "figures and tables", False),
]


def run(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    mins = (time.time() - t0) / 60
    print(f"  {'done' if rc == 0 else f'FAILED (exit {rc})'} in {mins:.1f} min",
          flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--target-tokens", type=float, default=500e6,
                    help="corpus size to aim for; stage 07 keeps everything if "
                         "the corpus is smaller than this")
    ap.add_argument("--fast-dedup", action="store_true",
                    help="exact-only deduplication instead of the MinHash pass; "
                         "about an hour rather than five, at the cost of leaving "
                         "near-duplicates in the corpus")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from-stage", default=None,
                    help="start here, e.g. 05")
    ap.add_argument("--quick-tokenizer", action="store_true",
                    help="one algorithm and one vocabulary size")
    ap.add_argument("--vocab-sizes", default=None,
                    help="override the swept vocabulary sizes, comma separated")
    ap.add_argument("--algorithms", default=None,
                    help="override the swept algorithms, comma separated")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = work_dir()
    py = sys.executable

    stages = list(STAGES)
    if args.fast_dedup:
        stages = [("06", "06b_dedup_exact.py", [], "exact deduplication", False)
                  if s[0] == "06" else s for s in stages]
    if args.skip_download:
        stages = [s for s in stages if s[0] != "01"]
    if args.from_stage:
        ids = [s[0] for s in stages]
        if args.from_stage not in ids:
            raise SystemExit(f"unknown stage {args.from_stage}; have {ids}")
        stages = stages[ids.index(args.from_stage):]

    section(f"Building the {lang_dir(args.lang)} corpus")
    kv("work directory", work)
    kv("token target", f"{args.target_tokens / 1e6:.0f}M")
    kv("deduplication", "exact only" if args.fast_dedup else "exact + MinHash")
    kv("stages", ", ".join(s[0] for s in stages))

    failed = []
    for sid, script, extra, desc, best_effort in stages:
        section(f"{sid}  {desc}")
        base = [py, f"scripts/{script}", "--lang", args.lang]
        if sid == "07":
            base += ["--target-tokens", args.target_tokens]
        if sid == "08":
            if args.quick_tokenizer:
                base += ["--skip-scratch-validation"]
            algos = args.algorithms or ("unigram" if args.quick_tokenizer else None)
            vocabs = args.vocab_sizes or ("32000" if args.quick_tokenizer else None)
            if algos:
                base += ["--algorithms", algos]
            if vocabs:
                base += ["--vocab-sizes", vocabs]

        if extra[:2] == ["--split", "EACH"]:
            ok = all(run(base + ["--split", s])
                     for s in ("train", "val", "test"))
        else:
            ok = run(base + extra)

        if not ok:
            failed.append(sid)
            if best_effort:
                print(f"  stage {sid} is best-effort, continuing", flush=True)
                continue
            section("Stopped")
            kv("failed at", f"{sid} ({desc})")
            print("  The error is above. Nothing after this stage was run,\n"
                  "  because it would build on missing input.")
            raise SystemExit(1)

    section(f"{lang_dir(args.lang)} finished")
    if failed:
        kv("best-effort stages skipped", ", ".join(failed))
    try:
        import json
        from pathlib import Path

        rep = Path(f"{work}/{lang_dir(args.lang)}/6_tokenized/token_report.json")
        if rep.exists():
            r = json.loads(rep.read_text())
            kv("training tokens", f"{r['train']['tokens']:,}")
            kv("documents", f"{r['train']['documents']:,}")
            kv("manual share", f"{r.get('manual_fraction_train', 0):.2%}")
            kv("tokenizer", f"{r.get('algorithm')} at {r.get('vocab_size'):,}")
    except Exception:
        pass
    print("\n  When the other language has finished too, run once:\n"
          "    python scripts/11_dataset_card.py\n"
          "    python scripts/12_package.py")


if __name__ == "__main__":
    main()
