"""Finish Phase 1 in one command.

Runs everything that is left: cleaning, deduplication, sizing, all three
tokenizers, the exact token counts, the figures, the dataset card, and finally
the packaging step that copies the deliverables into the repository and writes
the report.

    python scripts/finish.py                    finish from the collected data
    python scripts/finish.py --crawl-hours 4    crawl Nepali first, then finish
    python scripts/finish.py --quick            trim the tokenizer sweep

Every stage clears its own output before rewriting it, so this is safe to
re-run: interrupt it, run it again, and it rebuilds to the same numbers.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section

LANGS = ["tel", "nep"]

#: (script, per-language, extra args, description)
STEPS = [
    ("04_split.py", True, [], "hold out val and test before cleaning"),
    ("05_clean.py", True, ["--split", "EACH"], "language id, normalise, filter"),
    ("06_dedup.py", True, [], "deduplicate and decontaminate"),
    ("07_finalize.py", True, [], "size the corpus"),
    ("08_train_tokenizers.py", True, [], "train and sweep all three tokenizers"),
    ("09_tokenize.py", True, [], "encode, and count tokens exactly"),
    ("10_stats.py", True, [], "figures and tables"),
    ("11_dataset_card.py", False, [], "dataset card for the Drive folder"),
    ("12_package.py", False, [], "copy deliverables into the repo, write report"),
]


def run(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    mins = (time.time() - t0) / 60
    if rc != 0:
        print(f"  failed after {mins:.1f} min, exit code {rc}", flush=True)
        return False
    print(f"  finished in {mins:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=LANGS, default=None,
                    help="one language only; default is both")
    ap.add_argument("--crawl-hours", type=float, default=0.0,
                    help="crawl Nepali for this long first, to raise its manual "
                         "share, then extract and continue")
    ap.add_argument("--quick", action="store_true",
                    help="two algorithms and two vocabulary sizes instead of the "
                         "full sweep; about 40 minutes rather than 3 hours")
    ap.add_argument("--target-tokens", type=float, default=500e6,
                    help="corpus size to aim for in stage 07")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else LANGS
    ensure_drive_mounted()

    section("Finishing Phase 1")
    kv("languages", ", ".join(langs))
    kv("work directory", work_dir())
    kv("target", f"{args.target_tokens / 1e6:.0f}M tokens per language")
    if args.crawl_hours:
        kv("crawling Nepali first", f"{args.crawl_hours:.1f} h")
    if args.quick:
        kv("tokenizer sweep", "trimmed")

    py = sys.executable
    failed = []

    if args.crawl_hours:
        section(f"Collecting more Nepali, {args.crawl_hours:.1f} h")
        run([py, "scripts/02_crawl.py", "--lang", "nep",
             "--max-hours", args.crawl_hours])
        run([py, "scripts/02b_extract.py", "--lang", "nep"])

    for script, per_lang, extra, desc in STEPS:
        section(f"{script.split('_')[0]}  {desc}")
        for lang in (langs if per_lang else [None]):
            base = [py, f"scripts/{script}"]
            if lang:
                base += ["--lang", lang]
            if script == "07_finalize.py":
                base += ["--target-tokens", args.target_tokens]
            if script == "08_train_tokenizers.py" and args.quick:
                base += ["--skip-scratch-validation",
                         "--algorithms", "unigram,bpe",
                         "--vocab-sizes", "16000,32000"]

            # 05 runs once per split.
            if extra[:2] == ["--split", "EACH"]:
                ok = all(run(base + ["--split", s])
                         for s in ("train", "val", "test"))
            else:
                ok = run(base + extra)
            if not ok:
                # Every step here is an analysis stage, so a failure means a
                # real bug. Continuing produces an empty corpus, a tokenizer
                # trained on nothing, and a report full of zeros -- which is
                # worse than stopping, because it looks like a result.
                failed.append(f"{script} {lang or ''}".strip())
                section("Stopped: a stage failed")
                kv("failed at", failed[-1])
                print("  The error is in the output above. Nothing downstream")
                print("  is run, because it would silently produce zeros.")
                print("  Fix the cause, then re-run this command.")
                raise SystemExit(1)

    section("Phase 1 finished")
    if failed:
        kv("steps that failed", ", ".join(failed))
        print("  Re-run this command; completed stages are cheap to redo.")
    for lang in langs:
        rep = Path(f"{work_dir()}/{lang_dir(lang)}/6_tokenized/token_report.json")
        if not rep.exists():
            continue
        import json

        r = json.loads(rep.read_text())
        kv(lang, f"{r['train']['tokens']:,} training tokens, "
                 f"{r.get('manual_fraction_train', 0):.1%} manual, "
                 f"{r.get('algorithm', '?')} at {r.get('vocab_size', 0):,}")

    print("\n  Left to do by hand:")
    print("    1. Share the Drive folders and paste the links into README.md")
    print("    2. Run scripts/audit_filters.py, read the rejected documents,")
    print("       and write your judgements into report/phase1_report.md")
    print("    3. Commit and push the phase-1 branch")


if __name__ == "__main__":
    main()
