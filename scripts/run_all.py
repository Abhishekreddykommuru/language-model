"""Run the whole of Phase 1 in one command, for both languages.

    python scripts/run_all.py

Every stage is checkpointed, so re-running after a lost session picks up where
it stopped rather than recomputing.

    --from 05            start at stage 05, skipping collection
    --only 08,09,10      run just these stages
    --lang tel           one language only
    --skip-crawl         public corpora only, no crawl or OCR
    --dry-run            print the plan without running anything

Crawl and OCR are the slow stages, so --from 05 is the one to use when
re-running the analysis after changing a threshold.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from common.io.paths import (ensure_drive_mounted, free_local_gb, lang_dir,
                             work_dir)
from common.log import kv, section

#: (id, script, per-language, extra args, human description, rough minutes)
STAGES = [
    ("01", "01_download_hf.py", True, [], "download public corpora", 90),
    ("02", "02_crawl.py", True, ["--max-pages-per-host", "40000"],
     "crawl your own pages", 450),
    ("02b", "02b_extract.py", True, [], "extract article text from HTML", 60),
    ("03", "03_ocr.py", True, [], "OCR scanned books", 105),
    ("03b", "03b_pdf_text.py", True, [], "extract text-layer PDFs", 45),
    ("04", "04_split.py", True, [], "hold out val/test BEFORE cleaning", 10),
    ("05", "05_clean.py", True, ["--split", "ALL"], "language ID + clean", 90),
    ("06", "06_dedup.py", True, [], "deduplicate + decontaminate", 80),
    ("07", "07_finalize.py", True, [], "finalise + enforce manual floor", 15),
    ("08", "08_train_tokenizers.py", True, [], "train all three tokenizers", 100),
    ("09", "09_tokenize.py", True, [], "encode corpus, exact token counts", 15),
    ("10", "10_stats.py", True, [], "tables and figures", 10),
    ("11", "11_dataset_card.py", False, [], "write DATASET_CARD.md", 2),
]

SLOW_STAGES = {"02", "02b", "03", "03b"}

#: Collection depends on the outside world: a site can be down, a source empty.
#: A failure there must not abort a pipeline that has run for hours, since stage
#: 07 sizes the corpus to whatever was actually collected. Analysis stages still
#: stop on failure, where a failure means a real bug.
BEST_EFFORT_STAGES = {"02", "02b", "03", "03b"}


def run(cmd: list, dry: bool) -> bool:
    """Run one subprocess, streaming its output. Returns True on success."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return True
    t0 = time.time()
    result = subprocess.run(cmd)
    mins = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"  failed after {mins:.1f} min, exit code {result.returncode}",
              flush=True)
        return False
    print(f"  finished in {mins:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["tel", "nep"], default=None,
                    help="one language only (default: both)")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="start from this stage id, e.g. 05")
    ap.add_argument("--only", default=None,
                    help="comma-separated stage ids to run, e.g. 08,09,10")
    ap.add_argument("--skip-crawl", action="store_true",
                    help="skip crawl, OCR and PDF collection")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going if a stage fails")
    ap.add_argument("--crawl-pages", type=int, default=None,
                    help="pages per host: the main time knob. 40000 (default) "
                         "is ~7.5 h/language across 10 Telugu hosts; 20000 "
                         "halves both the time and the corpus.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else ["tel", "nep"]

    stages = list(STAGES)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        stages = [s for s in stages if s[0] in wanted]
    elif args.from_stage:
        ids = [s[0] for s in stages]
        if args.from_stage not in ids:
            raise SystemExit(f"unknown stage {args.from_stage}; valid: {ids}")
        stages = stages[ids.index(args.from_stage):]
    if args.skip_crawl:
        stages = [s for s in stages if s[0] not in SLOW_STAGES]
    if args.crawl_pages:
        # Crawl time is linear in pages fetched, so scale the estimate with it.
        scale = args.crawl_pages / 40000
        stages = [(i, sc, pl,
                   (["--max-pages-per-host", str(args.crawl_pages)] if i == "02" else ex),
                   d, int(m * scale) if i == "02" else m)
                  for i, sc, pl, ex, d, m in stages]

    ensure_drive_mounted()
    total_min = sum(s[5] for s in stages) * len(langs)
    section("Phase 1 pipeline")
    kv("languages", ", ".join(langs))
    kv("work directory", work_dir())
    kv("local disk free", f"{free_local_gb():.1f} GB")
    kv("stages", ", ".join(s[0] for s in stages))
    kv("rough estimate", f"{total_min / 60:.1f} hours, mostly crawl and OCR")

    failures = []
    for stage_id, script, per_lang, extra, desc, _mins in stages:
        section(f"Stage {stage_id}, {desc}")
        for lang in (langs if per_lang else [None]):
            base = [sys.executable, f"scripts/{script}"]
            if lang:
                base += ["--lang", lang]
            best_effort = stage_id in BEST_EFFORT_STAGES or args.continue_on_error

            # 05_clean runs once per split.
            if extra[:2] == ["--split", "ALL"]:
                for split in ("train", "val", "test"):
                    if not run(base + ["--split", split], args.dry_run):
                        failures.append(f"{stage_id}/{lang}/{split}")
                        if not best_effort:
                            raise SystemExit(1)
            else:
                if not run(base + extra, args.dry_run):
                    failures.append(f"{stage_id}/{lang}")
                    if not best_effort:
                        raise SystemExit(1)
                    print(f"  stage {stage_id} is best-effort, continuing",
                          flush=True)

    section("Pipeline finished")
    if failures:
        kv("stages that failed", ", ".join(failures))
        print("  Collection stages are best-effort, so this is often fine.")
        print("  Check the token counts below before worrying about it.")
    for lang in langs:
        report = Path(f"{work_dir()}/{lang_dir(lang)}/6_tokenized/"
                      f"token_report.json")
        if not report.exists():
            continue
        import json

        r = json.loads(report.read_text())
        kv(lang, f"{r['train']['tokens']:,} training tokens, "
                 f"{r['manual_fraction_train']:.1%} manual")


if __name__ == "__main__":
    main()
