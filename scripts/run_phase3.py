"""Run Phase 3 end to end for one language, or both.

    python scripts/run_phase3.py --lang tel
    python scripts/run_phase3.py                    both languages, in sequence
    python scripts/run_phase3.py --skip-finetune    evaluate existing checkpoints
    python scripts/run_phase3.py --ablation         also run the bonus ablation

Finetuning dominates the wall clock and is resumable, so an interrupted run
picks up from its last checkpoint rather than starting over. Stages run in
order and a failure stops that language, because every later stage reads what
an earlier one wrote.
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, ".")

from common.log import kv, section

STAGES = [
    ("30", "30_make_reasoning.py", "generate the reasoning dataset"),
    ("31", "31_finetune.py", "finetune from the pretrained checkpoint"),
    ("32", "32_eval_reasoning.py", "pretrained against finetuned accuracy"),
    ("33", "33_attention_compare.py", "attention before and after finetuning"),
]


def run(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    print(f"  {'finished' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["tel", "nep"], default=None)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--ablation", action="store_true",
                    help="also run the bonus no-position ablation on Telugu")
    ap.add_argument("--ablation-lang", default="tel", choices=["tel", "nep"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    langs = [args.lang] if args.lang else ["tel", "nep"]
    py = sys.executable

    section("Phase 3")
    kv("languages", ", ".join(langs))
    kv("stages", ", ".join(s[0] for s in STAGES
                           if not (args.skip_finetune and s[0] == "31")))
    kv("bonus ablation", args.ablation_lang if args.ablation else "not requested")

    failed = []
    for lang in langs:
        for sid, script, desc in STAGES:
            if args.skip_finetune and sid == "31":
                continue
            section(f"{sid}  {desc}  [{lang}]")
            cmd = [py, f"scripts/{script}", "--lang", lang]
            if args.work:
                cmd += ["--work", args.work]
            # Stage 30 only writes data files and has no device to run on.
            if args.device and sid != "30":
                cmd += ["--device", args.device]
            if sid == "31" and args.max_steps:
                cmd += ["--max-steps", args.max_steps]
            if not run(cmd):
                failed.append(f"{sid}/{lang}")
                print("  stopping: later stages depend on this one")
                break

    if args.ablation:
        section(f"34  bonus ablation  [{args.ablation_lang}]")
        cmd = [py, "scripts/34_ablation.py", "--lang", args.ablation_lang]
        if args.work:
            cmd += ["--work", args.work]
        if args.device:
            cmd += ["--device", args.device]
        if not run(cmd):
            failed.append("34/ablation")

    section("35  assemble the final report")
    cmd = [py, "scripts/35_final_report.py"]
    if args.work:
        cmd += ["--work", args.work]
    run(cmd)

    section("Phase 3 finished")
    if failed:
        kv("stages that failed", ", ".join(failed))
    else:
        kv("status", "all stages completed")
    print("\n  Left to write by hand: the discussion in "
          "report/final_report.md,\n"
          "  and the Drive links for the finetuned checkpoints in README.md.")


if __name__ == "__main__":
    main()
