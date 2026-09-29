"""Run Phase 2 end to end for one language, or both.

    python scripts/run_phase2.py --lang tel
    python scripts/run_phase2.py                  both languages, in sequence
    python scripts/run_phase2.py --skip-train     evaluate existing checkpoints

Training dominates the wall clock and is resumable, so an interrupted run picks
up from its last checkpoint rather than starting over.
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, ".")

from common.log import kv, section

STAGES = [
    ("19", "19_sanity.py", "parameters, ln(V) loss, causal mask"),
    ("20", "20_train.py", "pretrain"),
    ("21", "21_evaluate.py", "loss, perplexity, bits-per-byte"),
    ("22", "22_generate.py", "BLEU, chrF++, ROUGE-L, diversity"),
    ("23", "23_attention.py", "heatmaps, entropy, mean distance"),
]


def run(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    mins = (time.time() - t0) / 60
    print(f"  {'finished' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {mins:.1f} min", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["tel", "nep"], default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--work", default=None,
                    help="output root; defaults to the platform's work dir")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else ["tel", "nep"]
    py = sys.executable

    section("Phase 2")
    kv("languages", ", ".join(langs))
    kv("stages", ", ".join(s[0] for s in STAGES
                           if not (args.skip_train and s[0] == "20")))

    failed = []
    for lang in langs:
        for sid, script, desc in STAGES:
            if args.skip_train and sid == "20":
                continue
            section(f"{sid}  {desc}  [{lang}]")
            cmd = [py, f"scripts/{script}", "--lang", lang]
            if args.device:
                cmd += ["--device", args.device]
            # 19_sanity builds a model from the config alone and has no --work
            if args.work and sid != "19":
                cmd += ["--work", args.work]
            if sid == "20" and args.max_steps:
                cmd += ["--max-steps", args.max_steps]
            if not run(cmd):
                failed.append(f"{sid}/{lang}")
                print("  stopping: later stages depend on this one")
                break

    section("24  assemble the report")
    run([py, "scripts/24_phase2_report.py"])

    section("Phase 2 finished")
    if failed:
        kv("stages that failed", ", ".join(failed))
    else:
        kv("status", "all stages completed")
    print("\n  Left to write by hand: the H-vs-L discussion, the attention\n"
          "  interpretation, and the resource-level comparison in\n"
          "  report/phase2_report.md.")


if __name__ == "__main__":
    main()
