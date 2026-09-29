"""Phase 3 on Kaggle, from a cold notebook to a packaged result, in one command.

Fetches the pretrained checkpoints and tokenizers from Drive, checks that each
tokenizer matches the checkpoint it will be used with, runs stages 30 to 33 for
both languages, optionally runs the bonus ablation, assembles the report and
tars up everything worth downloading.

    python scripts/kaggle_phase3.py \
        --tel-checkpoint <drive link> \
        --nep-checkpoint <drive link> \
        --tel-tokenizer  <drive link>

Every download is skipped if the file is already in place, so re-running after a
session dies costs nothing. Training itself resumes from its last checkpoint.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from common.log import kv, section, warn

PACKAGES = ["sentencepiece", "pyyaml", "gdown", "sacrebleu"]


def sh(cmd: str, check: bool = True) -> int:
    rc = subprocess.run(cmd, shell=True).returncode
    if rc and check:
        raise SystemExit(f"command failed (exit {rc}): {cmd}")
    return rc


def run_stage(cmd: list) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    print(f"  {'finished' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc == 0


# ---------------- Drive ----------------

def drive_id(link: str) -> str:
    """The file or folder id, whether given a full URL or a bare id."""
    for pattern in (r"/d/([A-Za-z0-9_-]{15,})",
                    r"/folders/([A-Za-z0-9_-]{15,})",
                    r"[?&]id=([A-Za-z0-9_-]{15,})"):
        found = re.search(pattern, link)
        if found:
            return found.group(1)
    return link.strip()


def size_of(path: Path) -> str:
    n = path.stat().st_size
    return f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def fetch_file(link: str, dest: str, min_mb: float, force: bool = False) -> Path:
    """One Drive file, with a size guard.

    A Drive link that is not shared returns an HTML page and gdown saves it
    happily. That page is a few tens of kilobytes, so a size floor catches it
    here rather than an hour later inside a training run.

    `force` overwrites a file that is already there. Checkpoints land in the
    work directory, where their presence means a previous run fetched them, so
    those are left alone. Tokenizers land in the repository, which ships one
    already -- so an existing file there is no evidence at all that it is the
    right one, and skipping the download would silently keep the wrong
    vocabulary.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > min_mb * 1e6 and not force:
        kv(dest.name, f"already here, {size_of(dest)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(f'gdown "https://drive.google.com/uc?id={drive_id(link)}" -O "{dest}"')
    mb = dest.stat().st_size / 1e6 if dest.exists() else 0
    if mb < min_mb:
        raise SystemExit(
            f"\n{dest} came back as {mb:.2f} MB, under the {min_mb} MB floor.\n"
            f"Drive returned an HTML page rather than the file. Open the link, "
            f"set sharing to 'Anyone with the link', and run this again.")
    kv(dest.name, size_of(dest))
    return dest


def fetch_folder(link: str, dest: str) -> Path:
    """Every file in a Drive folder, flattened into `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    sh(f'gdown --folder "https://drive.google.com/drive/folders/'
       f'{drive_id(link)}" -O "{dest}" --remaining-ok')
    # gdown --folder recreates the folder name underneath the target, so pull
    # anything one level down back up to where the pipeline expects it.
    for child in list(dest.iterdir()):
        if child.is_dir():
            for f in child.iterdir():
                f.rename(dest / f.name)
    kv(dest.name, ", ".join(sorted(p.name for p in dest.iterdir())) or "empty")
    return dest


# ---------------- checks ----------------

def preflight(work: str, langs: list):
    """Refuse to spend GPU time on a tokenizer that does not match its model."""
    import torch
    import sentencepiece as spm

    from common.io.paths import lang_dir

    section("Preflight")
    problems = []
    for lang in langs:
        langdir = lang_dir(lang)
        ckpt = Path(f"{work}/{langdir}/7_checkpoints/{lang}_best.pt")
        model_file = Path(f"{langdir}/tokenizer/{lang}.model")
        if not ckpt.exists():
            problems.append(f"{lang}: no pretrained checkpoint at {ckpt}")
            continue
        if not model_file.exists():
            problems.append(f"{lang}: no tokenizer at {model_file}")
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sp = spm.SentencePieceProcessor(model_file=str(model_file))
        v_ck = ck["model_config"]["vocab_size"]
        v_sp = sp.get_piece_size()
        ok = v_ck == v_sp
        kv(f"{lang}", f"checkpoint {v_ck:,} pieces, tokenizer {v_sp:,}, "
                      f"step {ck['step']:,}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(
                f"{lang}: the checkpoint was trained with a {v_ck:,}-piece "
                f"tokenizer but {model_file} has {v_sp:,}. Token ids would mean "
                f"something different to the weights. Find the tokenizer that "
                f"produced this checkpoint.")
    if problems:
        raise SystemExit("\n\n" + "\n".join(f"  {p}" for p in problems) + "\n")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tel-checkpoint", default=None,
                    help="Drive link to tel_best.pt")
    ap.add_argument("--nep-checkpoint", default=None,
                    help="Drive link to nep_best.pt")
    ap.add_argument("--tel-tokenizer", default=None,
                    help="Drive link to the tel.model that matches the "
                         "Telugu checkpoint")
    ap.add_argument("--nep-tokenizer", default=None,
                    help="only needed if the repository copy is wrong")
    ap.add_argument("--tel-tokenizer-folder", default=None,
                    help="Drive folder holding tel.model and friends, instead "
                         "of a single file link")
    ap.add_argument("--tel-tokenized-folder", default=None,
                    help="Drive folder holding train/val/test.bin and "
                         "token_report.json; only the ablation needs it")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the bonus no-positional-embedding retrain")
    ap.add_argument("--ablation-steps", type=int, default=None,
                    help="shorter ablation budget; the comparison is then made "
                         "at the matching step of the standard run")
    ap.add_argument("--lang", choices=["tel", "nep"], default=None,
                    help="one language only; default is both")
    ap.add_argument("--work", default=None)
    ap.add_argument("--skip-setup", action="store_true",
                    help="packages and fonts are already installed")
    ap.add_argument("--skip-finetune", action="store_true",
                    help="evaluate checkpoints that already exist")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else ["tel", "nep"]
    py = sys.executable
    t_start = time.time()

    # ---- setup ----
    if not args.skip_setup:
        section("Setup")
        sh(f"pip install -q {' '.join(PACKAGES)}", check=False)
        # fonts-noto-core carries Telugu and Devanagari. Without it the
        # attention heatmaps label their axes with empty boxes.
        sh("apt-get -qq update > /dev/null && "
           "apt-get -qq install -y fonts-noto-core > /dev/null", check=False)
        sh("rm -rf ~/.cache/matplotlib", check=False)
        import torch
        kv("torch", torch.__version__)
        kv("gpu", torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "none, this will be very slow")

    from common.io.paths import lang_dir, work_dir
    work = args.work or work_dir()
    kv("work directory", work)

    # ---- downloads ----
    section("Fetching from Drive")
    links = [("tel", args.tel_checkpoint, f"{work}/telugu/7_checkpoints/tel_best.pt", 100, False),
             ("nep", args.nep_checkpoint, f"{work}/nepali/7_checkpoints/nep_best.pt", 100, False),
             ("tel", args.tel_tokenizer, "telugu/tokenizer/tel.model", 0.05, True),
             ("nep", args.nep_tokenizer, "nepali/tokenizer/nep.model", 0.05, True)]
    for lang, link, dest, floor, force in links:
        if link and lang in langs:
            fetch_file(link, dest, floor, force)
    if args.tel_tokenizer_folder:
        fetch_folder(args.tel_tokenizer_folder, "telugu/tokenizer")
    if args.tel_tokenized_folder:
        fetch_folder(args.tel_tokenized_folder, f"{work}/telugu/6_tokenized")

    preflight(work, langs)

    # ---- the pipeline ----
    cmd = [py, "scripts/run_phase3.py", "--work", work]
    if args.lang:
        cmd += ["--lang", args.lang]
    if args.skip_finetune:
        cmd += ["--skip-finetune"]
    # A shortened ablation is run separately below, so it is not handed to
    # run_phase3 here -- that would train the full budget first.
    if args.ablation and not args.ablation_steps:
        cmd += ["--ablation", "--ablation-lang", "tel"]
    ok = run_stage(cmd)

    if args.ablation and args.ablation_steps:
        section("Bonus ablation, shortened budget")
        ok &= run_stage([py, "scripts/34_ablation.py", "--lang", "tel",
                         "--work", work, "--max-steps", args.ablation_steps])
        run_stage([py, "scripts/35_final_report.py", "--work", work])

    # ---- package ----
    section("Packaging")
    out = "/kaggle/working" if Path("/kaggle/working").is_dir() else "."
    wanted = ["report/tables", "report/figures", "report/final_report.md",
              "report/final_report_generated.md",
              "telugu/configs/finetune_trained.yaml",
              "nepali/configs/finetune_trained.yaml",
              "telugu/tokenizer/tel.model", "nepali/tokenizer/nep.model"]
    present = [p for p in wanted if Path(p).exists()]
    sh(f"tar czf {out}/phase3_results.tar.gz {' '.join(present)}", check=False)
    for lang in langs:
        for tag in ("ft", "nopos"):
            src = Path(f"{work}/{lang_dir(lang)}/7_checkpoints/{lang}_{tag}_best.pt")
            if src.exists():
                sh(f'cp "{src}" {out}/', check=False)
    sh(f"ls -lh {out}/phase3_results.tar.gz {out}/*_best.pt 2>/dev/null",
       check=False)

    section("Done" if ok else "Finished with failures")
    kv("wall clock", f"{(time.time() - t_start) / 60:.1f} min")
    kv("results", f"{out}/phase3_results.tar.gz")
    if not ok:
        warn("at least one stage failed; read the log above before downloading")
    print("\n  Download phase3_results.tar.gz and the *_best.pt files from the\n"
          "  Kaggle Output panel. The checkpoints go to Drive, and their links\n"
          "  go in README.md.")


if __name__ == "__main__":
    main()
