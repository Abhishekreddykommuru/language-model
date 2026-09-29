"""Platform detection, so the same code runs on Colab, Kaggle or a laptop."""
import os
import shutil
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/lma_phase1"


def detect_platform() -> str:
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.path.isdir("/content"):
        return "colab"
    return "local"


def work_dir() -> str:
    """Durable output root. Override with LMA_WORK."""
    if os.environ.get("LMA_WORK"):
        return os.environ["LMA_WORK"]
    return {
        "colab": DRIVE_ROOT,
        "kaggle": "/kaggle/working/lma_phase1",
        "local": os.path.abspath("./work"),
    }[detect_platform()]


def scratch_dir() -> str:
    """Fast local disk for staging, raw HTML and downloads.

    Corpus shards are never appended to directly on the Drive mount: an open
    file there may not sync, and reads back as 0 bytes in a later session.
    """
    d = {
        "colab": "/content/scratch",
        "kaggle": "/tmp/scratch",
        "local": os.path.abspath("./scratch"),
    }[detect_platform()]
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def ensure_drive_mounted():
    if detect_platform() == "colab" and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive

        drive.mount("/content/drive")
    Path(work_dir()).mkdir(parents=True, exist_ok=True)


def free_local_gb() -> float:
    return shutil.disk_usage(scratch_dir()).free / 1e9


def cleanup_scratch(keep_staging: bool = True):
    """Delete local scratch to reclaim disk. Drive artifacts are untouched."""
    for child in Path(scratch_dir()).iterdir():
        if keep_staging and child.name == "staging":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    print(f"  scratch cleared, {free_local_gb():.1f} GB free", flush=True)


def lang_dir(lang: str) -> str:
    return {"tel": "telugu", "nep": "nepali"}[lang]
