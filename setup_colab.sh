#!/usr/bin/env bash
# Run once per Colab session. Idempotent.
set -e

echo ">> system packages"
apt-get -qq update > /dev/null
apt-get -qq install -y zstd poppler-utils tesseract-ocr \
    tesseract-ocr-tel tesseract-ocr-nep \
    fonts-noto-core > /dev/null
# fonts-noto-core carries Telugu and Devanagari; without it the Phase 2
# attention heatmaps label their axes with empty boxes.

echo ">> python packages"
pip install -q -r requirements.txt 2>&1 | tail -2

echo ">> GlotLID model (language identification, 1.7 GB)"
mkdir -p models
python - <<'PY'
import os, shutil, sys
sys.path.insert(0, ".")
from pathlib import Path
from common.io.paths import work_dir

EXPECTED = 1_600_000_000          # ~1.69 GB; anything smaller is truncated
cache = Path(work_dir()) / "_models" / "glotlid.bin"
local = Path("models/glotlid.bin")

def ok(p):
    return p.exists() and p.stat().st_size > EXPECTED

# Drive cache survives between sessions; /content/code does not.
if ok(cache):
    print(f"using cached GlotLID from {cache}")
elif ok(local):
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local, cache)
else:
    from huggingface_hub import hf_hub_download
    print("downloading GlotLID (1.7 GB, a few minutes) ...")
    src = hf_hub_download("cis-lmu/glotlid", "model.bin")
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, cache)

if not ok(cache):
    print("WARNING: GlotLID download looks incomplete; language ID will fall "
          "back to script + function words")
else:
    if not ok(local):
        shutil.copy(cache, local)
    print(f"GlotLID ready ({cache.stat().st_size / 1e9:.2f} GB)")
PY

echo ">> verifying"
tesseract --list-langs | grep -E 'tel|nep' || echo "WARNING: Indic tesseract packs missing"
python -c "import sentencepiece, tokenizers, datasketch, trafilatura, zstandard; print('python deps OK')"
echo ">> setup complete"
