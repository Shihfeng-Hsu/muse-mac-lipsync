#!/usr/bin/env bash
# setup_mlx_musetalk.sh
# One-shot setup of MuseTalk 1.5 (MLX port) for Apple Silicon Macs.
#
# Layout created under PROJECT_DIR (default: the folder containing this script):
#   .venv/                       python virtualenv (torch is CPU-only here)
#   musetalk-mlx/                MLX inference code (xocialize/musetalk-mlx)
#   musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16/   MLX weights (from HF mlx-community)
#   musetalk-mlx/weights/dwpose/ DWPose ONNX detectors (yzd-v/DWPose)
#   refs/MuseTalk/               upstream MuseTalk (blending + S3FD code only)
#   refs/MuseTalk/models/face-parse-bisent/    bisenet blending weights
#
# Usage:
#   ./setup_mlx_musetalk.sh [PROJECT_DIR]
#
# All code and comments are pure ASCII by design.

set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "== MuseTalk-MLX setup =="
echo "Project dir: $PROJECT_DIR"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# bring the companion scripts (bench_mlx.py, run_lipsync.sh) into the project
for HELPER in bench_mlx.py run_lipsync.sh ; do
  if [ -f "$SCRIPT_DIR/$HELPER" ]; then
    cp -f "$SCRIPT_DIR/$HELPER" "$PROJECT_DIR/$HELPER" 2>/dev/null || true
    chmod +x "$PROJECT_DIR/$HELPER" 2>/dev/null || true
  fi
done

# ---------------------------------------------------------------- 1. python
PY_BIN=""
for CAND in python3.12 python3.11 python3.10 python3; do
  if command -v "$CAND" >/dev/null 2>&1; then
    VER="$("$CAND" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    MAJOR="${VER%%.*}"; MINOR="${VER#*.}"
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
      PY_BIN="$CAND"
      echo "Using python: $("$PY_BIN" -V 2>&1) ($("$PY_BIN" -c 'import sys; print(sys.executable)'))"
      break
    fi
  fi
done
if [ -z "$PY_BIN" ]; then
  echo "ERROR: no python >= 3.10 found. Install with:  brew install python@3.12"
  exit 1
fi

# ---------------------------------------------------------------- 2. ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found. Video assembly needs it."
  echo "         Install with:  brew install ffmpeg"
fi

# ---------------------------------------------------------------- 3. repos
[ -d musetalk-mlx ] || git clone --depth 1 https://github.com/xocialize/musetalk-mlx
mkdir -p refs
[ -d refs/MuseTalk ] || git clone --depth 1 https://github.com/TMElyralab/MuseTalk refs/MuseTalk

# ---------------------------------------------------------------- 4. venv + deps
if [ ! -d .venv ]; then
  "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet

echo "-- installing python deps (this pulls mlx + cpu torch, ~10 min first time) --"
pip install --quiet -e ./musetalk-mlx
pip install --quiet rtmlib onnxruntime gdown
# torch/torchvision are needed only for S3FD face detection + bisenet blending.
# Default macOS wheels are CPU+MPS, exactly what we want here.
# NOTE: check the venv interpreter ("python"), not PY_BIN (system python).
python -c "import torch" 2>/dev/null || pip install --quiet torch torchvision

# ---------------------------------------------------------------- 5. weights
python - <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

root = Path.cwd()
ok = True

# 5a. MLX snapshot (the actual inference weights, ~1.9 GB)
dest = root / "musetalk-mlx" / "dist" / "MuseTalk-1.5-MLX-fp16"
try:
    print("[1/3] mlx-community/MuseTalk-1.5-fp16 ->", dest)
    snapshot_download("mlx-community/MuseTalk-1.5-fp16", local_dir=str(dest))
except Exception as e:
    ok = False
    print("FAILED to download MLX weights:", e)

# 5b. DWPose ONNX detectors (landmarks)
dest = root / "musetalk-mlx" / "weights" / "dwpose"
dest.mkdir(parents=True, exist_ok=True)
try:
    print("[2/3] yzd-v/DWPose onnx ->", dest)
    snapshot_download(
        "yzd-v/DWPose", local_dir=str(dest),
        allow_patterns=["yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
    )
except Exception as e:
    ok = False
    print("FAILED to download DWPose onnx:", e)

# 5c. bisenet face-parsing weights (needed by upstream blending code)
dest = root / "refs" / "MuseTalk" / "models" / "face-parse-bisent"
dest.mkdir(parents=True, exist_ok=True)
target = dest / "79999_iter.pth"
if not target.exists():
    try:
        print("[3/3] bisenet 79999_iter.pth (google drive) ->", target)
        import gdown
        gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=str(target), quiet=False)
    except Exception as e:
        print("WARNING: gdown failed (%s)." % e)
        print("  Manual fallback: download 79999_iter.pth from the upstream MuseTalk")
        print("  README (face-parse-bisent section) and place it at:")
        print("    %s" % target)

resnet = dest / "resnet18-5c106cde.pth"
if not resnet.exists():
    try:
        print("[3/3] resnet18 backbone ->", resnet)
        import urllib.request
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth", str(resnet))
    except Exception as e:
        print("WARNING: resnet18 download failed:", e)

print("WEIGHTS DONE" if ok else "WEIGHTS INCOMPLETE (see messages above)")
PY

# ---------------------------------------------------------------- 6. verify layout
echo "-- verifying expected files --"
MISS=0
for F in \
  musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16/unet.safetensors \
  musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16/vae.safetensors \
  musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16/whisper_encoder.safetensors \
  musetalk-mlx/weights/dwpose/dw-ll_ucoco_384.onnx \
  musetalk-mlx/weights/dwpose/yolox_l.onnx \
  refs/MuseTalk/models/face-parse-bisent/79999_iter.pth \
  refs/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth ; do
  if [ -f "$F" ]; then echo "  ok    $F"; else echo "  MISS  $F"; MISS=1; fi
done
# S3FD weights auto-download on first use (adrianbulat.com), just report presence
if [ -f refs/MuseTalk/musetalk/utils/face_detection/detection/sfd/s3fd.pth ]; then
  echo "  ok    S3FD weights (bundled)"
else
  echo "  note  S3FD weights not bundled -> auto-downloaded on first landmarks run"
fi

# ---------------------------------------------------------------- 7. benchmark
echo "-- running MLX benchmark (faces/sec on this chip) --"
if python "$PROJECT_DIR/bench_mlx.py" musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16 ; then
  :
else
  echo "WARNING: benchmark failed. The download part is done; investigate the"
  echo "         benchmark separately with:"
  echo "           source .venv/bin/activate"
  echo "           python bench_mlx.py musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16"
fi

# ---------------------------------------------------------------- 8. summary
echo ""
echo "== setup finished =="
if [ "$MISS" -eq 0 ]; then
  echo "All expected files are in place. To generate a lip-synced video:"
  echo "  cd $PROJECT_DIR"
  echo "  ./run_lipsync.sh <source.mp4> <audio.wav> <out.mp4>"
else
  echo "Some files are MISSING (see MISS lines above). Fix them, then re-run this"
  echo "script -- it is idempotent and will only fetch what is absent."
fi