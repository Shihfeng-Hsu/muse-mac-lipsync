#!/usr/bin/env bash
# run_lipsync.sh - offline lip-sync generation entry point (MLX route).
#
# Usage:
#   ./run_lipsync.sh <source_video.mp4> <audio.wav> <out.mp4> [variant]
#
#   source_video : the talking-person footage (frames are cycled if audio runs longer)
#   audio        : driving audio (wav); keep its length <= source video length
#   out          : output mp4 (video + muxed audio)
#   variant      : MLX weight variant, default fp16 (also q8 / q4 if downloaded)
#
# Steps: extract frames -> DWPose+S3FD landmarks -> MLX face generation ->
#         bisenet blend -> ffmpeg assemble + audio mux.
# All code and comments are pure ASCII by design.

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <source_video.mp4> <audio.wav> <out.mp4> [variant]"
  exit 1
fi

SRC="$1"
AUD="$2"
OUT="$3"
VAR="${4:-fp16}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
MLX="$ROOT/musetalk-mlx"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "ERROR: venv not found at $PY"
  echo "       run setup_mlx_musetalk.sh first."
  exit 1
fi
if [ ! -f "$SRC" ] || [ ! -f "$AUD" ]; then
  echo "ERROR: input video or audio not found."
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with:  brew install ffmpeg"
  exit 1
fi

mkdir -p "$ROOT/frames"
rm -f "$ROOT/frames/"*.png 2>/dev/null || true

echo "== step 1/2: landmarks (DWPose onnx + S3FD, CPU; first run downloads s3fd.pth) =="
"$PY" "$MLX/scripts/extract_landmarks.py" "$SRC" "$ROOT/frames" "$ROOT/coords.pkl"

echo "== step 2/2: MLX generation + blending + mux =="
"$PY" "$MLX/scripts/build_video.py" "$ROOT/coords.pkl" "$AUD" "$OUT" --variant "$VAR"

echo "done: $OUT"
echo ""
echo "QA tips:"
echo "  - watch mouth closure timing and teeth stability against the source audio"
echo "  - if edges look harsh, try: --parsing-mode full  (default is jaw)"
echo "  - if the crop feels tight, re-run build_video.py with --extra-margin 20"