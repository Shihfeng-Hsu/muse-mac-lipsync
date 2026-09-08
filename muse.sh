#!/usr/bin/env bash
# muse.sh - multi-avatar front end for the local MuseTalk-MLX pipeline.
#
# Layout (project dir, defaults to the folder containing this script):
#   avatars/<name>/frames/       extracted source frames (heavy, reusable cache)
#   avatars/<name>/coords.pkl    landmark cache for this avatar
#   avatars/<name>/coords.meta   source-video signature (path:size)
#   avatars/<name>/profile.json  name / fps / frame count / source path
#   inbox/                       drop episode audio files (mp3/wav/m4a) here
#   outbox/                      finished videos land here (batch mode)
#
# The landmark cache (coords + frames) is expensive to build (CPU, minutes)
# and cheap to reuse (same avatar, any audio). It is bound to the avatar.
# Swapping audio files never re-runs landmark detection.
#
# Commands:
#   muse.sh avatar-add <name> <video.mp4>    one-time landmark extraction (slow)
#   muse.sh avatar-adopt <name> <video.mp4>  migrate the existing project-root
#                                            coords/frames cache into avatars/<name>
#   muse.sh set-source <name> <video.mp4>    update the recorded source video
#   muse.sh list                             show avatars and cache state
#   muse.sh render <name> <audio.(mp3|wav|m4a)> <out.mp4>
#   muse.sh batch <name> <outdir>            render every audio file in inbox/
#
# Tip: run long jobs under caffeinate:  caffeinate -is muse.sh batch <name>
# All code and comments are pure ASCII by design.

set -euo pipefail

PROJECT_DIR="${MUSE_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="$PROJECT_DIR/.venv/bin/python"
CHUNKS_SH="$PROJECT_DIR/run_lipsync_chunks.sh"
TOOLS="$PROJECT_DIR/muse_tools.py"

usage() {
  cat <<'EOF'
usage: muse.sh <command> [args]

  avatar-add <name> <video.mp4>              one-time landmark extraction (slow)
  avatar-adopt <name> <video.mp4>            migrate project-root cache to avatar
  set-source <name> <video.mp4>              update the recorded source video
  list                                       show avatars and cache state
  render <name> <audio.(mp3|wav|m4a)> <out.mp4>
  batch <name> [outdir]                      render every audio file in inbox/
EOF
}

if [ ! -x "$PY" ]; then echo "ERROR: venv not found at $PY"; exit 1; fi
if [ ! -f "$CHUNKS_SH" ]; then echo "ERROR: $CHUNKS_SH not found"; exit 1; fi
if [ ! -f "$TOOLS" ]; then echo "ERROR: $TOOLS not found (copy it next to muse.sh)"; exit 1; fi

check_name() {
  case "$1" in ''|*[!A-Za-z0-9_-]*)
    echo "ERROR: avatar name must use only letters, digits, _ or -"; exit 1;; esac
}

avatar_dir() { printf '%s/avatars/%s' "$PROJECT_DIR" "$1"; }

source_video_of() {
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('source_video',''))" \
    "$1/profile.json"
}

require_avatar() {
  local dir
  dir="$(avatar_dir "$1")"
  if [ ! -f "$dir/coords.pkl" ]; then
    echo "ERROR: avatar '$1' has no landmark cache."
    echo "       run: muse.sh avatar-add $1 <video.mp4>"
    exit 1
  fi
}

cmd="${1:-}"
if [ -z "$cmd" ]; then usage; exit 1; fi
shift

case "$cmd" in

  avatar-add)
    [ "$#" -eq 2 ] || { echo "usage: muse.sh avatar-add <name> <video.mp4>"; exit 1; }
    NAME="$1"; SRC="$2"
    check_name "$NAME"
    [ -f "$SRC" ] || { echo "ERROR: video not found: $SRC"; exit 1; }
    DIR="$(avatar_dir "$NAME")"
    if [ -f "$DIR/coords.pkl" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "ERROR: avatar '$NAME' already has a landmark cache."
      echo "       use FORCE=1 to re-extract from scratch."
      exit 1
    fi
    mkdir -p "$DIR/frames"
    SRC_ABS="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
    echo "== avatar '$NAME': landmark extraction (CPU, one-time; slow) =="
    "$PY" "$PROJECT_DIR/musetalk-mlx/scripts/extract_landmarks.py" \
      "$SRC_ABS" "$DIR/frames" "$DIR/coords.pkl"
    SIG="$(stat -f '%N:%z' "$SRC_ABS" 2>/dev/null || stat -c '%N:%s' "$SRC_ABS")"
    printf '%s\n' "$SIG" > "$DIR/coords.meta"
    "$PY" "$TOOLS" profile "$NAME" "$SRC_ABS" "$DIR/coords.pkl" "$DIR/profile.json" >/dev/null
    echo "avatar ready: $DIR"
    ;;

  avatar-adopt)
    [ "$#" -eq 2 ] || { echo "usage: muse.sh avatar-adopt <name> <video.mp4>"; exit 1; }
    NAME="$1"; SRC="$2"
    check_name "$NAME"
    [ -f "$SRC" ] || { echo "ERROR: video not found: $SRC"; exit 1; }
    if [ ! -f "$PROJECT_DIR/coords.pkl" ]; then
      echo "ERROR: no project-root coords.pkl to adopt; use avatar-add instead."
      exit 1
    fi
    DIR="$(avatar_dir "$NAME")"
    mkdir -p "$DIR/frames"
    SRC_ABS="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
    echo "== avatar '$NAME': adopting existing project cache (fast) =="
    "$PY" "$TOOLS" migrate "$PROJECT_DIR/coords.pkl" "$DIR/coords.pkl" "$DIR/frames"
    if ls "$PROJECT_DIR"/frames/*.png >/dev/null 2>&1; then
      mv "$PROJECT_DIR"/frames/*.png "$DIR/frames/"
    fi
    if [ -f "$PROJECT_DIR/coords.meta" ]; then
      mv "$PROJECT_DIR/coords.meta" "$DIR/coords.meta"
    fi
    rm -f "$PROJECT_DIR/coords.pkl"
    rmdir "$PROJECT_DIR/frames" 2>/dev/null || true
    "$PY" "$TOOLS" profile "$NAME" "$SRC_ABS" "$DIR/coords.pkl" "$DIR/profile.json" >/dev/null
    echo "avatar ready: $DIR"
    ;;

  set-source)
    [ "$#" -eq 2 ] || { echo "usage: muse.sh set-source <name> <video.mp4>"; exit 1; }
    NAME="$1"; SRC="$2"
    check_name "$NAME"
    [ -f "$SRC" ] || { echo "ERROR: video not found: $SRC"; exit 1; }
    require_avatar "$NAME"
    DIR="$(avatar_dir "$NAME")"
    SRC_ABS="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
    SIG="$(stat -f '%N:%z' "$SRC_ABS" 2>/dev/null || stat -c '%N:%s' "$SRC_ABS")"
    printf '%s\n' "$SIG" > "$DIR/coords.meta"
    "$PY" "$TOOLS" profile "$NAME" "$SRC_ABS" "$DIR/coords.pkl" "$DIR/profile.json" >/dev/null
    echo "source updated for '$NAME': $SRC_ABS"
    ;;

  list)
    echo "project: $PROJECT_DIR"
    found=0
    for d in "$PROJECT_DIR"/avatars/*/; do
      [ -d "$d" ] || continue
      found=1
      NAME="$(basename "$d")"
      if [ -f "$d/profile.json" ]; then
        "$PY" -c "import json,sys; p=json.load(open(sys.argv[1])); print('  avatar %-12s fps=%.0f  frames=%d  src=%s' % (p['name'], p['fps'], p['n_frames'], p.get('source_video','?')))" "$d/profile.json" || true
      else
        echo "  avatar $NAME (no profile.json)"
      fi
      if [ -f "$d/coords.pkl" ]; then
        echo "    landmarks: cached"
      else
        echo "    landmarks: MISSING (run avatar-add)"
      fi
    done
    if [ "$found" = "0" ]; then
      echo "  (no avatars yet; try: muse.sh avatar-add <name> <video.mp4>)"
    fi
    ;;

  render)
    [ "$#" -eq 3 ] || { echo "usage: muse.sh render <name> <audio.(mp3|wav|m4a)> <out.mp4>"; exit 1; }
    NAME="$1"; AUD="$2"; OUT="$3"
    require_avatar "$NAME"
    DIR="$(avatar_dir "$NAME")"
    [ -f "$DIR/profile.json" ] || { echo "ERROR: no profile.json in $DIR"; exit 1; }
    SRC="$(source_video_of "$DIR")"
    if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
      echo "ERROR: source video missing for '$NAME' ($SRC)"
      echo "       run: muse.sh set-source $NAME <video.mp4>"
      exit 1
    fi
    AVATAR_DIR="$DIR" bash "$CHUNKS_SH" "$SRC" "$AUD" "$OUT"
    ;;

  batch)
    [ "$#" -ge 1 ] || { echo "usage: muse.sh batch <name> [outdir]"; exit 1; }
    NAME="$1"; OUTDIR="${2:-$PROJECT_DIR/outbox}"
    require_avatar "$NAME"
    DIR="$(avatar_dir "$NAME")"
    [ -f "$DIR/profile.json" ] || { echo "ERROR: no profile.json in $DIR"; exit 1; }
    SRC="$(source_video_of "$DIR")"
    if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
      echo "ERROR: source video missing for '$NAME' ($SRC)"
      echo "       run: muse.sh set-source $NAME <video.mp4>"
      exit 1
    fi
    INBOX="$PROJECT_DIR/inbox"
    mkdir -p "$INBOX/done" "$OUTDIR"
    OK=0; FAIL=0
    for f in "$INBOX"/*; do
      [ -f "$f" ] || continue
      case "$f" in
        *.mp3|*.MP3|*.wav|*.WAV|*.m4a|*.M4A|*.flac|*.FLAC|*.aac|*.AAC) ;;
        *) continue ;;
      esac
      base="$(basename "$f")"
      stem="${base%.*}"
      out="$OUTDIR/$stem.mp4"
      if [ -f "$out" ]; then
        echo "== batch: $base -> skip (output exists) =="
        mv "$f" "$INBOX/done/" 2>/dev/null || true
        continue
      fi
      echo "== batch: $base =="
      if AVATAR_DIR="$DIR" bash "$CHUNKS_SH" "$SRC" "$f" "$out"; then
        mv "$f" "$INBOX/done/"
        OK=$((OK + 1))
      else
        echo "== batch: FAILED $base (file left in inbox for retry) =="
        FAIL=$((FAIL + 1))
      fi
      sleep "${BATCH_GAP:-120}"
    done
    echo "batch done: ok=$OK fail=$FAIL  (outputs in $OUTDIR)"
    ;;

  *)
    usage
    exit 1
    ;;
esac