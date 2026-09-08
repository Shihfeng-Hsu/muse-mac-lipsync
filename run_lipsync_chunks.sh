#!/usr/bin/env bash
# run_lipsync_chunks.sh - full-length lip-sync render, chunked + cooled down.
#
# Renders the whole audio in fixed-length chunks (default 30 s of audio per
# chunk), resting between chunks (thermal pacing), then losslessly
# concatenates the chunks and muxes the full audio track.
#
# Chunk runs are frame-for-frame identical to one big render: the builder
# slices whisper features from the full audio encode and keeps head/body
# pose continuous via a global frame offset. Cutting position is visually
# clean anywhere because every output frame is generated independently.
#
# Resume support: completed chunks are kept and skipped. Ctrl-C any time
# (best during the cooldown sleep) and re-run the same command to continue.
#
# Usage:
#   caffeinate -is ./run_lipsync_chunks.sh <source.mp4> <audio.(wav|mp3|m4a)> <out.mp4>
#
# Env overrides:
#   CHUNK_SEC=30      seconds of audio rendered per chunk
#   COOLDOWN_SEC=300  rest between chunks (seconds)
#   VARIANT=fp16      MLX weight variant
#   RERUN=1           re-render existing chunks from scratch
#   AVATAR_DIR=path   landmark cache dir (default project root; set by muse.sh)
#   Runtime temp (converted audio, whisper cache, chunks, chunk frames) is
#   written under <project>/work/ so the project root stays clean.
#
# All code and comments are pure ASCII by design.

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <source.mp4> <audio.(wav|mp3|m4a)> <out.mp4>"
  echo "       run under caffeinate:  caffeinate -is $0 <src> <aud> <out>"
  exit 1
fi

SRC="$1"; AUD="$2"; OUT="$3"
CHUNK_SEC="${CHUNK_SEC:-30}"
COOLDOWN_SEC="${COOLDOWN_SEC:-300}"
VARIANT="${VARIANT:-fp16}"
RERUN="${RERUN:-0}"

case "$CHUNK_SEC" in ''|*[!0-9]*) echo "ERROR: CHUNK_SEC must be an integer"; exit 1;; esac
case "$COOLDOWN_SEC" in ''|*[!0-9]*) echo "ERROR: COOLDOWN_SEC must be an integer"; exit 1;; esac

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
BUILDER="$ROOT/build_video_chunk.py"
# all runtime temp (converted audio, whisper cache, chunk videos/frames)
# lives under work/ -> the project root stays clean
WORK="$ROOT/work"
mkdir -p "$WORK"

if [ ! -x "$PY" ]; then echo "ERROR: venv not found at $PY"; exit 1; fi
if [ ! -f "$BUILDER" ]; then echo "ERROR: builder not found at $BUILDER"; exit 1; fi
if [ ! -f "$SRC" ] || [ ! -f "$AUD" ]; then echo "ERROR: input video or audio not found."; exit 1; fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with:  brew install ffmpeg"; exit 1
fi

# builder chdir's internally -> every path we pass must be ABSOLUTE
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
OUT_DIR="$(dirname "$OUT")"; mkdir -p "$OUT_DIR"
OUT="$(cd "$OUT_DIR" && pwd)/$(basename "$OUT")"

# ---- audio: convert anything (mp3/m4a/...) to 16k mono wav ----
ORIG="$AUD"
AUD_EXT="$(echo "${AUD##*.}" | tr '[:upper:]' '[:lower:]')"
if [ "$AUD_EXT" = "wav" ]; then
  AUD="$(cd "$(dirname "$AUD")" && pwd)/$(basename "$AUD")"
else
  AUD_FULL="$WORK/audio_full.wav"
  if [ "$RERUN" = "1" ] || [ ! -f "$AUD_FULL" ] || [ "$AUD" -nt "$AUD_FULL" ]; then
    echo "== converting audio -> 16k mono wav =="
    ffmpeg -y -i "$AUD" -ar 16000 -ac 1 -c:a pcm_s16le "$AUD_FULL" >/dev/null 2>&1
  fi
  AUD="$AUD_FULL"
fi

DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$AUD")"
echo "audio duration: ${DUR}s"

# ---- landmarks: per-avatar cache (AVATAR_DIR defaults to the project root) ----
# Set AVATAR_DIR to avatars/<name> to bind the cache to one avatar (muse.sh
# does this). Cache matching accepts a moved/renamed source video (size part).
AVATAR_DIR="${AVATAR_DIR:-$ROOT}"
SRC_SIG="$(stat -f '%N:%z' "$SRC" 2>/dev/null || stat -c '%N:%s' "$SRC")"
CACHED=0
if [ -f "$AVATAR_DIR/coords.pkl" ] && [ -f "$AVATAR_DIR/coords.meta" ]; then
  OLD_SIG="$(cat "$AVATAR_DIR/coords.meta")"
  if [ "$OLD_SIG" = "$SRC_SIG" ]; then
    CACHED=1
  elif [ "${OLD_SIG##*:}" = "${SRC_SIG##*:}" ]; then
    printf '%s\n' "$SRC_SIG" > "$AVATAR_DIR/coords.meta"
    CACHED=1
  fi
fi
if [ "$CACHED" = "1" ]; then
  echo "== landmarks -> CACHED for this source video (skipped) =="
else
  echo "== landmarks (DWPose + S3FD on CPU; one-time per source video) =="
  mkdir -p "$AVATAR_DIR/frames"
  rm -f "$AVATAR_DIR/frames/"*.png 2>/dev/null || true
  "$PY" "$ROOT/musetalk-mlx/scripts/extract_landmarks.py" "$SRC" "$AVATAR_DIR/frames" "$AVATAR_DIR/coords.pkl"
  printf '%s\n' "$SRC_SIG" > "$AVATAR_DIR/coords.meta"
fi

FPS="$("$PY" -c "import pickle,sys; print(int(round(pickle.load(open(sys.argv[1],'rb'))['fps'])))" "$AVATAR_DIR/coords.pkl")"
if [ "$FPS" -lt 1 ]; then echo "ERROR: bad fps in coords.pkl"; exit 1; fi
CHUNK_FRAMES=$((CHUNK_SEC * FPS))
CHUNKS_EST=$(( ( ${DUR%%.*} * FPS + CHUNK_FRAMES - 1 ) / CHUNK_FRAMES ))
EST_MIN=$(( ( CHUNKS_EST * ( CHUNK_FRAMES * 4 / 3 + COOLDOWN_SEC ) ) / 60 ))
echo "plan: fps=$FPS  chunk=${CHUNK_SEC}s audio (~${CHUNK_FRAMES} frames)  rest=${COOLDOWN_SEC}s"
echo "rough estimate: ~${CHUNKS_EST} chunks, ~${EST_MIN} min wall time (heuristic)"

# chunks are keyed by the original audio content (md5): different episodes
# never reuse each other's segments, while re-runs of the same audio resume
AUD_KEY="$(md5 -q "$ORIG" 2>/dev/null || md5sum "$ORIG" 2>/dev/null | cut -d' ' -f1 || echo plain)"
[ -n "$AUD_KEY" ] || AUD_KEY=plain
CHUNK_DIR="$WORK/chunks/$AUD_KEY"; mkdir -p "$CHUNK_DIR"

# ---- render loop (resumable) ----
start=0
idx=0
while : ; do
  chunk_file="$CHUNK_DIR/$(printf 'chunk_%03d.mp4' "$idx")"
  if [ -s "$chunk_file" ] && [ "$RERUN" != "1" ]; then
    echo "== chunk $idx exists -> skip =="
  else
    echo "== chunk $idx: global frames [$start, +$CHUNK_FRAMES) =="
    "$PY" "$BUILDER" "$AVATAR_DIR/coords.pkl" "$AUD" "$chunk_file" \
      --variant "$VARIANT" --start-frame "$start" --num-frames "$CHUNK_FRAMES"
    if [ ! -s "$chunk_file" ]; then
      echo "== builder produced nothing -> all audio covered =="
      break
    fi
    rm -rf "$WORK"/chunk_frames_* 2>/dev/null || true
  fi
  got="$(cat "$chunk_file.frames")"
  if [ -z "$got" ] || [ "$got" -le 0 ]; then echo "ERROR: bad chunk $idx"; exit 1; fi
  if [ "$got" -lt "$CHUNK_FRAMES" ]; then
    echo "== chunk $idx was the tail ($got frames) -> done rendering =="
    break
  fi
  start=$((start + got))
  idx=$((idx + 1))
  if [ "$COOLDOWN_SEC" -gt 0 ]; then
    echo "== cooldown ${COOLDOWN_SEC}s (thermal pacing; Ctrl-C here is safe, resume later) =="
    sleep "$COOLDOWN_SEC"
  fi
done

# ---- lossless concat + one full-audio mux ----
LIST="$CHUNK_DIR/list.txt"
: > "$LIST"
i=0
while [ -f "$CHUNK_DIR/$(printf 'chunk_%03d.mp4' "$i")" ]; do
  echo "file '$CHUNK_DIR/$(printf 'chunk_%03d.mp4' "$i")'" >> "$LIST"
  i=$((i + 1))
done
if [ "$i" -eq 0 ]; then echo "ERROR: no chunks produced"; exit 1; fi
echo "== concat $i chunks (lossless) =="
CONCAT="$WORK/concat_silent.mp4"
ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$CONCAT" >/dev/null 2>&1
echo "== mux full audio =="
ffmpeg -y -i "$CONCAT" -i "$AUD" -c:v copy -c:a aac -shortest "$OUT" >/dev/null 2>&1
rm -f "$CONCAT" "$LIST"

FINAL_DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT")"
echo "done: $OUT  (audio ${DUR}s / final ${FINAL_DUR}s / elapsed $((SECONDS / 60)) min)"
echo ""
echo "QA tips:"
echo "  - check chunk joins (every ~${CHUNK_SEC}s) for pose continuity"
echo "  - happy with it? free disk with:  rm -rf $CHUNK_DIR"
echo "  - re-render everything:           RERUN=1 plus the same command"