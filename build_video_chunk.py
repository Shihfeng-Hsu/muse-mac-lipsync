#!/usr/bin/env python
# build_video_chunk.py - chunked variant of musetalk-mlx build_video.py.
#
# Renders the frame range [start, start+num) of the full audio-driven output,
# while a separate chunk run stays frame-for-frame identical to a single
# full-length render:
#   - whisper features come from encoding the FULL audio once, then slicing
#     [start:start+num] (never from slicing the audio file itself)
#   - the source-frame cycle uses the same rule as upstream
#     (c = frames + reversed(frames)) but indexed with the global offset,
#     so head/body pose continues exactly where the previous chunk stopped
#
# Vendored from xocialize/musetalk-mlx scripts/build_video.py (MIT license).
# Changes: --start-frame / --num-frames, per-chunk temp frame dir, whisper
# feature cache (.npy), sidecar "<out>.frames" file with the rendered count.
#
# Usage:
#   python build_video_chunk.py <coords.pkl> <audio.wav> <out.mp4> \
#       [--variant fp16] [--start-frame 0] [--num-frames 0] \
#       [--extra-margin 10] [--parsing-mode jaw]
#   num-frames 0 means "render from start to the end of the audio".
#
# All code and comments are pure ASCII by design.

import argparse
import functools
import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "musetalk-mlx"
UPSTREAM = ROOT / "refs" / "MuseTalk"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "musetalk" / "utils"))

# legacy bisenet checkpoint needs weights_only=False (torch 2.6+ default flip)
import torch  # noqa: E402
torch.load = functools.partial(torch.load, weights_only=False)

ap = argparse.ArgumentParser()
ap.add_argument("coords"); ap.add_argument("audio"); ap.add_argument("out")
ap.add_argument("--variant", default="fp16")
ap.add_argument("--start-frame", type=int, default=0)
ap.add_argument("--num-frames", type=int, default=0)
ap.add_argument("--extra-margin", type=int, default=10)
ap.add_argument("--parsing-mode", default="jaw")
args = ap.parse_args()

mx.set_default_device(mx.gpu)
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline  # noqa: E402
from musetalk.utils.blending import get_image  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def whisper_chunks(pipe, audio_path, fps):
    # encode full audio once and cache as fp16 .npy keyed by the file's md5,
    # so different audio files never reuse each other's features
    cache_npy = os.path.join(os.path.dirname(audio_path),
                             "features_%s.npy" % _md5(audio_path))
    if os.path.exists(cache_npy):
        arr = np.load(cache_npy)
        print("audio features -> CACHED (%s)" % cache_npy, flush=True)
        return mx.array(arr)
    chunks = pipe.encode_audio_from_wav(audio_path, fps=int(round(fps)))
    np.save(cache_npy, np.array(chunks.astype(mx.float16)))
    print("audio features encoded + cached (%s)" % cache_npy, flush=True)
    return chunks.astype(mx.float16)


with open(args.coords, "rb") as f:
    meta = pickle.load(f)
coords, fps, frame_paths = meta["coords"], meta["fps"], meta["frames"]
frames = [cv2.imread(p) for p in frame_paths]
n_src = len(frames)
print("frames=%d fps=%.2f" % (n_src, fps), flush=True)

pipe = MuseTalkPipeline.from_pretrained_mlx(
    ROOT / "dist" / ("MuseTalk-1.5-MLX-" + args.variant))
# FaceParsing loads from hardcoded ./models/... paths -> run from upstream dir
os.chdir(UPSTREAM)
fp = FaceParsing()
os.chdir(SCRIPT_DIR)

chunks = whisper_chunks(pipe, os.path.realpath(args.audio), fps)
video_num = int(chunks.shape[0])
start = max(0, min(args.start_frame, video_num))
end = video_num if args.num_frames <= 0 else min(video_num, start + args.num_frames)
n = end - start
if n <= 0:
    print("NOTHING TO RENDER (start=%d end=%d total=%d)" % (start, end, video_num))
    sys.exit(0)
print("render frames [%d, %d) of %d" % (start, end, video_num), flush=True)

# per-frame crops + 8-ch latents (same as upstream build_video)
PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)
boxes, latents = [], []
for bbox, frame in zip(coords, frames):
    if bbox == PLACEHOLDER:
        boxes.append(None); latents.append(None); continue
    x1, y1, x2, y2 = bbox
    y2 = min(y2 + args.extra_margin, frame.shape[0])
    crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    boxes.append((x1, y1, x2, y2))
    latents.append(pipe.get_latents_for_unet(crop))

# safety: if any source frame lacked a face, reuse the nearest valid latent
if any(l is None for l in latents):
    ref = next(l for l in latents if l is not None)
    latents = [l if l is not None else ref for l in latents]

# same cycle rule as upstream (c = lst + lst[::-1]) with the global offset
lat_cycle = latents + latents[::-1]
box_cycle = boxes + boxes[::-1]
frame_cycle = frames + frames[::-1]
n_cycle = len(lat_cycle)

lat_stack = mx.concatenate(
    [lat_cycle[(start + i) % n_cycle] for i in range(n)], axis=0).astype(mx.float16)
chunk_slice = chunks[start:end].astype(mx.float16)
recon = pipe.run_batched(lat_stack, chunk_slice, batch_size=8)
print("generated %d faces" % len(recon), flush=True)

# blend back + write frames (chunk-local numbering for ffmpeg)
# temp frames live under <project>/work/ to keep the project root clean
work_dir = SCRIPT_DIR / "work"
tmp = work_dir / ("chunk_frames_%08d" % start)
tmp.mkdir(parents=True, exist_ok=True)
for p in tmp.glob("*.png"):
    p.unlink()
for i in range(n):
    g = (start + i) % n_cycle
    box = box_cycle[g]; ori = frame_cycle[g].copy()
    if box is None:
        cv2.imwrite(str(tmp / ("%08d.png" % i)), ori); continue
    x1, y1, x2, y2 = box
    res = cv2.resize(recon[i].astype(np.uint8), (x2 - x1, y2 - y1))
    combined = get_image(ori, res, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    cv2.imwrite(str(tmp / ("%08d.png" % i)), combined)
print("blended %d frames" % n, flush=True)

# encode this chunk as a silent mp4 (audio is muxed once, after concat)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", str(tmp / "%08d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open(str(out_path) + ".frames", "w") as f:
    f.write(str(n))
print("WROTE %s (frames=%d)" % (out_path, n), flush=True)