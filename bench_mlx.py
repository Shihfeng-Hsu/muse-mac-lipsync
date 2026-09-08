#!/usr/bin/env python
# bench_mlx.py - measure MuseTalk-MLX generation speed on this Mac.
#
# Usage:
#   python bench_mlx.py <weights_dir>     e.g. musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16
#
# Reports: model load time, faces/sec at batch 8 (25 fps = realtime threshold),
# peak memory, and an audio-encoder path check. Pure ASCII by design.

import resource
import sys
import time

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)


def peak_rss_gb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024.0 ** 3)  # macOS reports bytes
    return raw / (1024.0 ** 2)      # Linux reports KB


def main():
    if len(sys.argv) < 2:
        print("usage: python bench_mlx.py <weights_dir>")
        return 1
    weights = sys.argv[1]

    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

    print("loading pipeline from:", weights)
    t0 = time.time()
    pipe = MuseTalkPipeline.from_pretrained_mlx(weights)
    print("load time: %.1fs" % (time.time() - t0))

    # one dummy 256x256 BGR face crop -> 8-channel latent
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    lat = pipe.get_latents_for_unet(crop)

    n = 64
    lat_stack = mx.concatenate([lat.astype(mx.float16)] * n, axis=0)
    chunks = mx.zeros((n, 50, 384)).astype(mx.float16)

    print("warm-up (8 faces) ...")
    _ = pipe.run_batched(lat_stack[:8], chunks[:8], batch_size=8)

    print("measuring %d faces at batch 8 ..." % n)
    t0 = time.time()
    recon = pipe.run_batched(lat_stack, chunks, batch_size=8)
    dt = time.time() - t0
    fps = n / dt
    print("faces/sec: %.1f" % fps)
    if fps >= 25.0:
        print("realtime check: OK (>= 25 fps)")
    else:
        print("realtime check: below 25 fps -- still fine for offline rendering")

    # audio feature path check: 1 s of 440 Hz sine -> whisper chunks
    try:
        import soundfile as sf
        t = np.arange(16000) / 16000.0
        wav = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write("bench_sine.wav", wav, 16000)
        enc = pipe.encode_audio_from_wav("bench_sine.wav", fps=25)
        print("audio path ok, chunks shape:", tuple(enc.shape))
    except Exception as exc:  # noqa: BLE001
        print("audio path check failed:", exc)

    print("peak RSS: %.2f GB" % peak_rss_gb())
    return 0


if __name__ == "__main__":
    sys.exit(main())