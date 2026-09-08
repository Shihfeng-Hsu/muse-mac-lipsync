#!/usr/bin/env python
# muse_app.py - local Gradio UI over the muse / chunk-render pipeline.
#
# Launch (from the project dir):
#   .venv/bin/python muse_app.py
#   -> opens http://127.0.0.1:7860 in the browser
#
# Tabs:
#   產片     pick avatar + upload one audio -> chunked render with cooldowns
#   新增 avatar  upload a source video -> one-time landmark extraction
#   批次     upload several audio files -> sequential renders into outbox/
#
# The UI only shells out to the already-validated scripts; renders keep
# running even if the browser tab is closed (and are resumable via CLI).
# Run only ONE job at a time (a busy flag guards the buttons).
#
# All code and comments pure ASCII; UI strings contain no full-width
# punctuation by design.

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr

PROJECT = Path(__file__).resolve().parent
AVATARS = PROJECT / "avatars"
OUTBOX = PROJECT / "outbox"
CHUNKS_SH = PROJECT / "run_lipsync_chunks.sh"
EXTRACT = PROJECT / "musetalk-mlx" / "scripts" / "extract_landmarks.py"
TOOLS = PROJECT / "muse_tools.py"
PY = PROJECT / ".venv" / "bin" / "python"

AUDIO_TYPES = [".mp3", ".wav", ".m4a", ".flac", ".aac"]
VIDEO_TYPES = [".mp4", ".mov", ".m4v"]

BUSY = False


def as_paths(x):
    # gradio may hand us str paths or FileData objects; normalize to paths
    if x is None:
        return []
    items = x if isinstance(x, list) else [x]
    out = []
    for i in items:
        if isinstance(i, str):
            out.append(i)
        elif hasattr(i, "name"):
            out.append(i.name)
        elif isinstance(i, dict) and "name" in i:
            out.append(i["name"])
    return [p for p in out if p]


def list_avatars():
    AVATARS.mkdir(parents=True, exist_ok=True)
    return sorted(p.parent.name for p in AVATARS.glob("*/coords.pkl"))


def stream_run(cmd, env=None):
    # generator: yields (log_text, done). The child process is NOT killed if
    # the UI disconnects - renders are resumable, so let it finish.
    e = os.environ.copy()
    if env:
        e.update(env)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(PROJECT), env=e)
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line.rstrip())
            yield "\n".join(lines[-220:]), False
        code = proc.wait()
    finally:
        if proc.poll() is None:
            pass  # detached by design; chunk renders are resumable
    tail = "\n".join(lines[-220:]) + "\n== EXIT CODE: %d ==" % code
    yield tail, True


def do_render(avatar, audio, chunk_sec, cooldown_sec):
    global BUSY
    if BUSY:
        yield "another job is running; wait for it or restart the app", None
        return
    paths = as_paths(audio)
    if not avatar:
        yield "choose an avatar first (tab: 新增 avatar, if the list is empty)", None
        return
    if not paths:
        yield "upload an audio file first", None
        return
    src = json.load(open(AVATARS / avatar / "profile.json")).get("source_video", "")
    if not src or not Path(src).exists():
        yield "source video missing for '%s'; re-register it via CLI:\n" \
              "  muse.sh set-source %s <video.mp4>" % (avatar, avatar), None
        return
    stem = Path(paths[0]).stem
    OUTBOX.mkdir(parents=True, exist_ok=True)
    out = OUTBOX / (stem + ".mp4")
    if out.exists():
        out = OUTBOX / (stem + "_" + time.strftime("%H%M%S") + ".mp4")
    env = {
        "AVATAR_DIR": str(AVATARS / avatar),
        "CHUNK_SEC": str(int(chunk_sec)),
        "COOLDOWN_SEC": str(int(cooldown_sec)),
    }
    BUSY = True
    try:
        log = "== output: %s ==" % out
        for text, done in stream_run(
                ["bash", str(CHUNKS_SH), src, paths[0], str(out)], env):
            yield log + "\n" + text, None
        ok = "EXIT CODE: 0" in text
        yield (log + "\n" + text + "\n" + ("DONE" if ok else "FAILED (see log)"),
               str(out) if ok else None)
    finally:
        BUSY = False


def do_avatar_add(name, video):
    global BUSY
    if BUSY:
        yield "another job is running; wait for it", gr.update()
        return
    paths = as_paths(video)
    if not name or not name.strip():
        yield "give the avatar a name (letters, digits, _ -)", gr.update()
        return
    name = name.strip()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        yield "name may only use letters, digits, _ or -", gr.update()
        return
    if not paths:
        yield "upload a source video first", gr.update()
        return
    avdir = AVATARS / name
    if (avdir / "coords.pkl").exists():
        yield "avatar '%s' already exists; pick another name" % name, gr.update()
        return
    (avdir / "frames").mkdir(parents=True, exist_ok=True)
    src = avdir / "source_video.mp4"
    shutil.move(paths[0], str(src))
    BUSY = True
    try:
        for text, done in stream_run(
                [str(PY), str(EXTRACT), str(src), str(avdir / "frames"),
                 str(avdir / "coords.pkl")]):
            yield text, gr.update()
        if "EXIT CODE: 0" not in text:
            yield text + "\nFAILED", gr.update()
            return
        sig = "%s:%d" % (str(src), os.path.getsize(str(src)))
        (avdir / "coords.meta").write_text(sig + "\n")
        subprocess.run([str(PY), str(TOOLS), "profile", name, str(src),
                        str(avdir / "coords.pkl"), str(avdir / "profile.json")],
                       check=True, capture_output=True)
        yield text + "\nAVATAR READY: %s" % name, gr.update(choices=list_avatars())
    finally:
        BUSY = False


def do_batch(avatar, files, gap):
    global BUSY
    if BUSY:
        yield "another job is running; wait for it", None
        return
    paths = as_paths(files)
    if not avatar or not paths:
        yield "choose an avatar and upload audio files", None
        return
    src = json.load(open(AVATARS / avatar / "profile.json")).get("source_video", "")
    if not src or not Path(src).exists():
        yield "source video missing for '%s'" % avatar, None
        return
    OUTBOX.mkdir(parents=True, exist_ok=True)
    BUSY = True
    try:
        outputs, report = [], []
        for n, p in enumerate(paths, 1):
            stem = Path(p).stem
            out = OUTBOX / (stem + ".mp4")
            if out.exists():
                report.append("%d/%d  %s -> SKIP (exists)" % (n, len(paths), stem))
                outputs.append(str(out))
                yield "\n".join(report), outputs
                continue
            report.append("%d/%d  %s -> rendering ..." % (n, len(paths), stem))
            yield "\n".join(report), outputs
            env = {"AVATAR_DIR": str(AVATARS / avatar),
                   "CHUNK_SEC": "30", "COOLDOWN_SEC": "300"}
            ok = False
            for text, done in stream_run(
                    ["bash", str(CHUNKS_SH), src, p, str(out)], env):
                yield "\n".join(report + ["--- %s ---" % stem] + text.split("\n")[-8:]), outputs
            ok = "EXIT CODE: 0" in text
            if ok:
                outputs.append(str(out))
                report[-1] = "%d/%d  %s -> OK" % (n, len(paths), stem)
            else:
                report[-1] = "%d/%d  %s -> FAILED" % (n, len(paths), stem)
            yield "\n".join(report), outputs
            if n < len(paths) and gap > 0:
                report.append("rest %ds ..." % int(gap))
                yield "\n".join(report), outputs
                time.sleep(float(gap))
        yield "\n".join(report + ["BATCH DONE"]), outputs
    finally:
        BUSY = False


with gr.Blocks(title="Muse 產片台") as demo:
    gr.Markdown("# Muse 產片台 - MuseTalk MLX 本地版\n"
                "引擎 = 已驗證的分段渲染管線;一次只跑一個工作;關掉瀏覽器渲染會繼續(可用 CLI 續傳)")

    with gr.Tabs():
        with gr.Tab("產片"):
            with gr.Row():
                dd_render = gr.Dropdown(choices=list_avatars(), label="avatar",
                                        interactive=True)
                audio_in = gr.File(label="音檔 (mp3/wav/m4a)", file_types=AUDIO_TYPES)
            with gr.Row():
                chunk_s = gr.Slider(10, 120, value=30, step=5,
                                    label="每段音檔長度 (秒)")
                cool_s = gr.Slider(0, 900, value=300, step=30,
                                   label="段間休息 (秒)")
            render_btn = gr.Button("開始渲染", variant="primary")
            render_log = gr.Textbox(label="執行紀錄", lines=20)
            render_out = gr.Video(label="成品")

        with gr.Tab("新增 avatar"):
            gr.Markdown("上傳來源影片後會做一次性地標偵測 (CPU, 慢);"
                        "之後這個 avatar 配任何音檔都不用重跑"
                        "舊專案的 avatar-1 快取請用 CLI 遷移:"
                        " muse.sh avatar-adopt <名字> <影片>")
            with gr.Row():
                name_tb = gr.Textbox(label="avatar 名字 (英數-_)")
                video_in = gr.File(label="來源影片", file_types=VIDEO_TYPES)
            add_btn = gr.Button("註冊 avatar (會跑地標偵測)", variant="primary")
            add_log = gr.Textbox(label="執行紀錄", lines=20)

        with gr.Tab("批次"):
            gr.Markdown("多支音檔按檔名順序排隊渲染, 輸出到 outbox/; 集間休息可調")
            with gr.Row():
                dd_batch = gr.Dropdown(choices=list_avatars(), label="avatar",
                                       interactive=True)
                files_in = gr.File(label="音檔 (可多選)", file_types=AUDIO_TYPES,
                                   file_count="multiple")
            gap_s = gr.Slider(0, 600, value=60, step=30, label="集間休息 (秒)")
            batch_btn = gr.Button("開始批次", variant="primary")
            batch_log = gr.Textbox(label="佇列狀態", lines=16)
            batch_out = gr.Files(label="成品下載")

    refresh = [gr.update(choices=list_avatars())] * 2
    demo.load(lambda: refresh, outputs=[dd_render, dd_batch])
    render_btn.click(do_render, [dd_render, audio_in, chunk_s, cool_s],
                     [render_log, render_out])
    add_btn.click(do_avatar_add, [name_tb, video_in], [add_log, dd_render])
    batch_btn.click(do_batch, [dd_batch, files_in, gap_s], [batch_log, batch_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)