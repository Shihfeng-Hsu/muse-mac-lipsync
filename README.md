# muse-mac-lipsync

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Model](https://img.shields.io/badge/model-MuseTalk%201.5%20%28MLX%29-8A2BE2)](https://huggingface.co/mlx-community/MuseTalk-1.5-fp16)

> **English TL;DR** — Turn one person's source video plus any audio file into a lip-synced talking video, entirely offline on an Apple Silicon Mac — free, scriptable, batchable. Built on the MLX port of MuseTalk 1.5, it keeps per-avatar landmark caches (detect once, reuse forever), ships a Gradio production UI with a batch queue, and renders long audio in resumable 30-second chunks with seamless ping-pong looping. Measured on a base M2 (24 GB): 3.9 faces/s bench, ~2.6 fps rendering — a 7 min 22 s episode takes ~86 min of pure compute, zero GPU cost. Pairs with [voxclone-mac](https://github.com/Shihfeng-Hsu/voxclone-mac) for cloned voiceovers.

在 **Apple Silicon Mac** 上,把「一位人物的影片 + 任意音檔」變成嘴型同步的講話影片 — 全程本機、免費、可批次。基於 [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) 的 MLX 移植版,自帶多 avatar 快取、圖形化介面與批次佇列。

在 M2(24GB)上實測:**7 分 22 秒的整片一次跑完**,純渲染 86 分鐘(2.6 fps,含散熱休息約 2.5 小時),零 GPU 費用。

## 目錄

- [環境需求](#環境需求)
- [安裝](#安裝)
- [快速開始(GUI)](#快速開始gui)
- [快速開始(CLI)](#快速開始cli)
- [調速參數](#調速參數)
- [目錄結構](#目錄結構)
- [成本與速度(M2 24GB 實測)](#成本與速度m2-24gb-實測)
- [常見狀況](#常見狀況)
- [備份](#備份)
- [負責任使用](#負責任使用)
- [授權與致謝](#授權與致謝)

## 特色

- **多 avatar 快取**:地標偵測每個 avatar 只做一次(CPU,慢),之後換任何音檔都不用重跑;換音檔 = 幾乎只花渲染時間
- **Gradio 產片台**:產片 / 新增 avatar / 批次三個分頁,關掉瀏覽器渲染照跑
- **批次佇列**:音檔丟進 `inbox/`,一條指令整批渲染,成品落 `outbox/`,失敗自動留原地重跑續傳
- **分塊渲染**:長片自動切成 30 秒段,可中斷(Ctrl-C)、可續傳,最後無損接合
- **跨集隔離**:分段與 whisper 特徵快取按音檔內容 md5 分目錄,換下一集不會吃到上一集的嘴型
- **無縫循環**:基底影片過短時,正播+倒播(乒乓)素材配合幀循環,切點乾淨不破圖

> 這台機器的定位是**離線產片器**(M2 實測 2.6 fps,做不了 30 fps 即時主播)。訓練/微調請用 CUDA 設備。

## 環境需求

- Apple Silicon Mac(M 系列晶片;在 M2 基礎版 8 核 GPU 上驗證,24GB 統一記憶體)
- macOS、`git`、`ffmpeg`(`brew install ffmpeg`)、Python ≥ 3.10(`brew install python@3.12`)
- 磁碟空間約 4GB(引擎 + 權重)

## 安裝

```bash
git clone https://github.com/Shihfeng-Hsu/muse-mac-lipsync.git
cd muse-mac-lipsync
bash setup_mlx_musetalk.sh
```

setup 腳本是**冪等**的(重跑只補缺的),它會:

1. 建立 `.venv/`(mlx + CPU torch)
2. clone 推理引擎 [xocialize/musetalk-mlx](https://github.com/xocialize/musetalk-mlx) 與上游 [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)(只取融合與偵測程式碼)
3. 下載權重(~2.5GB):MuseTalk 1.5 MLX fp16、DWPose 地標偵測、face-parse-bisent 融合
4. 跑一次 benchmark 驗機(會印出這台晶片的 faces/s)

## 快速開始(GUI)

```bash
.venv/bin/python muse_app.py
```

瀏覽器自動開 `http://127.0.0.1:7860`:

- **產片**:選 avatar、拖音檔(mp3/wav/m4a 都行)、開始渲染;關瀏覽器照跑
- **新增 avatar**:上傳來源影片,做一次性地標偵測(慢,但這個 avatar 之後配任何音檔都不用重跑)
- **批次**:多支音檔按檔名順序排隊渲染,輸出到 `outbox/`

## 快速開始(CLI)

```bash
# 註冊 avatar(一次性地標偵測,之後永久重用)
muse.sh avatar-add myanchor ~/來源影片.mp4

# 單集產片
caffeinate -is muse.sh render myanchor ~/某集.mp3 ~/out_ep01.mp4

# 整批:把音檔丟進 inbox/,然後
caffeinate -is muse.sh batch myanchor
# 完成的音檔移到 inbox/done/,成品在 outbox/;失敗的留原地,重跑即續傳

# 其他
muse.sh list                              # 看 avatar 與快取狀態
muse.sh avatar-adopt 名字 ~/影片.mp4        # 遷移既有快取成 avatar(秒級)
muse.sh set-source 名字 ~/影片.mp4          # 來源影片搬家後重新指定
```

`caffeinate` 防 Mac 睡著;**跑的時候插電、別闔上螢幕蓋**,風扇口不要擋住。

簡單模式(不建 avatar、單支直接跑):`./run_lipsync.sh <source.mp4> <audio> <out.mp4>`

## 調速參數

加在指令最前面(環境變數形式):

```bash
COOLDOWN_SEC=120 muse.sh render ...    # 段間休息縮到 2 分鐘(預設 300)
CHUNK_SEC=60   muse.sh render ...      # 每段音檔長度改 60 秒(預設 30)
BATCH_GAP=60   muse.sh batch ...       # 集間休息改 1 分鐘(預設 120)
RERUN=1        muse.sh render ...      # 忽略已有分段,整片重渲
```

## 目錄結構

```
muse_app.py                     Gradio 介面入口(最常用)
muse.sh / muse_tools.py         CLI + avatar 管理
run_lipsync.sh                  單支直跑(不建 avatar 的簡單模式)
run_lipsync_chunks.sh           分段渲染引擎入口(UI 和 muse.sh 都呼叫它)
build_video_chunk.py            分段渲染核心(一般不用直接碰)
bench_mlx.py                    MLX 驗機測速
setup_mlx_musetalk.sh           一鍵安裝(冪等)
musetalk-mlx/                   引擎程式 + 權重(setup 建立,git 忽略)
.venv/                          python 環境(setup 建立,git 忽略)
avatars/<名字>/                  每 avatar 的快取 ← 備份這個就夠
inbox/ (+done/)                 批次佇列(第一次 batch 自動建立)
outbox/                         成品
work/                           執行期中繼(可整包刪,自動重建)
```

## 成本與速度(M2 24GB 實測)

實測時間 2026-09:

| 項目 | 耗時 | 頻率 |
|---|---|---|
| MLX bench | 3.9 faces/s | 驗機一次 |
| 渲染 | 2.6 fps ≈ 每 1 分鐘音檔 12 分鐘 | 每集 |
| 地標偵測 | 約每 30 幀 1 分鐘(CPU) | 每 avatar 一次 |
| 音檔特徵編碼 | 約 1 分鐘 | 每集一次 |

## 常見狀況

| 狀況 | 處理 |
|---|---|
| render 顯示 landmarks 在重跑 | 來源影片路徑或大小變了;`muse.sh set-source` 指定新路徑 |
| 嘴部邊緣融合太硬 | 引擎加參數 `--parsing-mode full`(預設 jaw) |
| 臉部裁切太緊 | 引擎加參數 `--extra-margin 20` |
| 7860 埠被佔用 | 關掉舊的 muse_app.py 程序,或改 port 重啟 |
| 環境壞了(例如升級 macOS 後) | 重跑 `bash setup_mlx_musetalk.sh`(冪等,只補缺的) |
| 想清磁碟 | 確認成品 OK:`rm -rf work/*` |

## 備份

只有 `avatars/` 是「貴且難重建」的資產(地標快取 + 抽好的幀),定期用 Time Machine 或外接碟備份它;權重、venv、引擎都可以重抓重建。

## 負責任使用

這套工具會讓畫面中的人物「開口說話」。請只用**你自己**或**取得當事人同意**的影像與聲音,不要用於冒充、詐騙或其他侵害他人的用途。

## 授權與致謝

- 本 repo 的腳本與文件以 **MIT** 授權釋出(見 `LICENSE`)
- 引擎與權重**不隨本 repo 散布**,由 setup 腳本另行下載,各自遵循上游授權:
  - [xocialize/musetalk-mlx](https://github.com/xocialize/musetalk-mlx)(MLX 移植)+ [mlx-community/MuseTalk-1.5-fp16](https://huggingface.co/mlx-community/MuseTalk-1.5-fp16)(權重)
  - [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)(上游模型與融合程式碼)
  - [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose)(地標偵測)
- 配音可搭配 [voxclone-mac](https://github.com/Shihfeng-Hsu/voxclone-mac)(VoxCPM2 本機聲音克隆),輸出音檔直接丟進本管線