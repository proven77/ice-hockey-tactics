# 冰球 What-If 單幀與戰術板動畫

固定示例使用 `input-videos/snapshot-demo.mp4` 的第 150 幀（從 0 起算）。白隊的追蹤 ID 4 從約 `(42.42, 15.82)` 公尺移至手動指定的 `(44.65, 15.82)` 公尺，位移 2.23 公尺。這個案例用來檢查重定位、修補和動畫的底層流程，協防落點需要由使用者依情境決定。 [示範片來源](https://github.com/danielrzhang/SnapShot#before-snapshot-processing)

## 本機環境

在 Windows 使用 `cv` 執行影像處理，ComfyUI 使用其安裝目錄內的 `.venv`。本案例以 ComfyUI 0.37.0 和 RTX 4060 8 GB 執行。模型放在 ComfyUI 的 `models` 子目錄，`diffusion_models` 放 `z_image_turbo_bf16.safetensors`，`text_encoders` 放 `qwen_3_4b.safetensors`，`vae` 放 `ae.safetensors`。 [ComfyUI 專案](https://github.com/Comfy-Org/ComfyUI)

從 repo 根目錄啟動本機 ComfyUI，保持這個 PowerShell 視窗開啟。以下是這台 Windows 機器已驗證的低顯存設定。工作目錄在 `all-runs`，資料庫使用記憶體模式。

```powershell
$comfy = 'E:\ComfyUI\ComfyUI\ComfyUI'
$whatIfBase = Join-Path $PWD 'all-runs\whatif-comfy'
& "$comfy\.venv\Scripts\python.exe" -u "$comfy\main.py" --base-directory $whatIfBase --models-directory "$comfy\models" --database-url 'sqlite:///:memory:' --listen 127.0.0.1 --port 8191 --disable-all-custom-nodes --disable-api-nodes --disable-auto-launch --lowvram --disable-dynamic-vram --disable-async-offload --disable-pinned-memory
```

保留 `--database-url 'sqlite:///:memory:'`。首次測試以 `--base-directory` 指定工作目錄時漏傳 `--database-url`。ComfyUI 因新工作目錄尚無資料庫，將原安裝目錄的 `user/comfyui.db` 改名為 `.bak`，並複製到 `all-runs/whatif-comfy/user/comfyui.db`。原檔已復原，原檔、備份與工作目錄副本的 SHA-256 完全一致，原檔通過 SQLite `quick_check`。記憶體模式會略過搬移，匯出的 PNG、MP4 與 JSON 仍寫入磁碟。 [ComfyUI 資料庫初始化程式](https://github.com/Comfy-Org/ComfyUI/blob/master/app/database/db.py)

另一個 PowerShell 視窗也進入 repo 根目錄。第一次設定 `cv` 可依 [第一版使用說明](TACTICS_V1.md) 安裝相依套件。若尚無 `output-videos/snapshot-cv-rink640/tracks.csv`，先執行下列 `analyze.py` 指令產生它。每次執行 `what_if.py` 時使用新的輸出目錄。

```powershell
$cvPy = Join-Path $env:USERPROFILE 'anaconda3\envs\cv\python.exe'
$env:YOLO_CONFIG_DIR = Join-Path $PWD 'all-runs\ultralytics-config'
if (-not (Test-Path 'output-videos/snapshot-cv-rink640/tracks.csv')) {
    & $cvPy analyze.py input-videos/snapshot-demo.mp4 --team-a green --team-b white --device 0 --half --rink-imgsz 640 --max-frames 360 --output output-videos/snapshot-cv-rink640
}
& $cvPy what_if.py what_if_case.json --output output-videos/whatif-local-run
```

## 處理流程

1. `what_if_case.json` 記錄這一幀的冰場單應性矩陣、冰刀錨點、球員與球桿輪廓、陰影和鄰近球員保護範圍。先用反單應性矩陣把目標冰面座標轉成畫面上的冰刀落點，再依目標位置的局部透視比例縮放原人物裁切。這個矩陣與輪廓只適用於此畫面，切換關鍵幀時要重新校正與標註。
2. 以原位置附近未遮罩的冰面估計局部色調，再用本機 ComfyUI 的 Z-Image Turbo 遮罩流程修補。取生成的局部紋理，將低頻色調對齊周圍冰面，並在遮罩內羽化接縫。此修補假設原位置附近為沒有明顯線條的冰面。
3. 將原球員與球桿以透明遮罩合成到指定落點，加入近似的冰刀接觸陰影。程式逐像素比較原圖和結果，若合成遮罩外有變化就會回報錯誤。遮罩範圍涵蓋原位置修補、新位置人物和接觸陰影。
4. 戰術板以同一對冰面座標繪製 3 秒軌跡。其他球員取自第 150 幀並保持固定，移動速度由平滑插值決定。動畫適合展示移動方向和終點，未模擬比賽中的滑行速度或攻守反應。

## 輸出

| 檔案 | 檢查項目 |
| --- | --- |
| `what-if.png`、`original.png` | 原始解析度的結果圖與原圖 |
| `comparison.png` | 左右並排預覽 |
| `tactical-path.mp4` | 30 fps、90 幀的戰術板動畫 |
| `edit-mask.png` | 白色為允許修改的像素 |
| `report.json` | 移動前後座標、合成錨點、遮罩外變更像素數與 ComfyUI 工作 ID |
| `comfy-workflow-api.json`、`comfy-repair.png` | ComfyUI API 工作流和生成裁切，供檢查修補過程 |

`anchor_roundtrip_error_m` 量測的是畫面錨點經同一組矩陣回投後的數值誤差。真實冰面定位精度仍取決於原先的冰場校正。

## 本機驗收

最終結果位於 `output-videos/whatif-final`。冰刀落點約為 `(1141.41, 519.74)` px。放大原位置檢查，已移除球員與球桿輪廓，修補區仍有輕微冰面色調變化。原圖與結果圖均為 1920 × 1080，逐像素比對後，41,452 個變更像素皆在 `edit-mask.png` 白色範圍內。戰術板 MP4 完整解碼得到 90 幀。

同一台機器的一次冷啟動測得約 301 秒，權重已載入後的下一次修補約 14 秒。兩筆時間來自個別執行結果。

下一步的關鍵幀、封堵規則與驗收條件見 [封堵射門線實作計畫](TACTICS_SHOT_LANE_PLAN.md)。

教練拖曳、放開後的影像更新與防守球員標記見 [防守復盤 What-If 計畫](TACTICS_DEFENSE_PLAN.md)。
