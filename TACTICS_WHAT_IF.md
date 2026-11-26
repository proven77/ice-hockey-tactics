# 冰球 What-If 單幀與戰術板動畫

2026-11-27 已新增姿態重繪、三種畫面比較、新落點非同步生成及人工接點核對。最新操作見 [實驗 5.4 使用說明](TACTICS_REVIEW_REDRAW.md)，以下保留前四個實驗的建立與驗收紀錄。

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

程式會先確認本機 ComfyUI API 已就緒，再建立暫存輸出資料夾；全部結果完成後才移至 `--output` 指定的位置。若 ComfyUI 尚未啟動，會提示啟動方式且不建立輸出資料夾。若生成期間服務中斷，錯誤會列出 ComfyUI prompt ID，並清除未完成的暫存輸出；修復服務後可用原指令重跑。已存在的正式輸出資料夾不會被覆寫，需改用新路徑或先移動舊資料夾。

同一關鍵幀要比較多個落點時，重複傳入 `--target X_M Y_M`。程式先檢查每個落點，再送出一次 ComfyUI 修補。所有結果共用同一張 `repaired-background.png` 與同一份原球員裁切，各落點輸出在 `target-01`、`target-02` 等子目錄。省略 `--target` 時沿用案例 JSON 的落點及單一目錄輸出。

```powershell
& $cvPy what_if.py what_if_case.json --output output-videos/whatif-defense-step1 `
    --target 44.65 15.82 --target 40.8 14.8 --target 45 13
& $cvPy -m unittest -v test_tactics test_what_if
```

## 處理流程

1. `what_if_case.json` 記錄這一幀的冰場單應性矩陣、冰刀錨點、球員與球桿輪廓、陰影和鄰近球員保護範圍。先用反單應性矩陣把目標冰面座標轉成畫面上的冰刀落點，再依目標位置的局部透視比例縮放原人物裁切。這個矩陣與輪廓只適用於此畫面，切換關鍵幀時要重新校正與標註。
2. 以原位置附近未遮罩的冰面估計局部色調，再用本機 ComfyUI 的 Z-Image Turbo 遮罩流程修補。取生成的局部紋理，將低頻色調對齊周圍冰面，並在遮罩內羽化接縫。此修補假設原位置附近為沒有明顯線條的冰面。
3. 將原球員與球桿以透明遮罩合成到指定落點，加入近似的冰刀接觸陰影。程式逐像素比較原圖和結果，若合成遮罩外有變化就會回報錯誤。遮罩範圍涵蓋原位置修補、新位置人物和接觸陰影。
4. 戰術板以同一對冰面座標繪製 3 秒軌跡。其他球員取自案例關鍵幀並保持固定，D1 以平滑插值移動。動畫適合展示移動方向和終點。實際滑行速度與攻守反應需要另行建模。

## 輸出

| 檔案 | 檢查項目 |
| --- | --- |
| `what-if.png`、`original.png` | 原始解析度的結果圖與原圖 |
| `comparison.png` | 左右並排預覽 |
| `repaired-background.png`、`repair-mask.png` | 一次修補產生的共用背景與原位置修補遮罩 |
| `tactical-path.mp4` | 30 fps、90 幀的戰術板動畫 |
| `edit-mask.png` | 白色為允許修改的像素 |
| `report.json` | 移動前後座標、合成錨點、遮罩外變更像素數與 ComfyUI 工作 ID |
| `comfy-workflow-api.json`、`comfy-repair.png` | ComfyUI API 工作流和生成裁切，供檢查修補過程 |

`anchor_roundtrip_error_m` 量測的是畫面錨點經同一組矩陣回投後的數值誤差。真實冰面定位精度仍取決於原先的冰場校正。

## 本機驗收

最終結果位於 `output-videos/whatif-final`。冰刀落點約為 `(1141.41, 519.74)` px。放大原位置檢查，已移除球員與球桿輪廓，修補區仍有輕微冰面色調變化。原圖與結果圖均為 1920 × 1080，逐像素比對後，41,452 個變更像素皆在 `edit-mask.png` 白色範圍內。戰術板 MP4 完整解碼得到 90 幀。

同一台機器的一次冷啟動測得約 301 秒，權重已載入後的下一次修補約 14 秒。兩筆時間來自個別執行結果。

### 防守實驗 1

第 150 幀的追蹤 ID 4 使用一次本機 ComfyUI 修補，三張原尺寸結果均重用同一份修補背景。本次冷啟動到背景就緒為 325.0 秒，結果在 `output-videos/whatif-defense-step1`。

| 指定冰面座標（公尺） | 位移（公尺） | 合成冰刀錨點（px） | 遮罩外變更像素 |
| --- | ---: | --- | ---: |
| `(44.65, 15.82)` | 2.23 | `(1141.41, 519.74)` | 0 |
| `(40.80, 14.80)` | 1.91 | `(750.53, 532.70)` | 0 |
| `(45.00, 13.00)` | 3.82 | `(1048.60, 452.90)` | 0 |

逐張解碼 PNG 並核對指定落點、保護區及遮罩外像素。三支戰術板 MP4 均解碼出 90 幀。相鄰球員重疊、人物裁切及冰刀錨點出畫三種案例各回報原因，且於提交修補前被拒絕。局部放大可見人物與球桿輪廓已清除，原冰面區域仍有輕微色差。定位檢查使用同一組冰場矩陣，尚未量測與真實冰場的定位偏差。

修補就緒後，以三個落點循環測試，暖身 3 次後採樣 60 次。落點檢查加局部合成 p50 為 4.08 ms、p95 為 5.69 ms。計入 1080p PNG 編碼後 p95 為 57.57 ms。這組數據採高解析度計時器，量測後端處理，介面、磁碟寫入及影片匯出耗時另計。

### 防守實驗 2

在 repo 根目錄以 Windows `cv` 啟動復盤介面。若已完成實驗 1，可直接使用該次修補結果，本機瀏覽器開啟 `http://127.0.0.1:8765`。

```powershell
$cvPy = Join-Path $env:USERPROFILE 'anaconda3\envs\cv\python.exe'
& $cvPy defense_review.py --repair-run output-videos/whatif-defense-step1
```

若需要新修補，先依前段指令啟動 ComfyUI 並等它就緒，再執行 `& $cvPy defense_review.py`。選取 `D1` 會提交一次修補。同一頁切換參考幀或重新拖曳時可重用已完成的背景。若本機服務尚未啟動而出現失敗提示，啟動後按「重試修補」。

- 第 150 幀可選取或拖曳 `D1`。也可用方向鍵移動，每次 0.2 公尺，按 Enter 後套用。第 149 幀供原始畫面對照，切換時會清除目前結果。
- 放開後可切換原圖與 What-If，關閉畫面上的 `D1` 疊圖，或還原原位。下載項目包含乾淨的 `what-if.png`、帶 `D1` 與定位環的 `what-if-annotated.png`、以及包含來源幀、追蹤 ID 和冰刀落點的 `report.json`。
- 瀏覽器在 Windows Chrome 無頭模式、1440 × 900 下，以三個落點循環做 30 次拖曳。時鐘從放開按鍵算到結果繪入畫布後的繪製週期。快取就緒時 p50 為 144.6 ms，p95 為 166.0 ms，最慢 179.2 ms。此結果是同一台機器的本機測量。
- 額外延遲前一次請求以驗證快速連拖與切幀。確認晚回傳的畫面未覆蓋最新落點或第 149 幀。另驗證相鄰球員重疊提示、還原、標記切換和三項下載。匯出的兩張 PNG 為 1920 × 1080，乾淨圖編輯遮罩外變更為 0 像素，分析圖的額外像素變更限於圖標與冰刀標記區。

### 防守實驗 3

在 repo 根目錄用 Windows `cv` 執行下列指令。第 592 幀的冰球、門柱、冰刀與其他球員站位已存於 `what_if_shot_case.json`。它使用手動複核的同幀資料，可直接從示範影片重現。

```powershell
$cvPy = Join-Path $env:USERPROFILE 'anaconda3\envs\cv\python.exe'
& $cvPy defense_review.py --case what_if_shot_case.json --measure-only
& $cvPy what_if.py what_if_shot_case.json --output output-videos/whatif-defense-step3-guided
& $cvPy defense_review.py --case what_if_shot_case.json --repair-run output-videos/whatif-defense-step3-guided --port 8770
```

第一個指令直接輸出射門幾何與建議落點 JSON，無須啟動 ComfyUI。第二個指令需要本機 ComfyUI，並建立新的結果資料夾。已有這份快取時可直接執行第三個指令，在瀏覽器開啟 `http://127.0.0.1:8770`。也可省略 `--repair-run`，由復盤介面在選取球員時啟動原位修補。

1. 拖曳 `D1` 時，戰術板和原鏡頭以青色顯示球門角度、琥珀色顯示 D1 覆蓋的角度。放開後更新 What-If 圖。點「套用規則站位」才會套用建議。
2. 用「原圖」與「What-If」看移動前後的影像與射門區域。第 591 幀供畫面對照。第 150 幀案例缺少冰球量測證據，會顯示「待確認」，仍可拖曳。
3. 下載乾淨圖、帶 D1 與射門角度的分析圖，或內含 `defense` 欄位的 JSON。報告記錄像素標註、冰面座標、遮擋半徑、校正依據與前後結果。

滑鼠和方向鍵會把 D1 留在戰術板的圓角冰場內。放開時若落點在鏡頭外，或 D1 的身體、冰刀與保護物有明顯重疊，會顯示原因並回到上一個有效站位。第 592 幀的保護區沿球員、球桿、球門和冰球標註。合成時讓原畫面的物件遮住 D1 輪廓的輕微交疊和接觸陰影，並逐像素確認保護區保持原樣。JSON 記錄裁掉的球員輪廓與陰影像素數。

初次實驗匯出的落點為 `(54.11, 12.56) m`，移動 2.73 m。以 0.45 m 半徑近似 D1 的水平遮擋，球門可見開口約 11.3°，覆蓋由 0.0% 變為 76.2%（約 8.6°）。更新保護區與遮擋判定後，規則可選擇更近的 `(54.51, 12.60) m`，移動 2.57 m、覆蓋 71.2%。兩個落點均遮擋中央方向。這是單一防守者在固定畫面的估計，會受到半徑和投影誤差影響。推估射門結果或出手前可達性需要球高、其他球員動作和時間資料。

第 592 幀的原位同時遮住擋板與冰面標線。修補時對齊第 628 幀，將擋板和標線裁在修補遮罩內，再用本機 ComfyUI 修補其餘冰面。實拍結果經逐像素比對，編輯遮罩外變更為 0。Chrome 驗證了拖曳中的量測、切幀後捨棄過期回應、套用建議與匯出，以及第 150 幀缺冰球時的拖曳。校正和規則詳見 [封堵射門線實作計畫](TACTICS_SHOT_LANE_PLAN.md)。

教練拖曳、放開後的影像更新與防守球員標記見 [防守復盤 What-If 計畫](TACTICS_DEFENSE_PLAN.md)。

### 防守實驗 4

依實驗 3 的指令開啟復盤頁。在第 592 幀拖曳 D1 或點「套用規則站位」，等 What-If 圖就緒後按「戰術動畫 MP4」。下載檔名為 `tactical-path.mp4`，規格為 1000 × 550、30 fps、90 幀，共 3 秒。若要一併保留圖與數值，按「分析圖 PNG」及「防守報告 JSON」。

- 影片沿用當前結果的畫面幀、D1、追蹤 ID、原位與落點。起點與終點都會停留片刻。第 592 幀另標示冰球、門柱、球門可見角度與隨 D1 移動更新的覆蓋。其他球員固定在原位。移動的時間與路徑供復盤示意，途中位置不另驗證碰撞或是否能在比賽中到達。
- 第 150 幀缺少射門量測資料時仍可匯出，影片會顯示「Shot coverage: needs review」。切幀或修改落點時，介面會取消前一筆尚未完成的下載。
- 命令列 `what_if.py` 與復盤頁共用影片產生程式。JSON 內的 `animation` 記錄起訖的戰術板像素、解析度、幀數與幀率。影片用 OpenCV 的 `mp4v` 編碼。若瀏覽器無法直接預覽，可先下載再用支援 MPEG-4 Part 2 的播放器開啟。

實測兩段瀏覽器匯出的影片均完整解碼 90 幀，並檢查 D1 第一幀、中段與最後一幀位置。第 592 幀 D1 的影片終點 `(854, 269) px` 對應圖與 JSON 的冰面座標 `(54.51, 12.60) m`，位移 2.57 m，覆蓋由 0.0% 變為 71.2%。同一落點下，命令列渲染函式與瀏覽器匯出的 MP4 逐位元組一致。
