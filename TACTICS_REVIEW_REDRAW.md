# 實驗 5.4｜復盤介面的姿態重繪與位置核對

2026-11-27。Windows `cv`，本機 ComfyUI。球衣號碼與圖案依使用者指示不列入本輪採用條件。

## 已提供的操作

拖動戰術板上的 D1，放開後先顯示 OpenCV 即時合成。按「重繪 D1」才產生新的防守人物。可以切換原圖、即時預覽與姿態重繪，也可以在生成期間繼續拖曳或切到前一幀。新圖先標為「位置待確認」。

目前支援 `what_if_case.json` 第 150 幀與 `what_if_shot_case.json` 第 592 幀，來源姿態由 `player_redraw_cases.json` 人工標註。第一輪限制在原位 8 m 內，仍須通過冰場圓角、鏡頭範圍、縮放、身體與保護物交疊檢查。其他來源若沒有相符標註，介面不開放姿態重繪。

第 150 幀缺少冰球與門柱量測證據，只展示人物移位；射門量測保持待確認。第 592 幀可以比較單一 D1 的水平射門角度覆蓋。「套用規則站位」和「套用已評估落點」用途不同：前者依現有幾何規則選站位，後者重現 5.3 已核對的固定圖片，並不保證它是最佳防守選擇。

## 啟動

在 repo 根目錄開啟 PowerShell。影片、背景快取及模型的準備方法見 [What-If 使用說明](TACTICS_WHAT_IF.md) 與 [5.3 實驗紀錄](TACTICS_PLAYER_REDRAW_PLAN.md#53-姿態重繪實作與驗收)。現有本機已保存上述資料。大型影片、模型與實驗結果不包含在 GitHub 原始碼中。

```powershell
$cvPy = Join-Path $env:USERPROFILE 'anaconda3\envs\cv\python.exe'

# 第 150 幀
& $cvPy defense_review.py --case what_if_case.json --repair-run output-videos/whatif-defense-step1 --port 8765

# 另一個 PowerShell：第 592 幀，適合防守覆蓋比較
& $cvPy defense_review.py --case what_if_shot_case.json --repair-run output-videos/whatif-defense-step3-guided --port 8770
```

分別開啟 `http://127.0.0.1:8765` 與 `http://127.0.0.1:8770`。使用現有修補背景和 5.3 快取時，ComfyUI 不必啟動。要產生新落點，先啟動 `http://127.0.0.1:8191` 的 ComfyUI。沿用原有 Z-Image Turbo、Qwen、VAE 及 `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors`。需要的節點是 `ModelPatchLoader` 與 `ZImageFunControlnet`，無須安裝新模型或外部 API。

參數 `--comfy-port` 指定生成服務，`--redraw-run` 指定 5.3 候選目錄，`--redraw-cache` 指定新落點的持久快取。省略 `--repair-run` 會在首次選取 D1 時執行原位修補，首次載入模型可能較久。ComfyUI 的既有 SQLite 記憶體設定維持不變，本功能不新增資料庫或執行資料庫遷移。

本機最終驗收中，ComfyUI 重啟後首張新落點約 6 分 19 秒，暖機後約 26–27 秒，已完成快取的顯圖約 0.07–1.40 秒。API 顯示就緒仍可能尚未載入權重，復盤前可先完成一張新落點生成；生成期間即時預覽仍可用。詳細量測條件見 [冷啟動紀錄](TACTICS_PLAYER_REDRAW_PLAN.md#546-最終版本冷啟動與回歸驗收)。

## 操作順序

1. 按「套用已評估落點」，等待即時預覽。第 150 幀使用 `(44.65, 15.82) m`／seed 105，第 592 幀使用 `(54.0, 12.5) m`／seed 104。
2. 按「重繪 D1」。快取完全相符時直接載入；否則建立新輸入並排入 GPU 工作。介面顯示排隊、生成、完成或失敗狀態。新的要求只保留最新一筆尚未執行的工作；已送入 ComfyUI 的工作會繼續完成自己的快取。
3. 用「原圖」「即時預覽」「姿態重繪」比較。改位置後先回到新位置的預覽；前一張生成圖不會跟著人物標記移動。
4. 檢查生成圖時按「標記冰刀接點」。可點原尺寸影像所對應的畫面位置，或下面的放大區。放大區以指定接點為中心，取原圖 160 × 80 px，放大 3 倍；黃色十字是指定接點，綠色十字是人工標點。
5. 另外核對人物完整、承重冰刀與冰面接觸、球桿沒有不合理地穿過冰刀。確認後勾選「已確認人物與冰刀接地合理」。看不清時按「接點不清楚，保留待確認」。勾選合理性與標點是兩個獨立條件，球衣圖案不影響本輪判定。
6. 按所需的匯出鈕。乾淨 PNG、分析 PNG 和 JSON 對應目前選中的畫面；報告內有 `result_kind`。原圖模式的 `target_m` 是來源位置，另以 `requested_target_m` 保存戰術板的編輯目標。原圖模式不匯出移位動畫。

方向鍵每次微調 0.2 m，按 Enter 套用預覽。滑鼠移動超過 3 個螢幕像素才視為拖曳；很小的位移可用方向鍵操作。還原與切到前一幀都會清除目前編輯狀態。

## 位置判定代表什麼

像素誤差為人工標點與指定接點的歐氏距離。公尺誤差將兩點經**同幀** `image_to_ice` 映射後計算。兩項都須符合 `≤8 px` 與 `≤0.20 m`，還須人工確認接地合理。這是人工估計，預設標點不確定度約 ±2 px；程式沿此半徑取樣估計冰面誤差範圍，若不確定度可能跨過門檻，仍保留待複核。

5.3 的兩張固定圖沿用原本的 assistant 人工估計及其日期與方法，新落點不繼承舊核對結果。黃色 D1 定位環表示**要求的位置**，生成後的實際接點仍以人工標點紀錄為準。這套量測沒有校正外部真值，也沒有自動判斷滑冰動作或戰術是否正確。

新圖的像素檢查是確定性的：保護區、目標遮罩外相對已修補背景、原位修補與目標重繪總區域外相對原圖，變更像素都必須為 0，否則拒絕成圖。通過此檢查不代表人物品質或定位已通過。

## 底層流程與核心程式

| 程式／函式 | 運作與用途 |
| --- | --- |
| `defense_review.py`／`Review` | 載入來源幀、前一幀、人工遮罩與同幀校正；核對並共用原位修補背景。`/render` 提供快速合成，`/animation` 提供戰術板 MP4。 |
| `review_redraw.py`／`RedrawQueue.submit` | 驗證落點，對來源影像、完整案例與座標、背景、來源關節、工作流參數及程式版本建立 SHA-256 識別。精確命中才重用，不採鄰近落點或四捨五入座標。每個伺服器用一個 GPU worker，背景修補也共用此 worker；不同伺服器的請求由 ComfyUI 再排隊。 |
| `prepare_player_redraw.py`／`build_bundle` | 每個落點重新投影承重腳、轉換關節與球桿提示、建立目標活動遮罩，扣除保護區，裁切與縮放為 512 × 512。供 5.1 命令列與介面共用。 |
| `redraw_player.py`／`generate_candidate` | 命令列與介面共用的單張生成函式：上傳條件圖、遮罩、Pose，建立 `pose_workflow`，呼叫 `run_prompt`，回貼並驗證。第一輪固定 8 steps、CFG 1、denoise 0.65、strength 0.85，兩個來源分別 seed 105／104。 |
| `review_redraw.py`／`validate_result` | 重新建立輸入包，核對輸入 SHA-256、來源、完整落點、seed、模型識別與實際 API graph，再檢查成圖尺寸與逐像素保護。檔案不符會顯示錯誤，不會退回使用一張相似圖片。模型 SHA 欄位記錄既有受控實驗的權重識別，不是從 API 即時讀取的權重雜湊。更換權重時應分開使用快取目錄。 |
| `review_redraw.py`／`contact` | 驗證標點在畫面內，計算兩種誤差與門檻不確定性，保存該精確輸入對應的人工檢查。保留 5.3 原始報告。 |
| `defense_review/review.js` | 維護目前幀、目標、編輯版本、生成工作 ID 與顯示模式。送出後查詢狀態；顯圖前與解碼後均核對版本，改位置、切幀或還原時捨棄過期結果。PNG 分析圖與 MP4 的延後下載也有版本檢查。 |

冰面 Homography 僅反投影冰上的接點，人體關節由已標註的 2D 姿態轉換，不是透過 `H⁻¹` 恢復人體 3D。ComfyUI 重繪的是局部人物，最終由程式依允許遮罩回貼；原位修補背景可共用。

## 輸出與 API

新快取預設位於 `output-videos/defense-review-redraw/<輸入識別>/<工作 ID>/`，包含完整輸入包、`result/` 原始生成、工作流、ComfyUI 歷史、成圖與 `review-report.json`。只有完整驗證通過後才寫入該輸入的 `ready.json`，中斷或失敗的資料夾不會當作可用成圖。`contact-review.json` 保存該輸入的最新人工核對，重開伺服器後仍會載入。失敗工作不阻止同落點重試。

| 請求 | 回傳 |
| --- | --- |
| `POST /redraw`，`{"target":[x,y]}` | 工作 ID、精確輸入識別與 queued/running/ready 狀態。 |
| `GET /redraw/status?id=...` | queued、running、ready、failed 或 superseded；ready 附報告。 |
| `GET /redraw/image?id=...` | 已完成工作的原尺寸 PNG。 |
| `POST /redraw/cancel`，`{"id":"..."}` | 取消尚未執行的工作；已在生成的工作仍保存自己的結果。 |
| `POST /redraw/contact` | `id`、原圖像素 `point` 與 `plausible`；`point:null` 保留待確認。 |

JSON 包含來源幀、D1、原位與完整目標、公尺與像素定位、`result_kind`、工作 ID、seed、快取來源、工作流與輸入識別、像素驗證、生成耗時、介面顯圖耗時、人工接點與核對狀態。`provenance` 是該成圖的原始來源；`delivery` 區分本次直接從記憶體取用、磁碟快取或等候生成工作。`generation.request_seconds` 是原始 ComfyUI 請求耗時，快取的本次顯圖應看 `ui_display_seconds`。`queue_seconds` 僅量測此復盤伺服器 worker 的等待；ComfyUI 中其他工作的等待包含在請求耗時，目前未另拆分。背景分頁可能暫停繪圖回呼，完成狀態不依賴此回呼，另記錄 `ui_result_ready_seconds`。MP4 仍是 3 秒、90 幀的戰術板軌跡示意，沿用**戰術板要求的目標**，不是生成圖片中實測接點的動畫，也不是生成轉播影片。

服務未啟動或中斷時，可先使用預覽，修復 ComfyUI 後再按「重繪 D1」。若是來源、遮罩或快取工作流不符，先檢查案例與快取來源；保留原檔，使用新的 `--redraw-cache` 目錄排查。場外、鏡頭外與明顯交疊會先被拒絕，回到上一個合法預覽。

## 驗收與限制

驗收數據與檔案位置記錄於 [5.4 結果](TACTICS_PLAYER_REDRAW_PLAN.md#54-復盤介面接入計畫)。自動化操作中的合成滑鼠標點只驗證互動與報告儲存，**不作為人工定位正確的證據**。新落點仍保留待確認，球衣號碼細節沒有納入判定。

```powershell
& $cvPy -m unittest test_tactics test_what_if test_shot_lane test_review_redraw
# 需先開本機 headless Chrome 的 CDP 9234，以及兩個指定連接埠的復盤伺服器
node verify_review_redraw.mjs offline 8781 8782 output-videos/whatif-defense-step5-4/browser-offline-new
node verify_review_redraw.mjs live 8783 8784 output-videos/whatif-defense-step5-4/browser-live-new
& $cvPy verify_review_artifacts.py --browser output-videos/whatif-defense-step5-4/browser-live-new
```

`offline` 模式兩個伺服器須使用未啟動的 ComfyUI 連接埠（驗收為 `--comfy-port 1`），才能測失敗與重試；`live` 使用正常服務。兩種模式請依序執行，並用新的輸出目錄，避免瀏覽器下載設定與同名檔案互相影響。離線錯誤測例的目標不能已有生成快取；已有時應換獨立的 `--redraw-cache`，否則直接讀快取是預期行為。
