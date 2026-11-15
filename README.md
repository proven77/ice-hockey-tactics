# Ice Hockey Tactics

冰上曲棍球戰術分析第一版的安裝、操作與驗收紀錄見 [TACTICS_V1.md](TACTICS_V1.md)。使用 `analyze.py` 輸出戰術板、控球率和逐軌跡滑行距離。

本機 ComfyUI 修補與 What-If 戰術板動畫的重現步驟見 [TACTICS_WHAT_IF.md](TACTICS_WHAT_IF.md)。

[防守復盤介面](TACTICS_WHAT_IF.md#防守實驗-3)支援拖曳 `D1`、比較影像與水平射門角度覆蓋，並匯出分析結果。

[防守復盤計畫](TACTICS_DEFENSE_PLAN.md)涵蓋拖曳、影像更新和球員標記。[封堵射門線計畫](TACTICS_SHOT_LANE_PLAN.md)記錄關鍵幀與量測驗收方式。

本專案以 [SnapShot](https://github.com/danielrzhang/SnapShot/tree/ccfc904c51643b95fce7e3b0c0305df4a331e2ee) 為基礎，保留 [GPL-3.0 授權](LICENSE)。以下為上游專案介紹。

## SnapShot

SnapShot is a Python application that analyzes hockey game video clips using YOLO object detection, custom machine learning models, and Roboflow datasets. It automatically annotates key gameplay elements, processing videos frame-by-frame with computer vision techniques and a deep convolutional neural network. The tool outputs annotated videos and statistical summaries, ideal for clips of 30 to 120 seconds to ensure fast processing and accurate results. SnapShot is designed for coaches, analysts, and hockey enthusiasts to break down gameplay dynamics efficiently.

Watch a sample of SnapShot in action:  
### Before SnapShot Processing  
[Original NHL](https://github.com/user-attachments/assets/7fd4311b-6692-468c-8a49-b977d13f9174)  

### After SnapShot Processing  
[Processed NHL](https://github.com/user-attachments/assets/9523a3b1-15fc-441a-8618-b74015f6869b)  

## Features
- **Player Annotation by Team**: Labels players from opposing teams with position markers and team-specific identifiers using advanced classification.  
- **Puck Position Tracking**: Tracks the puck’s location and visualizes its estimated path in each frame.  
- **Puck Possession Detection**: Identifies which player or team controls the puck based on proximity and movement patterns.  
- **Rink Keypoint Detection**: Detects rink keypoints (e.g., center ice, blue lines, goal lines) for perspective correction, zone identification, and spatial analytics.  
- **Pass and Interception Detection**: Tracks successful passes within each team and interceptions caused by opponent actions.  
- **Puck Possession Percentage**: Calculates each team’s puck possession time as a percentage of the clip’s duration.  
- **Custom Models**: Utilizes pre-trained Roboflow datasets and a custom YOLO model for accurate object detection.  

## YOLO Training Summary
The training for the puck, player, and rink models was conducted locally on the hardware specified below. 

⚠️ Note: A GPU is required for efficient training; using a CPU alone significantly increases processing time and may cause performance issues.  

### Training Device Specs
- **CPU**: Intel Core Ultra 7 155H  
- **GPU**: NVIDIA RTX 3000 Ada  

### Training Times
- **Player Recognition**: 18 hours (on a dataset of ~3,500 images, 100 epochs)  
- **Puck Recognition**: 56 hours (on a dataset of ~47,000 images, 500 epochs)  
- **Rink Keypoints**: 8 hours (on a dataset of ~100 images, 500 epochs)

### Datasets Used
- **Player (YOLOv5)**: [Hockey-3fz5i](https://universe.roboflow.com/francisco-workspace/hockey-3fz5i/dataset/7)  
- **Puck (YOLOv5)**: [Puck-Detection-4](https://universe.roboflow.com/rapid-q94xs/puck-detection-4/dataset/5)  
- **Rink (YOLOv8)**: [HockeyCV](https://universe.roboflow.com/hockeycv-hpnix/hockeycv)  

### Fine-Tuned Models
- **Player and Puck**: [HockeyAI](https://huggingface.co/spaces/SimulaMet-HOST/HockeyAI/tree/main)  
- **Rink**: [HockeyRink](https://huggingface.co/SimulaMet-HOST/HockeyRink/tree/main)  

## Requirements
- **Python**: 3.10 or higher (tested with 3.13)  
- **Dependencies**:  
  ```bash
  ultralytics
  pytorch
  roboflow
  opencv-python
  numpy
  ```

## Work In Progress
- **Improved Puck Tracking**: Enhance accuracy for high-speed puck movement, limited by camera frame rates, using advanced prediction algorithms.  
- **Tactical Overview**: Create a real-time overhead rink view using rink keypoints to visualize player positions, puck paths, and team strategies.  
- **Offside Detection**: Add rule-based logic to detect offside violations by tracking players entering the offensive zone ahead of the puck relative to blue lines.  
- **Icing Detection**: Detect icing by tracking puck shots from behind the center line crossing the opponent’s goal line without touches.  
- **Goal Detection and Shot Analytics**: Detect goals by monitoring puck proximity to the net and calculate shot speeds for analytics.  

## License
This project is licensed under the GNU General Public License. See [LICENSE](LICENSE) for details.
