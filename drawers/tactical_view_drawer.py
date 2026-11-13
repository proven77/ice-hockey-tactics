import cv2
import numpy as np
from utils import get_foot_position

class TacticalViewDrawer:
  def __init__(self, team_1_color, team_2_color):
    self.start_x = 1450
    self.start_y = 40
    self.team_1_color = team_1_color
    self.team_2_color = team_2_color
  
  def draw(self, video_frames, rink_image_path, width, height, tactical_rink_keypoints, tactical_player_positions, player_assignment = None, puck_acquisition = None):
    rink_image = cv2.imread(rink_image_path, cv2.IMREAD_UNCHANGED)
    rink_image = cv2.resize(rink_image, (width, height), interpolation=cv2.INTER_AREA)
    
    rink_bgr = rink_image[:, :, :3]
    rink_alpha = rink_image[:, :, 3].astype(float) / 255.0
    
    rink_alpha = np.stack([rink_alpha] * 3, axis=2) 
    
    output_video_frames = []
    for frame_idx, frame in enumerate(video_frames):
      frame = frame.copy()
      x1 = self.start_x
      y1 = self.start_y
      x2 = x1 + width
      y2 = y1 + height
      
      if (y2 <= frame.shape[0] and x2 <= frame.shape[1] and 
        rink_image.shape[0] == y2 - y1 and rink_image.shape[1] == x2 - x1):
        roi = frame[y1:y2, x1:x2].copy()
        blended = (rink_bgr * rink_alpha + roi * (1 - rink_alpha)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
        
      for key_point_index, keypoint in enumerate(tactical_rink_keypoints):
        x, y = keypoint
        x += self.start_x
        y += self.start_y
        # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        # cv2.putText(frame, str(key_point_index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame_positions = tactical_player_positions[frame_idx]
        frame_assignment = player_assignment[frame_idx]
        player_with_puck = puck_acquisition[frame_idx]
        
        for player_id, position in frame_positions.items():
          team_id = frame_assignment.get(player_id)
          color = self.team_1_color if team_id == 1 else self.team_2_color
          x, y = int(position[0] + self.start_x), int(position[1] + self.start_y)
          player_radius = 3
          cv2.circle(frame, (x, y), player_radius, color, -1)
          
          if player_id == player_with_puck:
            cv2.circle(frame, (x, y), player_radius + 3, (0, 0, 255), 2)
      
      output_video_frames.append(frame)
    
    return output_video_frames


class LiveTacticalView:
  def __init__(self, rink_path, size_m, colors):
    rink = cv2.imread(str(rink_path))
    if rink is None:
      raise ValueError(f"Could not read rink image: {rink_path}")
    self.rink = cv2.resize(rink, (424, 180), interpolation=cv2.INTER_AREA)
    self.size_m = size_m
    self.colors = colors

  def draw(self, frame, players, teams, positions, puck_bbox, puck, holder, stats, calibrated):
    width = 448
    height = max(360, frame.shape[0])
    panel = np.full((height, width, 3), (30, 25, 21), dtype=np.uint8)
    panel[64:244, 12:436] = self.rink

    def board_point(point):
      x, y = point
      return 12 + round(423 * x / self.size_m[0]), 64 + round(179 * y / self.size_m[1])

    def label(canvas, value, origin, color=(240, 240, 240), size=0.55):
      cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    label(panel, "ICE HOCKEY  /  TACTICAL VIEW", (12, 28), size=0.65)
    status = "Ice-plane projection" if calibrated else "No valid rink calibration"
    label(panel, status, (12, 51), (190, 205, 220))
    annotated = frame.copy()
    for track_id, player in players.items():
      team = teams[track_id]
      color = self.colors[team]
      x1, y1, x2, y2 = map(round, player["bbox"])
      cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
      distance = stats.distances.get(track_id)
      caption = f"#{track_id}" if distance is None else f"#{track_id}  {distance:.1f} m"
      origin = (x1, max(12, y1 - 4))
      cv2.putText(annotated, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 3, cv2.LINE_AA)
      label(annotated, caption, origin)
      if track_id in positions:
        x, y = board_point(positions[track_id])
        cv2.circle(panel, (x, y), 7, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 5, color, -1, cv2.LINE_AA)
        label(panel, str(track_id), (x + 7, y - 5), (25, 25, 25), 0.4)
        if track_id == holder:
          cv2.circle(panel, (x, y), 10, (40, 160, 240), 2, cv2.LINE_AA)

    if puck_bbox is not None:
      x1, y1, x2, y2 = puck_bbox
      center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
      cv2.circle(annotated, center, 8, (40, 230, 255), 2, cv2.LINE_AA)
    if puck is not None:
      cv2.circle(panel, board_point(puck), 4, (25, 25, 25), -1, cv2.LINE_AA)

    unknown, team_a, team_b = stats.possession_pct()
    label(panel, "ESTIMATED PUCK CONTROL  /  ALL FRAMES", (12, 274), size=0.5)
    label(panel, f"A {team_a:.1f}%    B {team_b:.1f}%    Unknown {unknown:.1f}%", (12, 305))
    label(panel, "Black dot: puck  /  Labels: skater track IDs", (12, 337), (190, 205, 220), 0.49)
    annotated = cv2.copyMakeBorder(annotated, 0, height - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return cv2.hconcat([annotated, panel])
