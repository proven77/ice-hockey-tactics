import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import supervision as sv
import webcolors

from drawers.tactical_view_drawer import LiveTacticalView
from rink_keypoint_detector import RinkKeypointDetector
from tactical_view_converter import TacticalViewConverter
from tactics import JerseyColors, MatchStats, project_frame
from trackers import PlayerTracker


ROOT = Path(__file__).resolve().parent
SKATER_NAMES = {"player", "players", "skater", "skaters", "goalie", "goalies", "goaltender", "goaltenders"}


def class_ids(names):
    normalized = {int(idx): name.lower().strip() for idx, name in names.items()}
    skaters = [idx for idx, name in normalized.items() if name in SKATER_NAMES]
    pucks = [idx for idx, name in normalized.items() if name == "puck"]
    if not skaters or not pucks:
        raise ValueError(f"Detector needs ice hockey skater and puck classes. Found {list(names.values())}")
    return skaters, pucks


def run(args, cap, fps):
    tracker = PlayerTracker(str(args.weights), frame_rate=round(fps))
    skater_ids, puck_ids = class_ids(tracker.model.names)
    rink_model = RinkKeypointDetector(str(args.rink_weights)).model
    rink_path = ROOT / "images" / "nhl-hockey-rink.png"
    converter = TacticalViewConverter(rink_path)
    colors = {
        0: (145, 145, 145),
        1: tuple(reversed(webcolors.name_to_rgb(args.team_a))),
        2: tuple(reversed(webcolors.name_to_rgb(args.team_b))),
    }
    if colors[1] == colors[2]:
        raise ValueError("Choose distinct jersey colors for both teams")
    assigner = JerseyColors(colors)
    stats = MatchStats(fps)
    drawer = LiveTacticalView(rink_path, converter.size_m, colors)
    predict = {"device": args.device, "imgsz": args.imgsz, "verbose": False, "half": args.half}
    rink_predict = {**predict, "imgsz": args.rink_imgsz or args.imgsz}

    args.output.mkdir(parents=True, exist_ok=False)
    writer = None
    frames = calibrated_frames = puck_frames = projected_puck_frames = 0
    seen_tracks = set()
    started = perf_counter()

    try:
        with (args.output / "frames.csv").open("w", newline="", encoding="utf-8") as frame_file, \
             (args.output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_file:
            frame_rows = csv.writer(frame_file)
            track_rows = csv.writer(track_file)
            frame_rows.writerow(["frame", "time_s", "calibration_inliers", "puck_confidence", "puck_x_m", "puck_y_m",
                                 "holder_track_id", "possession_team", "team_a_pct", "team_b_pct", "unknown_pct"])
            track_rows.writerow(["frame", "time_s", "track_id", "team", "confidence", "x1_px", "y1_px", "x2_px", "y2_px",
                                 "x_m", "y_m", "observed_distance_m"])

            while args.max_frames is None or frames < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id = args.start_frame + frames
                timestamp = round(frame_id / fps, 6)
                result = tracker.model.predict(frame, conf=0.1, **predict)[0]
                detections = sv.Detections.from_ultralytics(result)
                players = tracker.track_detections(detections[np.isin(detections.class_id, skater_ids)])
                seen_tracks.update(players)
                pucks = detections[np.isin(detections.class_id, puck_ids) & (detections.confidence >= args.puck_conf)]
                puck_bbox = puck_score = None
                if len(pucks):
                    best = int(np.argmax(pucks.confidence))
                    puck_bbox = pucks.xyxy[best].tolist()
                    puck_score = float(pucks.confidence[best])

                keypoints = rink_model.predict(frame, conf=0.25, **rink_predict)[0].keypoints
                homography = converter.fit_frame(keypoints)
                positions, puck = project_frame(homography, players, puck_bbox, converter.size_m)
                teams = assigner.assign(frame, players)
                holder, team = stats.update(positions, puck, teams)
                unknown_pct, team_a_pct, team_b_pct = stats.possession_pct()
                calibrated_frames += homography is not None
                puck_frames += puck_bbox is not None
                projected_puck_frames += puck is not None

                frame_rows.writerow([frame_id, timestamp, homography.inliers if homography else 0, puck_score,
                                     *(puck or (None, None)), holder, team, team_a_pct, team_b_pct, unknown_pct])
                for track_id, player in players.items():
                    position = positions.get(track_id, (None, None))
                    track_rows.writerow([frame_id, timestamp, track_id, teams[track_id], player["confidence"],
                                         *player["bbox"], *position, stats.distances.get(track_id)])

                rendered = drawer.draw(frame, players, teams, positions, puck_bbox, puck, holder, stats, homography is not None)
                if writer is None:
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(str(args.output / "tactics.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise ValueError("Could not open the MP4 video writer")
                writer.write(rendered)
                frames += 1
                if args.preview:
                    cv2.imshow("Ice Hockey Tactics", rendered)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
    finally:
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    if frames == 0:
        raise ValueError("No readable video frames at the requested start frame")
    elapsed = perf_counter() - started
    summary = {
        "source": str(args.source), "weights": str(args.weights), "rink_weights": str(args.rink_weights),
        "team_colors": {"A": args.team_a, "B": args.team_b}, "device": args.device, "imgsz": args.imgsz,
        "rink_imgsz": rink_predict["imgsz"], "half": args.half, "puck_confidence_threshold": args.puck_conf,
        "start_frame": args.start_frame, "frames": frames, "source_fps": fps, "processing_fps": frames / elapsed,
        "calibrated_frames": calibrated_frames, "puck_frames": puck_frames, "projected_puck_frames": projected_puck_frames,
        "possession_frames": dict(zip(("unknown", "A", "B"), stats.possession_frames)),
        "possession_pct_all_frames": dict(zip(("unknown", "A", "B"), stats.possession_pct())),
        "tracks": {track_id: {"observed_distance_m": stats.distances.get(track_id),
                              "valid_distance_segments": stats.distance_segments.get(track_id, 0)}
                   for track_id in sorted(seen_tracks)},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {frames} frames at {summary['processing_fps']:.1f} FPS -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Track hockey players and render a rink-space tactical view.")
    parser.add_argument("source", type=Path, help="Video of one continuous camera shot")
    parser.add_argument("--team-a", required=True, help="CSS jersey color, e.g. red")
    parser.add_argument("--team-b", required=True, help="CSS jersey color, e.g. white")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "player-detector.pt",
                        help="Hockey YOLO detection weights with player and puck labels")
    parser.add_argument("--rink-weights", type=Path, default=ROOT / "models" / "rink-detector.pt",
                        help="HockeyRink pose weights with the original 56-landmark ordering")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--rink-imgsz", type=int, help="Override the ice landmark model input size")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on a supported GPU")
    parser.add_argument("--puck-conf", type=float, default=0.25)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--preview", action="store_true", help="Show the annotated view. Press Q to finish early")
    args = parser.parse_args()

    if not all(path.is_file() for path in (args.source, args.weights, args.rink_weights)):
        parser.error("Source video and model weights must exist locally")
    if args.output.exists():
        parser.error("Output directory already exists. Choose a new path")
    if args.start_frame < 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("Frame start must be non-negative and frame limit must be positive")
    if args.imgsz <= 0 or (args.rink_imgsz is not None and args.rink_imgsz <= 0) or not 0.1 <= args.puck_conf <= 1:
        parser.error("Image size must be positive and puck confidence must be in [0.1, 1]")
    if args.device == "cpu" and args.half:
        parser.error("FP16 requires a supported GPU")

    cap = cv2.VideoCapture(str(args.source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not isfinite(fps) or fps <= 0:
            parser.error("Could not open a video with a valid frame rate")
        if args.start_frame and not cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame):
            parser.error("Could not seek to the requested start frame")
        run(args, cap, fps)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
