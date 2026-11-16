"""Draw a short, deterministic replay of one confirmed What-If placement."""

import cv2
import numpy as np


FPS = 30
FRAME_COUNT = 90
ORIGIN = (50, 84)
BOARD_SIZE = (900, 382)
CANVAS_SIZE = (1000, 550)
BACKGROUND = (29, 23, 18)
LIGHT = (241, 241, 238)
MUTED = (187, 190, 177)
AMBER = (106, 205, 255)
CYAN = (180, 171, 58)
TEAM_COLORS = {0: (146, 151, 150), 1: (89, 132, 42), 2: (245, 244, 237)}


def board_pixel(position, rink_size):
    return tuple(np.rint(np.asarray(position) / rink_size * (np.asarray(BOARD_SIZE) - 1) + ORIGIN).astype(int))


def animation_info(rink_size, source, target):
    return {
        "fps": FPS,
        "frames": FRAME_COUNT,
        "seconds": FRAME_COUNT / FPS,
        "codec": "mp4v",
        "size_px": list(CANVAS_SIZE),
        "start_marker_px": [int(value) for value in board_pixel(source, rink_size)],
        "end_marker_px": [int(value) for value in board_pixel(target, rink_size)],
        "motion": "illustrative_smoothstep",
    }


def text(canvas, value, point, size, color=LIGHT, thickness=1):
    cv2.putText(canvas, value, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def draw_shot_geometry(canvas, puck, goal, blocked_spans):
    shade = canvas.copy()
    cv2.fillConvexPoly(shade, np.array([puck, *goal]), CYAN, cv2.LINE_AA)
    for span in blocked_spans:
        cv2.fillConvexPoly(shade, np.array([puck, *span]), AMBER, cv2.LINE_AA)
    cv2.addWeighted(shade, .38, canvas, .62, 0, dst=canvas)
    for post in goal:
        cv2.line(canvas, puck, post, CYAN, 1, cv2.LINE_AA)
    for span in blocked_spans:
        for endpoint in span:
            cv2.line(canvas, puck, endpoint, (36, 133, 208), 1, cv2.LINE_AA)
    cv2.line(canvas, *goal, CYAN, 3, cv2.LINE_AA)
    for post in goal:
        cv2.circle(canvas, post, 3, CYAN, -1, cv2.LINE_AA)
    cv2.circle(canvas, puck, 5, LIGHT, -1, cv2.LINE_AA)
    cv2.circle(canvas, puck, 3, BACKGROUND, -1, cv2.LINE_AA)


def write_animation(path, rink_path, rink_size, tracks, report, coverage_at=None):
    rink = cv2.imread(str(rink_path), cv2.IMREAD_COLOR)
    if rink is None:
        raise OSError("Could not read the tactical rink image")
    rink = cv2.resize(rink, BOARD_SIZE, interpolation=cv2.INTER_AREA)
    source = np.asarray(report["source_m"])
    target = np.asarray(report["target_m"])
    label = report["label"]
    start_px = tuple(report["animation"]["start_marker_px"])
    target_px = tuple(report["animation"]["end_marker_px"])
    defense = report["defense"]
    measured = defense["status"] == "ready"
    if measured:
        puck = board_pixel(defense["puck_m"], rink_size)
        goal = [board_pixel(point, rink_size) for point in defense["goal_posts_m"]]
    spots = [(board_pixel(track["position"], rink_size), TEAM_COLORS.get(track["team"], TEAM_COLORS[0]))
             for track in tracks]

    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), BACKGROUND, dtype=np.uint8)
    left, top = ORIGIN
    width, height = BOARD_SIZE
    canvas[top:top + height, left:left + width] = rink
    text(canvas, "What-If  /  Defensive repositioning", (left, 45), .91, thickness=2)
    text(canvas, f"FRAME {report['source_frame']}   /   TRACK {report['track_id']}", (725, 44), .62, MUTED)
    text(canvas, f"{label}  ({source[0]:.2f}, {source[1]:.2f}) -> ({target[0]:.2f}, {target[1]:.2f}) m"
                 f"   |   {np.linalg.norm(target - source):.2f} m",
         (left, 493), .63)
    text(canvas, "Illustrative path", (783, 523), .55, MUTED)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, CANVAS_SIZE)
    try:
        if not writer.isOpened():
            raise OSError("Could not open the MP4 writer")
        for index in range(FRAME_COUNT):
            board = canvas.copy()
            # Hold the starting and ending positions so both are easy to inspect.
            fraction = np.clip((index - 12) / 55, 0, 1)
            progress = fraction ** 2 * (3 - 2 * fraction)
            position = source + progress * (target - source)
            if measured:
                if fraction == 0:
                    current = defense["before"]
                elif fraction == 1:
                    current = defense["after"]
                else:
                    current = coverage_at(position)
                spans = [[board_pixel(point, rink_size) for point in span]
                         for span in current["blocked_goal_spans_m"]]
                draw_shot_geometry(board, puck, goal, spans)
                line = (f"Shot angle blocked  {defense['before']['blocked_pct']:.1f}% -> {defense['after']['blocked_pct']:.1f}%"
                        f"   |   Now {current['blocked_pct']:.1f}% ({current['blocked_angle_deg']:.1f} deg)")
                text(board, line, (left, 523), .62, AMBER)
            else:
                text(board, "Shot coverage: needs review", (left, 523), .62, MUTED)
            for point, color in spots:
                cv2.circle(board, point, 10, BACKGROUND, -1, cv2.LINE_AA)
                cv2.circle(board, point, 8, color, -1, cv2.LINE_AA)
            if start_px != target_px:
                cv2.arrowedLine(board, start_px, target_px, (32, 129, 206), 2, cv2.LINE_AA, tipLength=.20)
                cv2.circle(board, start_px, 9, (69, 90, 101), 2, cv2.LINE_AA)
                cv2.circle(board, target_px, 15, (32, 129, 206), 2, cv2.LINE_AA)
            if fraction == 0:
                x, y = start_px
            elif fraction == 1:
                x, y = target_px
            else:
                x, y = board_pixel(position, rink_size)
            cv2.circle(board, (x, y), 13, BACKGROUND, -1, cv2.LINE_AA)
            cv2.circle(board, (x, y), 10, AMBER, -1, cv2.LINE_AA)
            tag_x = x - 16
            tag_y = y - 37 if y - top >= 38 else y + 16
            cv2.rectangle(board, (tag_x, tag_y), (tag_x + 33, tag_y + 20), BACKGROUND, -1)
            text(board, label, (tag_x + 6, tag_y + 15), .48, AMBER)
            writer.write(board)
    finally:
        writer.release()
