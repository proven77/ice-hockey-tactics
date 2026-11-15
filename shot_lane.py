"""Horizontal shot coverage for a single skater at a reviewed keyframe."""

from math import atan2, degrees, hypot

import numpy as np

from what_if import place_player, project


GOAL_WIDTH_M = 6 * 0.3048
GOAL_WIDTH_TOLERANCE_M = 0.15
RAY_COUNT = 361


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segment_distance(start, end, points):
    direction = end - start
    norm_sq = direction @ direction
    if norm_sq == 0:
        return np.linalg.norm(points - start, axis=-1)
    fraction = np.clip((points - start) @ direction / norm_sq, 0, 1)
    return np.linalg.norm(points - (start + fraction[..., None] * direction), axis=-1)


class ShotLane:
    def __init__(self, puck, posts, radius):
        self.puck = np.asarray(puck, dtype=float)
        self.posts = np.asarray(posts, dtype=float)
        self.radius = radius
        self.center = self.posts.mean(axis=0)

        start, end = self.posts - self.puck
        if abs(cross(start, end)) < 1e-8:
            raise ValueError("冰球與球門線共線")
        turn = atan2(cross(start, end), start @ end)
        self.opening_deg = abs(degrees(turn))
        # Divide by angle, not by equal distances along the goal line.
        angles = atan2(start[1], start[0]) + np.arange(RAY_COUNT + 1) * turn / RAY_COUNT
        midpoints = (angles[:-1] + angles[1:]) / 2
        self.edges = self.goal_intersections(angles)
        self.rays = self.goal_intersections(midpoints)
        self.ray_offsets = self.rays - self.puck
        self.ray_lengths_sq = np.sum(self.ray_offsets ** 2, axis=1)

    def goal_intersections(self, angles):
        direction = np.column_stack((np.cos(angles), np.sin(angles)))
        goal_line = self.posts[1] - self.posts[0]
        distance = cross(self.posts[0] - self.puck, goal_line) / cross(direction, goal_line)
        return self.puck + direction * distance[:, None]

    def coverage(self, position):
        point = np.asarray(position, dtype=float)
        relative = point - self.puck
        fraction = np.clip(self.ray_offsets @ relative / self.ray_lengths_sq, 0, 1)
        clearance = np.linalg.norm(relative - fraction[:, None] * self.ray_offsets, axis=1)
        blocked = clearance <= self.radius
        ratio = np.count_nonzero(blocked) / RAY_COUNT
        boundaries = np.flatnonzero(np.diff(np.r_[False, blocked, False]))
        return {
            "blocked_pct": 100 * ratio,
            "blocked_angle_deg": self.opening_deg * ratio,
            "central_blocked": bool(segment_distance(self.puck, self.center, point) <= self.radius),
            "blocked_goal_spans_m": self.edges[boundaries].reshape(-1, 2, 2).tolist(),
        }


class DefenseMeasurement:
    def __init__(self, case, player, rink_size, tracks):
        self.player = player
        self.rink_size = np.asarray(rink_size)
        self.lane = None
        self.others = np.array([track["position"] for track in tracks], dtype=float).reshape(-1, 2)
        evidence = case.get("shot_lane") or {}
        missing = [label for key, label in (("puck_px", "冰球"), ("goal_posts_px", "兩根門柱"))
                   if evidence.get(key) is None]
        if evidence.get("goal_side") not in ("left", "right"):
            missing.append("進攻方向")
        if evidence.get("calibration_reviewed") is not True:
            missing.append("射門區校正")
        source = {"source_video": case["video"], "source_frame": case["frame"], "track_id": case["track_id"]}
        self.metadata = {"status": "needs_review", **source}
        if missing:
            self.metadata["reason"] = "待確認：" + "、".join(missing)
            return
        try:
            puck_px = np.asarray(evidence["puck_px"], dtype=float)
            posts_px = np.asarray(evidence["goal_posts_px"], dtype=float)
            radius = float(evidence["blocking_radius_m"])
            if puck_px.shape != (2,) or posts_px.shape != (2, 2) or not 0 < radius <= 2:
                raise ValueError("冰球、門柱或遮擋半徑格式有誤")
            width, height = player.repair.shape[::-1]
            pixels = np.vstack([puck_px, posts_px])
            if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels >= [width, height]).any():
                raise ValueError("冰球或門柱超出來源畫面")
            puck, post_a, post_b = (project(player.image_to_ice, pixel) for pixel in pixels)
            points = np.vstack([puck, post_a, post_b])
            if (points < 0).any() or (points > self.rink_size).any():
                raise ValueError("冰球或門柱投影超出冰場")
            goal_width = float(np.linalg.norm(post_a - post_b))
            if abs(goal_width - GOAL_WIDTH_M) > GOAL_WIDTH_TOLERANCE_M:
                raise ValueError(f"門柱投影間距 {goal_width:.2f} m，超過案例校正容許差")
            right = evidence["goal_side"] == "right"
            expected_goal_x = self.rink_size[0] - 11 * 0.3048 if right else 11 * 0.3048
            if np.max(np.abs(points[1:, 0] - expected_goal_x)) > 0.35:
                raise ValueError("門柱與所選攻擊方向的球門線不符")
            if abs(points[1:, 1].mean() - self.rink_size[1] / 2) > 0.35:
                raise ValueError("門柱中心與冰場中軸不符")
            if (right and puck[0] >= min(post_a[0], post_b[0])) or (
                not right and puck[0] <= max(post_a[0], post_b[0])
            ):
                raise ValueError("冰球未在所選球門的冰場一側")
            self.lane = ShotLane(puck, points[1:], radius)
        except (KeyError, ValueError, TypeError) as exc:
            self.metadata["reason"] = f"待確認：{exc}"
            return

        self.before = self.lane.coverage(player.source_m)
        self.metadata = {
            "status": "ready", **source,
            "method": "horizontal_angle_uniform_sampling",
            "ray_count": RAY_COUNT,
            "puck_px": puck_px.tolist(), "goal_posts_px": posts_px.tolist(),
            "puck_m": puck.tolist(), "goal_posts_m": points[1:].tolist(),
            "goal_side": evidence["goal_side"], "goal_width_m": goal_width,
            "goal_width_tolerance_m": GOAL_WIDTH_TOLERANCE_M,
            "blocking_radius_m": radius, "opening_angle_deg": self.lane.opening_deg,
            "calibration": evidence.get("calibration"),
            "before": self.before, "suggestion": self.suggest(),
        }

    def measure(self, target):
        if self.lane is None:
            return self.metadata
        after = self.lane.coverage(target)
        return {
            **self.metadata, "target_m": np.asarray(target).tolist(), "after": after,
            "delta_percentage_points": after["blocked_pct"] - self.before["blocked_pct"],
        }

    def suggest(self):
        max_move, min_puck, min_net, min_player = 4.0, 1.5, 3.0, 1.0
        rules = {"max_move_m": max_move, "min_puck_m": min_puck, "min_net_center_m": min_net,
                 "min_player_clearance_m": min_player, "candidate_spacing_m": 0.05}
        if self.before["central_blocked"]:
            return {"status": "keep", "target_m": self.player.source_m.tolist(), "distance_m": 0.0, "rules": rules}
        direction = self.lane.center - self.lane.puck
        length = hypot(*direction)
        distances = np.arange(min_puck, length - min_net + 1e-9, rules["candidate_spacing_m"])
        candidates = self.lane.puck + distances[:, None] * direction / length
        valid = []
        for target in candidates:
            distance = float(np.linalg.norm(target - self.player.source_m))
            if distance > max_move:
                continue
            if np.any(segment_distance(self.player.source_m, target, self.others) < min_player):
                continue
            coverage = self.lane.coverage(target)
            if not coverage["central_blocked"] or coverage["blocked_pct"] <= self.before["blocked_pct"]:
                continue
            try:
                place_player(self.player, target, self.rink_size)
            except ValueError:
                continue
            valid.append((distance, -coverage["blocked_pct"], target))
        if not valid:
            return {"status": "unavailable", "reason": "條件內找不到能增加覆蓋的站位", "rules": rules}
        distance, _, target = min(valid, key=lambda item: item[:2])
        return {"status": "available", "target_m": target.tolist(), "distance_m": distance, "rules": rules}
