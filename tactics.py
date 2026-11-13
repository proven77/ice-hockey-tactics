from math import ceil, dist

import cv2
import numpy as np

from utils import get_foot_position


class JerseyColors:
    def __init__(self, colors_bgr):
        palette = np.array([colors_bgr[1], colors_bgr[2]], dtype=np.uint8)
        self.palette = cv2.cvtColor(palette[None], cv2.COLOR_BGR2LAB)[0].astype(float)
        self.votes = {}

    def assign(self, frame, players):
        assignments = {}
        height, width = frame.shape[:2]
        for track_id, player in players.items():
            x1, y1, x2, y2 = player["bbox"]
            w, h = x2 - x1, y2 - y1
            left, right = np.clip([x1 + 0.25 * w, x2 - 0.25 * w], 0, width).astype(int)
            top, bottom = np.clip([y1 + 0.22 * h, y1 + 0.55 * h], 0, height).astype(int)
            patch = frame[top:bottom:2, left:right:2]
            if patch.size:
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
                median = np.median(lab.reshape(-1, 3), axis=0)
                a, b = np.linalg.norm((self.palette - median) * [0.65, 1, 1], axis=1)
                vote = np.clip((b - a) / 35, -1, 1)
                self.votes[track_id] = 0.7 * self.votes.get(track_id, 0) + 0.3 * vote
            score = self.votes.get(track_id, 0)
            assignments[track_id] = 1 if score >= 0.2 else 2 if score <= -0.2 else 0
        return assignments


def project_frame(homography, players, puck_bbox, size_m):
    if homography is None:
        return {}, None
    ids = list(players)
    anchors = [get_foot_position(players[track_id]["bbox"]) for track_id in ids]
    if puck_bbox is not None:
        x1, y1, x2, y2 = puck_bbox
        anchors.append(((x1 + x2) / 2, (y1 + y2) / 2))
    points = homography.transform_points(np.asarray(anchors).reshape(-1, 2))
    valid = ((points >= 0) & (points <= size_m)).all(axis=1)
    positions = {track_id: tuple(map(float, point)) for track_id, point, ok in zip(ids, points, valid) if ok}
    puck = tuple(map(float, points[-1])) if puck_bbox is not None and valid[-1] else None
    return positions, puck


class MatchStats:
    def __init__(self, fps):
        self.fps = fps
        self.possession_frames = [0, 0, 0]
        self.distances = {}
        self.distance_segments = {}
        self.previous_positions = {}
        self.candidate = None
        self.candidate_frames = 0
        self.confirm_frames = max(2, ceil(0.12 * fps))

    def update(self, positions, puck, assignments):
        for track_id, position in positions.items():
            previous = self.previous_positions.get(track_id)
            if previous is None:
                continue
            movement = dist(previous, position)
            # Reject jumps over 15 m/s. Missing frames break the path.
            if movement * self.fps <= 15:
                self.distances[track_id] = self.distances.get(track_id, 0) + movement
                self.distance_segments[track_id] = self.distance_segments.get(track_id, 0) + 1
        self.previous_positions = positions

        candidate = None
        if puck is not None:
            nearest = sorted((dist(position, puck), track_id) for track_id, position in positions.items())
            if nearest and nearest[0][0] <= 2:
                if len(nearest) == 1 or nearest[1][0] - nearest[0][0] >= 0.5:
                    track_id = nearest[0][1]
                    team = assignments.get(track_id, 0)
                    if team:
                        candidate = (track_id, team)
        if candidate is None or candidate != self.candidate:
            self.candidate_frames = 0
        self.candidate = candidate
        self.candidate_frames += 1
        if candidate is not None and self.candidate_frames >= self.confirm_frames:
            holder, team = candidate
        else:
            holder, team = None, 0
        self.possession_frames[team] += 1
        return holder, team

    def possession_pct(self):
        total = sum(self.possession_frames)
        return [100 * count / total if total else 0.0 for count in self.possession_frames]
