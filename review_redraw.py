"""Exact-target redraw cache, a single GPU queue, and manual contact measurements."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import requests

from prepare_player_redraw import build_bundle, json_file
from redraw_player import (bundle_digest, generate_candidate, generation_settings,
                           validate_pose_output, pose_workflow, CONTROL_FILE)
from refine_player import read_image
from what_if import ROOT, check_comfy_ready, project, filesystem_path


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class RedrawQueue:
    def __init__(self, review, cache_root=None, legacy_root=None):
        self.review = review
        self.root = filesystem_path(cache_root or ROOT / "output-videos/defense-review-redraw")
        self.legacy_root = filesystem_path(legacy_root or ROOT / "output-videos/whatif-defense-step5-3")
        self.lock = RLock()
        self.jobs = {}
        self.latest = None
        self.fixture = None
        self.preset = None
        spec = json.loads((ROOT / "player_redraw_cases.json").read_text(encoding="utf-8"))
        for name, fixture in spec["fixtures"].items():
            source = json.loads((ROOT / fixture["case"]).read_text(encoding="utf-8"))
            if source == review.case:
                self.fixture = fixture
                self.preset = next(entry for entry in spec["placements"] if entry["fixture"] == name)
                break
        self.seed = 105 if review.case["frame"] == 150 else 104
        self.metadata = {"supported": self.fixture is not None, "seed": self.seed,
                         "preset": self.preset, "settings": generation_settings(), "max_distance_m": 8}
        # A code or annotation change creates a new cache namespace, never a nearest-point lookup.
        self.version = fingerprint({name: sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "review_redraw.py", "prepare_player_redraw.py", "redraw_player.py", "refine_player.py",
            "what_if.py", "shot_lane.py", "player_redraw_cases.json")})

    def identity(self, target):
        placement, _ = self.review.describe_placement(target)
        if self.fixture is None:
            raise ValueError("此來源尚無來源姿態標註，暫不提供姿態重繪")
        if self.review.background is None:
            raise ValueError("請先完成即時預覽，修補背景尚未就緒")
        if np.linalg.norm(placement.target_m - self.review.player.source_m) > 8:
            raise ValueError("第一輪重繪限原位 8 m 內，請縮短位移")
        return {"source": self.review.case, "source_sha256": sha256(self.review.frame.tobytes()).hexdigest(),
                "background_sha256": sha256(self.review.background[0].tobytes()).hexdigest(),
                "fixture": self.fixture, "target_m": placement.target_m.tolist(), "seed": self.seed,
                "settings": generation_settings(), "code_version": self.version}

    def submit(self, target):
        identity = self.identity(target)
        key = fingerprint(identity)
        with self.lock:
            existing = next((j for j in self.jobs.values() if j["key"] == key and
                             j["state"] in ("queued", "running", "ready")), None)
            if existing:
                self.cancel_queued_except(existing["id"])
                self.latest = existing["id"]
                return self.snapshot(existing)
            self.cancel_queued_except(None)
            job_id = uuid4().hex
            directory = self.root / key / job_id
            directory.mkdir(parents=True)
            job = {"id": job_id, "key": key, "identity": identity, "target_m": identity["target_m"],
                   "state": "queued", "submitted": perf_counter(), "directory": directory}
            self.jobs[job_id] = job
            self.latest = job_id
            job["future"] = self.review.worker.submit(self.run, job)
            return self.snapshot(job)

    def cancel_queued_except(self, keep):
        for job in self.jobs.values():
            if job["id"] != keep and job["state"] == "queued" and job["future"].cancel():
                job["state"] = "superseded"

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job["state"] == "queued" and job["future"].cancel():
                job["state"] = "superseded"
            return {"state": job["state"] if job else "unknown"}

    def snapshot(self, job):
        with self.lock:
            return deepcopy({k: job[k] for k in ("id", "key", "target_m", "state", "error", "report") if k in job})

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("未知的重繪工作")
            return self.snapshot(self.jobs[job_id])

    def image(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["state"] != "ready":
                raise ValueError("重繪結果尚未就緒")
            return (job["result_dir"] / "redrawn.png").read_bytes()

    def validate_result(self, folder, inputs):
        record = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        source = json.loads((inputs / "report.json").read_text(encoding="utf-8"))
        expected = {**generation_settings(), "case": source["name"], "seed": self.seed,
                    "input_sha256": bundle_digest(inputs)}
        if record["identity"] != expected:
            raise ValueError("姿態快取的工作流、模型、seed 或輸入不符")
        graph = json.loads((folder / "workflow-api.json").read_text(encoding="utf-8"))
        expected_graph = pose_workflow(graph["4"]["inputs"]["image"], graph["5"]["inputs"]["image"],
                                       graph["16"]["inputs"]["image"], graph["13"]["inputs"]["filename_prefix"],
                                       self.seed, .65, .85)
        if graph != expected_graph or any(record[key] != source[key] for key in ("source_frame", "track_id", "target_m")):
            raise ValueError("姿態快取的來源、落點或實際工作流不符")
        baseline = read_image(inputs / "opencv.png")
        mask = read_image(inputs / "redraw-mask-full.png", mask=True)
        edited = read_image(folder / "redrawn.png")
        if edited.shape != baseline.shape:
            raise ValueError("姿態快取尺寸不符")
        checks = validate_pose_output(inputs, baseline, edited, mask)
        if not checks["changed_from_opencv_pixels"]:
            raise ValueError("姿態快取沒有重繪變更")
        return record, checks

    def run(self, job):
        try:
            with self.lock:
                job["state"] = "running"
            started = perf_counter()
            queue_seconds = started - job["submitted"]
            # Recreate the complete pack; cached images must match the current annotations/calibration.
            is_preset = job["target_m"] == self.preset["target_m"]
            name = self.preset["name"] if is_preset else "target-" + job["key"][:16]
            build_bundle(job["directory"], self.review, self.fixture, {"name": name, "target_m": job["target_m"]})
            inputs = job["directory"] / name
            key_root = job["directory"].parent
            cached = key_root / "ready.json"
            result = None
            provenance = "new_generation"
            if cached.exists():
                saved = json.loads(cached.read_text(encoding="utf-8"))
                if saved["identity"] == job["identity"]:
                    result = key_root / saved["attempt"] / "result"
                    provenance = "local_cache"
            if result is None and is_preset:
                candidate = self.legacy_root / name / f"seed-{self.seed}"
                if candidate.is_dir():
                    result = candidate
                    provenance = "reviewed_5.3_cache"
            if result is None:
                result = job["directory"] / "result"
                result.mkdir()
                with requests.Session() as session:
                    session.trust_env = False
                    check_comfy_ready(session, self.review.comfy_url)
                    info = session.get(f"{self.review.comfy_url}/object_info", timeout=15)
                    info.raise_for_status()
                    nodes = info.json()
                    if not {"ModelPatchLoader", "ZImageFunControlnet"} <= nodes.keys() or CONTROL_FILE not in nodes["ModelPatchLoader"]["input"]["required"]["name"][0]:
                        raise ValueError("ComfyUI 缺少姿態節點或 ControlNet 模型")
                    identity = {**generation_settings(), "case": name, "seed": self.seed,
                                "input_sha256": bundle_digest(inputs)}
                    generate_candidate(session, self.review.comfy_url, inputs, result, identity)
            record, checks = self.validate_result(result, inputs)
            placement, tactics = self.review.describe_placement(job["target_m"])
            saved_review = key_root / "contact-review.json"
            position_review = (json.loads(saved_review.read_text(encoding="utf-8")) if saved_review.exists()
                               else record.get("position_review", {"status": "needs_review", "contact_measurement": None,
                                                                     "appearance_details": "excluded_by_user"}))
            report = {**tactics, "result_kind": "pose_redraw", "job_id": job["id"], "cache_key": job["key"],
                      "seed": self.seed, "provenance": provenance, "source_version": self.version,
                      "expected_contact_px": placement.applied_foot_px.tolist(),
                      "applied_foot_px": placement.applied_foot_px.tolist(),
                      "verification": checks, "generation": record["generation"], "identity": record["identity"],
                      "position_review": position_review, "active_review": "position_review",
                      "queue_seconds": round(queue_seconds, 3),
                      "ready_seconds": round(perf_counter() - job["submitted"], 3),
                      "animation_coordinate_policy": "requested_target_not_generated_contact"}
            json_file(job["directory"] / "review-report.json", report)
            if provenance == "new_generation":
                json_file(cached, {"identity": job["identity"], "attempt": job["id"]})
            with self.lock:
                job.update(state="ready", report=report, result_dir=result)
        except Exception as exc:
            with self.lock:
                job.update(state="failed", error=f"重繪失敗：{type(exc).__name__}: {exc}。保留預覽，可重試。")

    def contact(self, payload):
        with self.lock:
            job = self.jobs.get(payload["id"])
            if not job or job["state"] != "ready":
                raise ValueError("重繪結果尚未就緒")
            point = payload.get("point")
            measurement = None
            review_status = "needs_review"
            plausible = payload.get("plausible") is True
            if point is not None:
                point = np.asarray(point, dtype=float)
                if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= self.review.frame.shape[1::-1]):
                    raise ValueError("接點必須在來源影像內")
                expected = np.asarray(job["report"]["expected_contact_px"])
                error_px = float(np.linalg.norm(point - expected))
                H = self.review.player.image_to_ice
                error_m = float(np.linalg.norm(project(H, point) - project(H, expected)))
                # Conservative local uncertainty envelope, not an accuracy guarantee.
                offsets = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)]) * 2
                upper_m = max(float(np.linalg.norm(project(H, point + d) - project(H, expected))) for d in offsets)
                within = error_px <= 8 and error_m <= .20
                borderline = within and (error_px + 2 > 8 or upper_m > .20)
                review_status = "usable_for_position_demo" if within and not borderline and plausible else "needs_review"
                measurement = {"method": "user_manual_click", "frame_px": point.tolist(), "annotation_uncertainty_px": 2,
                               "error_px": error_px, "error_m": error_m, "pixel_threshold": 8, "metre_threshold": .20,
                               "status": "near_threshold" if borderline else "estimated_within_threshold" if within else "outside_threshold"}
            position_review = {"status": review_status, "contact_measurement": measurement,
                               "person_and_contact_plausible": plausible, "reviewer": "user_manual_review",
                               "appearance_details": "excluded_by_user"}
            # Preserve the historical report; this exact-input key owns the new manual review.
            json_file(job["directory"] / "contact-review.json", position_review)
            json_file(job["directory"].parent / "contact-review.json", position_review)
            job["report"]["position_review"] = position_review
            json_file(job["directory"] / "review-report.json", job["report"])
            return self.snapshot(job)
