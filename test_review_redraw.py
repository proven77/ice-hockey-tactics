import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from redraw_player import validate_pose_output, bundle_digest, generation_settings, pose_workflow
from review_redraw import RedrawQueue, fingerprint
from refine_player import read_image
from what_if import write_png, filesystem_path, ROOT


class RedrawTests(unittest.TestCase):
    def test_long_unicode_image_paths(self):
        with TemporaryDirectory(dir=filesystem_path(ROOT)) as temp:
            folder = Path(temp)
            while len(str(folder)) < 265:
                folder /= "防守圖像快取-long-directory"
            folder.mkdir(parents=True)
            expected = np.full((8, 8, 3), 127, np.uint8)
            path = folder / "冰刀.png"
            write_png(path, expected)
            self.assertTrue(np.array_equal(read_image(path), expected))

    def test_identity_keeps_full_target_precision(self):
        self.assertNotEqual(fingerprint({"target": [54., 12.5]}), fingerprint({"target": [54.00000001, 12.5]}))
        self.assertNotEqual(fingerprint({"frame": 150}), fingerprint({"frame": 592}))
        with self.assertRaises(ValueError):
            fingerprint({"target": [float("nan"), 0]})

    def test_only_newest_queued_job_survives(self):
        with TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=1) as worker:
            queue = object.__new__(RedrawQueue)
            queue.root, queue.lock, queue.jobs = Path(temp), RLock(), {}
            queue.review = SimpleNamespace(worker=worker)
            entered, release = Event(), Event()
            def run(job):
                with queue.lock:
                    job["state"] = "running"
                entered.set()
                release.wait(5)
                with queue.lock:
                    job["state"] = "ready"
            with patch.object(queue, "identity", side_effect=lambda target: {"target_m": target}), patch.object(queue, "run", side_effect=run):
                first = queue.submit([1, 1])
                self.assertTrue(entered.wait(2))
                second = queue.submit([2, 2])
                third = queue.submit([3, 3])
                self.assertEqual(queue.get(second["id"])["state"], "superseded")
                self.assertEqual(queue.get(first["id"])["state"], "running")
                self.assertEqual(queue.get(third["id"])["state"], "queued")
                queue.cancel(third["id"])
                self.assertEqual(queue.get(third["id"])["state"], "superseded")
                release.set()

    def test_contact_requires_plausibility_and_uncertainty_margin(self):
        with TemporaryDirectory() as temp:
            queue = object.__new__(RedrawQueue)
            queue.lock = RLock()
            queue.review = SimpleNamespace(frame=np.zeros((300, 300, 3), np.uint8),
                                          player=SimpleNamespace(image_to_ice=np.diag([.03, .03, 1.])))
            directory = Path(temp) / "key" / "attempt"
            directory.mkdir(parents=True)
            queue.jobs = {"a": {"id": "a", "state": "ready", "target_m": [1, 2], "key": "key",
                                "directory": directory, "report": {"expected_contact_px": [100, 100]}}}
            def measure(point, plausible=False):
                return queue.contact({"id": "a", "point": point, "plausible": plausible})["report"]["position_review"]
            self.assertEqual(measure([103, 100])["status"], "needs_review")
            self.assertEqual(measure([103, 100], True)["status"], "usable_for_position_demo")
            borderline = measure([105, 100], True)
            self.assertEqual(borderline["status"], "needs_review")
            self.assertEqual(borderline["contact_measurement"]["status"], "near_threshold")
            self.assertEqual(measure([109, 100], True)["status"], "needs_review")
            self.assertIsNone(measure(None)["contact_measurement"])
            self.assertEqual(json.loads((directory.parent / "contact-review.json").read_text())["status"], "needs_review")
            for point in ([float("nan"), 100], [-1, 100], [300, 100], [1, 2, 3]):
                with self.assertRaises(ValueError):
                    measure(point)

    def test_pixel_checks_reject_changes_outside_mask_and_protected_objects(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            baseline = np.zeros((12, 12, 3), np.uint8)
            mask = np.zeros((12, 12), np.uint8)
            mask[4:8, 4:8] = 255
            for name, image in {"original.png": baseline, "repaired-background.png": baseline,
                                "original-repair-mask.png": np.zeros_like(mask),
                                "protected-mask-full.png": np.zeros_like(mask)}.items():
                cv2.imwrite(str(folder / name), image)
            edited = baseline.copy()
            edited[5, 5] = 100
            self.assertEqual(validate_pose_output(folder, baseline, edited, mask)["protected_changed_pixels"], 0)
            edited[0, 0] = 100
            with self.assertRaises(ValueError):
                validate_pose_output(folder, baseline, edited, mask)
            edited[0, 0] = 0
            protected = np.zeros_like(mask)
            protected[5, 5] = 255
            cv2.imwrite(str(folder / "protected-mask-full.png"), protected)
            with self.assertRaises(ValueError):
                validate_pose_output(folder, baseline, edited, mask)

    def test_cache_checks_actual_workflow_and_source_metadata(self):
        with TemporaryDirectory() as temp:
            inputs, result = Path(temp) / "inputs", Path(temp) / "result"
            inputs.mkdir(); result.mkdir()
            baseline = np.zeros((12, 12, 3), np.uint8)
            mask = np.zeros((12, 12), np.uint8)
            mask[4:8, 4:8] = 255
            source = {"name": "test", "source_frame": 150, "track_id": 4, "target_m": [1., 2.]}
            (inputs / "report.json").write_text(json.dumps(source), encoding="utf-8")
            (inputs / "pose-target.json").write_text("{}", encoding="utf-8")
            for name in ("original.png", "repaired-background.png", "opencv.png", "conditioning-image.png",
                         "pose-target.png", "stick-guide.png"):
                cv2.imwrite(str(inputs / name), baseline)
            for name, image in {"original-repair-mask.png": np.zeros_like(mask),
                                "protected-mask-full.png": np.zeros_like(mask), "redraw-mask-full.png": mask}.items():
                cv2.imwrite(str(inputs / name), image)
            edited = baseline.copy(); edited[5, 5] = 100
            cv2.imwrite(str(result / "redrawn.png"), edited)
            record = {**source, "identity": {**generation_settings(), "case": "test", "seed": 105,
                                             "input_sha256": bundle_digest(inputs)}}
            (result / "report.json").write_text(json.dumps(record), encoding="utf-8")
            graph = pose_workflow("input", "mask", "pose", "output", 105, .65, .85)
            workflow = result / "workflow-api.json"
            workflow.write_text(json.dumps(graph), encoding="utf-8")
            queue = object.__new__(RedrawQueue); queue.seed = 105
            self.assertEqual(queue.validate_result(result, inputs)[1]["protected_changed_pixels"], 0)
            graph["11"]["inputs"]["seed"] = 104
            workflow.write_text(json.dumps(graph), encoding="utf-8")
            with self.assertRaises(ValueError):
                queue.validate_result(result, inputs)


if __name__ == "__main__":
    unittest.main()
