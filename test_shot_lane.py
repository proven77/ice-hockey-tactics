import unittest
from types import SimpleNamespace

import numpy as np

from shot_lane import DefenseMeasurement, ShotLane


class ShotLaneTests(unittest.TestCase):
    def test_rays_stop_at_goal_line_and_at_puck(self):
        lane = ShotLane([0, 0], [[10, -1], [10, 1]], 0.5)
        for position in ([12, 0], [-2, 0], [5, 2]):
            with self.subTest(position=position):
                result = lane.coverage(position)
                self.assertEqual(result["blocked_pct"], 0)
                self.assertFalse(result["central_blocked"])
                self.assertEqual(result["blocked_goal_spans_m"], [])

    def test_angular_sampling_across_atan2_wrap(self):
        forward = ShotLane([0, 0], [[10, -1], [10, 1]], 0.45)
        backward = ShotLane([0, 0], [[-10, -1], [-10, 1]], 0.45)
        a = forward.coverage([6, 0])
        b = backward.coverage([-6, 0])
        self.assertAlmostEqual(forward.opening_deg, backward.opening_deg)
        self.assertAlmostEqual(a["blocked_pct"], b["blocked_pct"])
        self.assertTrue(a["central_blocked"])
        self.assertAlmostEqual(a["blocked_pct"], 75.3, delta=0.5)
        self.assertAlmostEqual(a["blocked_goal_spans_m"][0][0][0], 10)
        self.assertEqual(forward.coverage([2, 0])["blocked_pct"], 100)

    def test_missing_puck_and_bad_width_withhold_measurement(self):
        player = SimpleNamespace(
            image_to_ice=np.eye(3), source_m=np.array([50, 8]), repair=np.zeros((40, 80), dtype=np.uint8)
        )
        evidence = {"calibration_reviewed": True, "goal_side": "right", "goal_posts_px": [[57.6, 11.8], [57.6, 14.1]],
                    "blocking_radius_m": 0.45}
        case = {"frame": 150, "video": "sample.mp4", "track_id": 4, "shot_lane": evidence}
        for puck, message in ((None, "冰球"), ([47, 12], "門柱投影間距")):
            with self.subTest(puck=puck):
                evidence["puck_px"] = puck
                result = DefenseMeasurement(case, player, (60.96, 25.908), []).measure([52, 10])
                self.assertEqual(result["status"], "needs_review")
                self.assertIn(message, result["reason"])
                self.assertNotIn("after", result)


if __name__ == "__main__":
    unittest.main()
