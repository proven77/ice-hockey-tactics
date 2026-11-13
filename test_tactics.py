import unittest

import numpy as np

from tactical_view_converter.homography import Homography
from tactics import MatchStats, project_frame


class TacticalAnalysisTests(unittest.TestCase):
    def test_projection_rejects_outlier_and_out_of_rink_point(self):
        pixels = np.array([[10, 10], [210, 10], [210, 110], [10, 110], [110, 60], [40, 75], [180, 70]])
        metres = (pixels - 10) / 10
        metres[-1] = [3, 3]
        homography = Homography(pixels, metres, ransac_threshold=0.1, min_inliers=6)
        self.assertEqual(homography.inliers, 6)
        players = {1: {"bbox": [50, 10, 70, 30]}, 2: {"bbox": [250, 30, 270, 50]}}
        positions, puck = project_frame(homography, players, [69, 49, 71, 51], (20, 10))
        self.assertEqual(set(positions), {1})
        np.testing.assert_allclose(positions[1], (5, 2), atol=1e-5)
        np.testing.assert_allclose(puck, (6, 4), atol=1e-5)

    def test_degenerate_calibration_is_rejected(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        with self.assertRaises(ValueError):
            Homography(points, points, ransac_threshold=0.1)

    def test_distance_breaks_on_missing_frames_and_rejects_jumps(self):
        stats = MatchStats(10)
        for positions in ({1: (0, 0)}, {1: (0.3, 0.4)}, {}, {1: (8, 8)}, {1: (20, 20)}, {1: (21, 20)}):
            stats.update(positions, None, {})
        self.assertAlmostEqual(stats.distances[1], 1.5)
        self.assertEqual(stats.distance_segments[1], 2)

    def test_possession_requires_consecutive_unambiguous_evidence(self):
        stats = MatchStats(10)
        puck = (2, 2)
        players = {11: (2, 3), 12: (6, 2)}
        teams = {11: 1, 12: 2}
        self.assertEqual(stats.update(players, puck, teams), (None, 0))
        self.assertEqual(stats.update(players, puck, teams), (11, 1))
        self.assertEqual(stats.update(players, None, teams), (None, 0))
        self.assertEqual(stats.update(players, puck, teams), (None, 0))
        self.assertEqual(stats.update({11: (2, 3), 12: (2, 1)}, puck, teams), (None, 0))
        self.assertEqual(stats.possession_pct(), [80.0, 20.0, 0.0])


if __name__ == "__main__":
    unittest.main()
