import json
import unittest

import numpy as np

from what_if import ROOT, composite, place_player, placement_report, prepare_player, project


class WhatIfPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / "what_if_case.json").read_text(encoding="utf-8"))
        cls.rink_size = (60.96, 25.908)
        cls.frame = np.full((1080, 1920, 3), 235, dtype=np.uint8)
        cls.frame[370:546, 895:996] = (40, 80, 130)
        cls.player = prepare_player(cls.frame, config)
        cls.background = cls.frame.copy()
        cls.background[cls.player.repair > 0] = 235

    def test_three_targets_reuse_background_and_preserve_unedited_pixels(self):
        background_before = self.background.copy()
        for target in ((44.65, 15.82), (40.8, 14.8), (45, 13)):
            with self.subTest(target=target):
                placement = place_player(self.player, target, self.rink_size)
                edited = composite(self.background, self.player, placement)
                np.testing.assert_allclose(placement.applied_foot_px, placement.target_px, atol=1e-9)
                np.testing.assert_allclose(project(self.player.image_to_ice, placement.applied_foot_px), target, atol=1e-9)
                x, y = np.rint(placement.applied_foot_px).astype(int)
                self.assertGreater(placement.alpha[y - placement.region[0].start, x - placement.region[1].start], 0)
                allowed = self.player.repair > 0
                allowed[placement.region] |= (placement.alpha > 0) | (placement.shadow > 0)
                np.testing.assert_array_equal(edited[~allowed], self.frame[~allowed])
                np.testing.assert_array_equal(edited[self.player.protected], self.frame[self.player.protected])
                self.assertTrue(np.any(edited[placement.region] != self.background[placement.region]))
        np.testing.assert_array_equal(self.background, background_before)

    def test_conflicts_have_distinct_errors(self):
        cases = (
            ((1250, 737), "protected object"),
            ((1915, 520), "cut off by the frame"),
            ((1922, 520), "outside the camera view"),
        )
        for pixels, message in cases:
            with self.subTest(pixels=pixels):
                target = project(self.player.image_to_ice, pixels)
                with self.assertRaisesRegex(ValueError, message):
                    place_player(self.player, target, self.rink_size)

    def test_rectangular_corners_are_outside_the_rounded_boards(self):
        with self.assertRaisesRegex(ValueError, "outside the rounded boards"):
            place_player(self.player, (0.1, 0.1), self.rink_size)

    def test_shadow_and_small_cutout_overlap_leave_protected_pixels_unchanged(self):
        for offset, overlaps_player in (([-1.5, 4.75], False), ([-1.25, 4.75], True)):
            with self.subTest(offset=offset):
                target = self.player.source_m + offset
                placement = place_player(self.player, target, self.rink_size)
                edited = composite(self.background, self.player, placement)
                _, report = placement_report(self.frame, edited, self.player, placement)
                self.assertGreater(report["clipped_shadow_pixels"], 0)
                self.assertEqual(report["clipped_player_pixels"] > 0, overlaps_player)
                self.assertEqual(report["protected_changed_pixels"], 0)
                self.assertEqual(report["outside_mask_changed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
