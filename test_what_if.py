import json
import unittest

import numpy as np

from what_if import ROOT, composite, place_player, prepare_player, project


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


if __name__ == "__main__":
    unittest.main()
