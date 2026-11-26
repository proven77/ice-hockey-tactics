"""Verify downloaded PNG/JSON pairs and fully decode each tactical MP4."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from defense_review import Review
from redraw_player import validate_pose_output
from refine_player import read_image
from tactical_animation import write_animation
from what_if import ROOT, RINK, composite, filesystem_path


def verify(browser, cache):
    reviews = {}
    results = []
    try:
        for folder, case, repair in (("frame150", "what_if_case.json", "whatif-defense-step1"),
                                    ("frame592", "what_if_shot_case.json", "whatif-defense-step3-guided")):
            reviews[folder] = Review(json.loads((ROOT / case).read_text(encoding="utf-8")),
                                     ROOT / "output-videos" / repair, 8191)
        for path in sorted(browser.glob("frame*/*/report.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            review = reviews[path.parent.parent.name]
            image = read_image(path.parent / "what-if.png")
            annotated = read_image(path.parent / "what-if-annotated.png")
            assert image.shape == annotated.shape == review.frame.shape
            assert np.any(image != annotated), "Missing analysis overlay"
            kind = report["result_kind"]
            if kind == "original":
                assert np.array_equal(image, review.frame)
                assert report["target_m"] == report["source_m"]
            elif kind == "opencv_preview":
                placement, _ = review.describe_placement(report["target_m"])
                assert np.array_equal(image, composite(review.background[0], review.player, placement))
            else:
                pack = next((cache / report["cache_key"] / report["job_id"]).glob("*/redraw-mask-full.png")).parent
                checks = validate_pose_output(pack, read_image(pack / "opencv.png"), image,
                                              read_image(pack / "redraw-mask-full.png", mask=True))
                assert checks == report["verification"]
            item = {"path": str(path.relative_to(browser)), "kind": kind, "frame": report["source_frame"],
                    "target_m": report["target_m"], "image_size_px": list(image.shape[1::-1])}
            movie = path.parent / "tactical-path.mp4"
            if movie.exists():
                video = cv2.VideoCapture(str(movie))
                count = 0
                try:
                    while True:
                        ok, frame = video.read()
                        if not ok:
                            break
                        assert frame.shape[:2] == (550, 1000)
                        count += 1
                finally:
                    video.release()
                assert count == 90
                with TemporaryDirectory(prefix="hockey-verify-") as temp:
                    expected = Path(temp) / "expected.mp4"
                    write_animation(expected, RINK, review.rink_size, review.tracks, report,
                                    review.defense.lane.coverage if review.defense.lane else None)
                    assert movie.read_bytes() == expected.read_bytes()
                item.update(decoded_frames=count, animation_matches_target=True)
            results.append(item)
        assert results, "No downloaded reports found"
        return {"artifacts": results, "images": len(results),
                "movies": sum("decoded_frames" in result for result in results), "all_passed": True}
    finally:
        for review in reviews.values():
            review.worker.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, default=ROOT / "output-videos/whatif-defense-step5-4/release-live")
    parser.add_argument("--cache", type=Path, default=ROOT / "output-videos/defense-review-redraw")
    parser.add_argument("--output", type=Path, default=ROOT / "output-videos/whatif-defense-step5-4/final-artifact-verification.json")
    args = parser.parse_args()
    record = verify(filesystem_path(args.browser), filesystem_path(args.cache))
    filesystem_path(args.output).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "artifacts"}))
