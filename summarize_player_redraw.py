"""Compare fixed pose-redraw candidates and verify full-frame protection."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_player_redraw import Crop, json_file
from redraw_player import validate_pose_output
from refine_player import read_image
from summarize_player_refinement import caption
from what_if import write_png


def foot_grid(image, anchor, name):
    x, y = np.rint(anchor).astype(int)
    left, top = x - 35, y - 35
    patch = cv2.resize(image[top:top + 70, left:left + 70], (350, 350), interpolation=cv2.INTER_NEAREST)
    for xx in range(left + (-left % 5), left + 70, 5):
        u = (xx - left) * 5
        cv2.line(patch, (u, 0), (u, 350), (85, 125, 85), 1)
        cv2.putText(patch, str(xx), (u, 13), 0, .32, (25, 25, 25), 1)
    for yy in range(top + (-top % 5), top + 70, 5):
        v = (yy - top) * 5
        cv2.line(patch, (0, v), (350, v), (85, 125, 85), 1)
        cv2.putText(patch, str(yy), (0, v + 10), 0, .32, (25, 25, 25), 1)
    return caption(patch, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("output-videos/whatif-defense-step5-1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads((args.bundle / "verification.json").read_text(encoding="utf-8"))
    rows, feet, records = [], [], []
    for entry in specification["cases"]:
        name = entry["name"]
        folder = args.bundle / name
        source = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        x, y, width, height = source["crop"]["xywh"]
        if width != height:
            raise ValueError("Expected square crop")
        crop = Crop(x, y, width)
        baseline = read_image(folder / "opencv.png")
        mask = read_image(folder / "redraw-mask-full.png", mask=True)
        pose = read_image(folder / "pose-target.png")
        panels = [caption(cv2.resize(baseline[crop.region], (300, 300)), "OpenCV"),
                  caption(cv2.resize(pose, (300, 300)), "Target pose")]
        foot_panels = [foot_grid(baseline, source["pose"]["contact_frame_px"], "OpenCV")]
        for seed in (103, 104, 105):
            job = args.output / name / f"seed-{seed}"
            generated = read_image(job / "redrawn.png")
            report = json.loads((job / "report.json").read_text(encoding="utf-8"))
            checks = validate_pose_output(folder, baseline, generated, mask)
            if checks != report["verification"]:
                raise ValueError(f"Saved pixels do not match verification: {job}")
            records.append({"case": name, "seed": seed, "generation": report["generation"], **checks})
            panels.append(caption(cv2.resize(generated[crop.region], (300, 300)), f"Seed {seed}"))
            contact_detail = foot_grid(generated, source["pose"]["contact_frame_px"], f"Seed {seed}")
            foot_panels.append(contact_detail)
            write_png(job / "contact-detail.png", cv2.resize(contact_detail, None, fx=2, fy=2,
                                                            interpolation=cv2.INTER_NEAREST))
        rows.append(caption(cv2.hconcat(panels), name))
        feet.append(caption(cv2.hconcat(foot_panels), name))
    write_png(args.output / "overview.png", cv2.vconcat(rows))
    write_png(args.output / "contact-grid.png", cv2.vconcat(feet))
    times = [record["generation"]["request_seconds"] for record in records]
    memory = [record["generation"]["gpu_total_used_peak_mb"] for record in records
              if record["generation"]["gpu_total_used_peak_mb"] is not None]
    summary = {"phase": "5.3", "candidates": records, "timing": {
        "first_controlnet_request_seconds": times[0], "subsequent_request_median_seconds": float(np.median(times[1:])),
        "subsequent_request_range_seconds": [min(times[1:]), max(times[1:])],
        "sampled_gpu_total_used_peak_mb": max(memory, default=None),
    }}
    json_file(args.output / "verification.json", summary)
    print(json.dumps(summary["timing"], indent=2))
    print(f"Verified {len(records)} candidates and produced overview and contact grids.")


if __name__ == "__main__":
    main()
