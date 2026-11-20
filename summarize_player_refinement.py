"""Reload refinement outputs, check protected pixels and assemble comparisons."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_player_redraw import Crop, json_file
from refine_player import read_image, refine_mask, verify_output
from what_if import write_png


def caption(image, title):
    panel = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .64, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def case_comparison(source, results, name):
    info = json.loads((source / "report.json").read_text(encoding="utf-8"))
    x, y, width, height = info["crop"]["xywh"]
    if width != height:
        raise ValueError("Expected a square source crop")
    crop = Crop(x, y, width)
    baseline = read_image(source / "opencv.png")
    mask, conditioning_mask, anchor_lock = refine_mask(source, crop, info)
    conditioning = read_image(source / "conditioning-image.png")
    images = [("OpenCV", baseline)]
    checks = []
    for seed in (103, 104, 105):
        job = results / f"seed-{seed}"
        refined = read_image(job / "refined.png")
        report = json.loads((job / "report.json").read_text(encoding="utf-8"))
        verified = verify_output(source, baseline, refined, mask, anchor_lock)
        if any(value != report["verification"][key] for key, value in verified.items()):
            raise ValueError(f"Saved verification does not match the output: {job}")
        raw = read_image(job / "comfy-raw.png")
        raw_outside_changes = int(np.count_nonzero(
            np.any(raw != conditioning, axis=2) & (conditioning_mask == 0)))
        checks.append({
            "case": name, "seed": seed, **verified,
            "request_seconds": report["generation"]["request_seconds"],
            "peak_device_mb": report["generation"]["gpu_total_used_peak_mb"],
            "raw_changed_outside_mask": raw_outside_changes,
        })
        images.append((f"Seed {seed}", refined))

    cards, foot_cards = [], []
    anchor_x, anchor_y = np.rint(info["pose"]["contact_frame_px"]).astype(int)
    for label, image in images:
        panel = cv2.resize(image[crop.region], (320, 320), interpolation=cv2.INTER_NEAREST)
        cards.append(caption(panel, label))
        foot = image[anchor_y - 75:anchor_y + 35, anchor_x - 48:anchor_x + 62]
        foot = cv2.resize(foot, (330, 330), interpolation=cv2.INTER_NEAREST)
        foot_cards.append(caption(foot, label))
    return checks, caption(cv2.hconcat(cards), name), caption(cv2.hconcat(foot_cards), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("output-videos/whatif-defense-step5-1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads((args.bundle / "verification.json").read_text(encoding="utf-8"))
    rows, foot_rows, results = [], [], []
    for entry in specification["cases"]:
        name = entry["name"]
        checks, row, feet = case_comparison(args.bundle / name, args.output / name, name)
        results.extend(checks)
        rows.append(row)
        foot_rows.append(feet)
    write_png(args.output / "overview.png", cv2.vconcat(rows))
    write_png(args.output / "skate-detail.png", cv2.vconcat(foot_rows))
    subsequent = [result["request_seconds"] for result in results
            if (result["case"], result["seed"]) != ("frame150-open-ice", 103)]
    memory_samples = [result["peak_device_mb"] for result in results if result["peak_device_mb"] is not None]
    timing = {
        "first_request_seconds": results[0]["request_seconds"],
        "subsequent_request_median_seconds": float(np.median(subsequent)),
        "subsequent_request_min_seconds": min(subsequent),
        "subsequent_request_max_seconds": max(subsequent),
        "sampled_peak_gpu_total_used_mb": max(memory_samples, default=None),
    }
    json_file(args.output / "verification.json", {"cases": results, "timing": timing})
    print(json.dumps(timing, indent=2))
    print(f"Verified {len(results)} candidates and assembled comparisons.")


if __name__ == "__main__":
    main()
