"""Build the fixed D1 redraw inputs from verified What-If background caches."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from defense_review import Review
from what_if import ROOT, placement_report, composite, project, write_png


CONDITION_SIZE = 512
JOINT_NAMES = (
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
)
# COCO-18 limb order and colors used by the OpenPose annotator in ControlNet.
LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
PALETTE = cv2.cvtColor(np.uint8([[(i * 10, 255, 255) for i in range(18)]]), cv2.COLOR_HSV2BGR)[0]


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    side: int

    @property
    def region(self):
        return np.s_[self.y:self.y + self.side, self.x:self.x + self.side]

    @property
    def scale(self):
        return CONDITION_SIZE / self.side

    def to_local(self, points):
        return (np.asarray(points) - [self.x, self.y] + .5) * self.scale - .5

    def to_frame(self, points):
        return (np.asarray(points) + .5) / self.scale + [self.x, self.y] - .5

    def sample(self, image, *, mask=False):
        interpolation = cv2.INTER_NEAREST_EXACT if mask else cv2.INTER_LINEAR
        return cv2.resize(image[self.region], (CONDITION_SIZE, CONDITION_SIZE), interpolation=interpolation)

    def restore(self, patch, *, mask=False):
        if patch.shape[:2] != (CONDITION_SIZE, CONDITION_SIZE):
            raise ValueError("Expected a 512 x 512 conditioning image or mask")
        interpolation = cv2.INTER_NEAREST_EXACT if mask else cv2.INTER_LINEAR
        return cv2.resize(patch, (self.side, self.side), interpolation=interpolation)


def crop_around(mask, padding=28):
    x, y, w, h = cv2.boundingRect(mask)
    height, width = mask.shape
    side = max(w, h) + 2 * padding
    if not w or side > min(CONDITION_SIZE, width, height):
        raise ValueError("The redraw region cannot fit the fixed conditioning crop")
    left = max(0, min(x + (w - side) // 2, width - side))
    top = max(0, min(y + (h - side) // 2, height - side))
    return Crop(left, top, side)


def place_landmarks(player, placement, fixture):
    def transform(points):
        return placement.applied_foot_px + (np.asarray(points, dtype=float) - player.source_px) * placement.scale

    source = fixture["source_joints_px"]
    offsets = fixture["target_joint_offsets_source_px"]
    support = fixture["support_ankle"]
    if support not in source or any(offsets.get(support, [0, 0])):
        raise ValueError("The support ankle must stay fixed")
    if (source.keys() - set(JOINT_NAMES)) or (offsets.keys() - source.keys()):
        raise ValueError("Unknown joints in pose annotations")
    placed = {name: transform(point) for name, point in source.items()}
    target = {name: transform(np.asarray(point) + offsets.get(name, [0, 0])) for name, point in source.items()}
    stick = np.asarray(fixture["source_stick_px"])
    stick_offsets = np.asarray(fixture["target_stick_offsets_source_px"])
    return placed, target, transform(stick), transform(stick + stick_offsets)


def pose_map(shape, joints):
    canvas = np.zeros((*shape, 3), np.uint8)
    points = {index: tuple(np.rint(joints[name]).astype(int))
              for index, name in enumerate(JOINT_NAMES) if name in joints}
    for index, (start, end) in enumerate(LIMBS):
        if start in points and end in points:
            color = tuple(int(value * .6) for value in PALETTE[index])
            cv2.line(canvas, points[start], points[end], color, 8, cv2.LINE_AA)
    for index, point in points.items():
        cv2.circle(canvas, point, 4, tuple(map(int, PALETTE[index])), -1, cv2.LINE_AA)
    return canvas


def line_mask(shape, points, thickness):
    mask = np.zeros(shape, np.uint8)
    cv2.polylines(mask, [np.rint(points).astype(np.int32)], False, 255, thickness)
    return mask


def dilate(mask, pixels):
    size = 2 * pixels + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def build_masks(player, placement, placed, target, placed_stick, target_stick):
    shape = player.repair.shape
    silhouette = np.zeros(shape, np.uint8)
    silhouette[placement.region] = (placement.alpha > 0).astype(np.uint8) * 255
    shadow = np.zeros_like(silhouette)
    shadow[placement.region] = (placement.shadow > 0).astype(np.uint8) * 255
    core = cv2.erode(silhouette, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    seam = (dilate(silhouette, 5) & ~core) | (shadow & ~core)
    shifts = np.vstack([np.asarray([target[joint] - placed[joint] for joint in target]),
                        target_stick - placed_stick])
    padding = int(np.ceil(np.linalg.norm(shifts, axis=1).max())) + 6
    guide = np.any(pose_map(shape, target) > 0, axis=2).astype(np.uint8) * 255
    redraw = dilate(silhouette | shadow, padding) | dilate(guide | line_mask(shape, target_stick, 5), 8)
    for mask in (seam, redraw):
        mask[player.protected] = 0
    return seam, redraw, padding


def compose_local(background, patch, mask, crop):
    """Paste the local result into the frame using the full-size allow mask."""
    result = background.copy()
    original_crop = result[crop.region]
    np.copyto(original_crop, crop.restore(patch), where=(mask[crop.region] != 0)[..., None])
    return result


def verify_input_pack(frame, background, baseline, player, placement, seam, redraw, crop):
    if np.any(background[player.repair == 0] != frame[player.repair == 0]):
        raise ValueError("Source repair changed pixels outside its mask")
    if np.any(player.protected & ((seam > 0) | (redraw > 0))):
        raise ValueError("A conditioning mask overlaps protected pixels")
    if np.count_nonzero(redraw) != np.count_nonzero(redraw[crop.region]):
        raise ValueError("Redraw mask is not contained by the crop")
    if np.any(baseline[redraw == 0] != background[redraw == 0]):
        raise ValueError("The redraw mask does not cover the baseline edit")
    anchor = placement.applied_foot_px
    anchor_local = crop.to_local(anchor)
    x, y = np.rint(anchor).astype(int)
    if not redraw[y, x]:
        raise ValueError("The support skate is outside the allowed redraw mask")

    probe = np.full((CONDITION_SIZE, CONDITION_SIZE, 3), (23, 204, 251), dtype=np.uint8)
    blended = compose_local(background, probe, redraw, crop)
    changed = np.any(blended != background, axis=2)
    counts = {
        "seam_mask_roundtrip_mismatch_pixels": int(np.count_nonzero(
            seam[crop.region] != crop.restore(crop.sample(seam, mask=True), mask=True))),
        "redraw_mask_roundtrip_mismatch_pixels": int(np.count_nonzero(
            redraw[crop.region] != crop.restore(crop.sample(redraw, mask=True), mask=True))),
        "protected_mask_roundtrip_mismatch_pixels": int(np.count_nonzero(
            player.protected[crop.region] != (crop.restore(crop.sample(
                player.protected.astype(np.uint8), mask=True), mask=True) > 0))),
        "probe_outside_target_mask_changed_pixels": int(np.count_nonzero(changed & (redraw == 0))),
        "probe_outside_both_masks_changed_pixels": int(np.count_nonzero(
            np.any(blended != frame, axis=2) & (redraw == 0) & (player.repair == 0))),
        "probe_protected_changed_pixels": int(np.count_nonzero(np.any(blended != frame, axis=2) & player.protected)),
    }
    marker = np.zeros((CONDITION_SIZE, CONDITION_SIZE), dtype=np.uint8)
    cv2.circle(marker, tuple(np.rint(anchor_local).astype(int)), 5, 255, -1)
    yy, xx = np.nonzero(crop.restore(marker, mask=True))
    marker_center = np.array([xx.mean() + crop.x, yy.mean() + crop.y])
    reprojection_error = float(np.linalg.norm(marker_center - anchor))
    if any(counts.values()) or not np.any(changed) or not reprojection_error <= 1:
        raise ValueError(f"Crop or paste verification failed: {counts}, marker error {reprojection_error:.3f} px")
    return {
        **counts, "probe_changed_pixels": int(np.count_nonzero(changed)),
        "anchor_transform_error_px": float(np.linalg.norm(crop.to_frame(anchor_local) - anchor)),
        "anchor_marker_roundtrip_error_px": reprojection_error,
    }


def json_file(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summary_row(name, local, pose, stick, protected, redraw, anchor, clipped):
    targets = (pose > 0).any(axis=2)
    posed = local.copy()
    posed[targets] = pose[targets]
    posed[stick > 0] = (249, 246, 238)
    masked = local.copy()
    masked[redraw > 0] = ((58, 170, 121) + .4 * masked[redraw > 0]).clip(0, 255).astype(np.uint8)
    masked[protected > 0] = ((124, 56, 138) + .4 * masked[protected > 0]).clip(0, 255).astype(np.uint8)
    panels = []
    for panel, title in zip((local.copy(), posed, masked),
                            ("OpenCV baseline", "Target pose and stick", "Editable (green) / protected (purple)")):
        pt = tuple(np.rint(anchor).astype(int))
        cv2.drawMarker(panel, pt, (51, 199, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        tile = cv2.resize(panel, (300, 300), interpolation=cv2.INTER_AREA)
        card = cv2.copyMakeBorder(tile, 8, 27, 6, 6, cv2.BORDER_CONSTANT, value=(23, 25, 28))
        cv2.putText(card, title, (8, 327), cv2.FONT_HERSHEY_SIMPLEX, .43, (241, 241, 241), 1, cv2.LINE_AA)
        panels.append(card)
    row = cv2.copyMakeBorder(cv2.hconcat(panels), 35, 8, 0, 0, cv2.BORDER_CONSTANT, value=(23, 25, 28))
    cv2.putText(row, f"{name}  |  OpenCV clipping: {clipped} px", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (220, 232, 245), 1, cv2.LINE_AA)
    return row


def build_bundle(destination, review, fixture, entry):
    placement, tactics = review.describe_placement(entry["target_m"])
    background, repair_prompt_id = review.background
    frame, player = review.frame, review.player
    baseline = composite(background, player, placement)
    baseline_mask, baseline_checks = placement_report(frame, baseline, player, placement)
    placed, target, placed_stick, target_stick = place_landmarks(player, placement, fixture)
    seam, redraw, padding = build_masks(player, placement, placed, target, placed_stick, target_stick)
    crop = crop_around(redraw)
    checks = verify_input_pack(frame, background, baseline, player, placement, seam, redraw, crop)
    anchor_local = crop.to_local(placement.applied_foot_px)
    local_joints = {joint: crop.to_local(point).tolist() for joint, point in target.items()}
    local_stick = crop.to_local(target_stick)
    pose = pose_map((CONDITION_SIZE, CONDITION_SIZE), local_joints)
    stick = line_mask(pose.shape[:2], local_stick, 4)
    local = crop.sample(baseline)
    protected = player.protected.astype(np.uint8) * 255
    protected_local = crop.sample(protected, mask=True)
    redraw_local = crop.sample(redraw, mask=True)
    out = destination / entry["name"]
    out.mkdir()
    for name, image in {
        "original.png": frame,
        "repaired-background.png": background,
        "opencv.png": baseline,
        "original-repair-mask.png": player.repair,
        "baseline-edit-mask.png": baseline_mask.astype(np.uint8) * 255,
        "redraw-mask-full.png": redraw,
        "seam-mask-full.png": seam,
        "protected-mask-full.png": protected,
        "conditioning-image.png": local,
        "conditioning-background.png": crop.sample(background),
        "redraw-mask.png": redraw_local,
        "seam-mask.png": crop.sample(seam, mask=True),
        "protected-mask.png": protected_local,
        "pose-target.png": pose,
        "stick-guide.png": stick,
    }.items():
        write_png(out / name, image)

    keypoints = [value for joint in JOINT_NAMES
                 for value in ([*local_joints[joint], 1] if joint in local_joints else [0, 0, 0])]
    json_file(out / "pose-target.json", {
        "version": 1.3, "canvas_width": CONDITION_SIZE, "canvas_height": CONDITION_SIZE,
        "people": [{"pose_keypoints_2d": keypoints}],
    })
    report = {
        **tactics, "phase": "5.1", "name": entry["name"], "baseline": baseline_checks,
        "image_size_px": list(frame.shape[1::-1]), "image_to_ice": player.image_to_ice.tolist(),
        "repair_cache": fixture["repair_run"], "repair_prompt_id": repair_prompt_id,
        "original_sha256": sha256(frame.tobytes()).hexdigest(),
        "repaired_background_sha256": sha256(background.tobytes()).hexdigest(),
        "crop": {"xywh": [crop.x, crop.y, crop.side, crop.side], "conditioning_size_px": CONDITION_SIZE,
                 "scale": crop.scale, "mapping": "local = (frame - origin + 0.5) * scale - 0.5"},
        "pose": {
            "annotation": "manual_2d_estimate", "support_ankle": fixture["support_ankle"],
            "support_skate_note": fixture["support_skate_note"], "contact_source_px": player.source_px.tolist(),
            "contact_frame_px": placement.applied_foot_px.tolist(), "contact_condition_px": anchor_local.tolist(),
            "contact_ice_m": project(player.image_to_ice, placement.applied_foot_px).tolist(),
            "inferred_source_joints": fixture["inferred_source_joints"], "source_joints_px": fixture["source_joints_px"],
            "baseline_joints_frame_px": {name: point.tolist() for name, point in placed.items()},
            "target_joints_frame_px": {name: point.tolist() for name, point in target.items()},
            "target_joints_condition_px": local_joints, "target_stick_frame_px": target_stick.tolist(),
            "target_stick_condition_px": local_stick.tolist(),
        },
        "masks": {"redraw_padding_px": padding, "redraw_pixels": int(np.count_nonzero(redraw)),
                  "seam_pixels": int(np.count_nonzero(seam))},
        "verification": checks,
    }
    json_file(out / "report.json", report)
    row = summary_row(entry["name"], local, pose, stick, protected_local, redraw_local, anchor_local,
                      placement.clipped_player_pixels)
    return {"name": entry["name"], "frame": review.case["frame"], "target_m": placement.target_m.tolist(),
            "contact_frame_px": report["pose"]["contact_frame_px"], "crop_xywh": report["crop"]["xywh"],
            "clipped_player_pixels": placement.clipped_player_pixels, **checks}, row


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed ComfyUI input packs for experiment 5.1.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=ROOT / "player_redraw_cases.json")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if spec["schema_version"] != 1 or not spec["placements"]:
            raise ValueError("Expected a nonempty version 1 fixture specification")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".redraw-inputs-") as temporary:
            staging = Path(temporary) / "bundle"
            staging.mkdir()
            reviews, results, rows = {}, [], []
            for entry in spec["placements"]:
                name = entry["name"]
                if not name or not name.replace("-", "").replace("_", "").isalnum():
                    raise ValueError(f"Invalid case name: {name}")
                key = entry["fixture"]
                fixture = spec["fixtures"][key]
                if key not in reviews:
                    case = json.loads((ROOT / fixture["case"]).read_text(encoding="utf-8"))
                    reviews[key] = Review(case, ROOT / fixture["repair_run"], 8191)
                report, row = build_bundle(staging, reviews[key], fixture, entry)
                results.append(report)
                rows.append(row)
            write_png(staging / "overview.png", cv2.vconcat(rows))
            summary = {"phase": "5.1", "conditioning_size_px": CONDITION_SIZE, "cases": results}
            json_file(staging / "verification.json", summary)
            staging.rename(output)
        print(json.dumps(summary, indent=2))
    except (KeyError, OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
