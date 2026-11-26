"""Run the experiment 5.2 seam-refinement comparison against local ComfyUI."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from prepare_player_redraw import CONDITION_SIZE, Crop, json_file
from what_if import ROOT, check_comfy_ready, repair_workflow, write_png, filesystem_path


PROMPT = (
    "A paused professional ice hockey television broadcast. One Toronto player in a white "
    "jersey with navy blue stripes, white helmet, navy gloves and shorts, white socks and "
    "black ice skates, holding the existing hockey stick. Preserve the exact same player, "
    "jersey markings, equipment, body pose, scale and skate position as the input. "
    "Natural soft contact shadow on the pale ice. Subtle broadcast blur and grain, "
    "smooth continuous edges matching the surrounding lighting."
)


def read_image(path, mask=False):
    content = filesystem_path(path).read_bytes()
    result = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR) if content else None
    if result is None:
        raise ValueError(f"Cannot read image: {path}")
    return result


def input_digest(folder):
    names = ("report.json", "opencv.png", "conditioning-image.png", "seam-mask-full.png",
             "original.png", "original-repair-mask.png", "protected-mask-full.png")
    digest = sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def refine_mask(folder, crop, report):
    mask = read_image(folder / "seam-mask-full.png", mask=True)
    protected = read_image(folder / "protected-mask-full.png", mask=True) > 0
    contact = tuple(np.rint(report["pose"]["contact_frame_px"]).astype(int))
    anchor_lock = np.zeros_like(mask)
    cv2.circle(anchor_lock, contact, 6, 255, -1)
    mask[protected | (anchor_lock > 0)] = 0
    if not np.any(mask):
        raise ValueError("Seam mask is empty")
    return mask, crop.sample(mask, mask=True), anchor_lock


def upload(session, url, folder, filename, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Cannot encode conditioning image")
    response = session.post(
        f"{url}/upload/image", files={"image": (filename, encoded.tobytes(), "image/png")},
        data={"subfolder": folder, "type": "input"}, timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    return f"{result['subfolder']}/{result['name']}"


def gpu_memory_mb():
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=True,
        )
        return int(response.stdout.splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def run_prompt(session, url, graph, directory):
    json_file(directory / "workflow-api.json", graph)
    started = monotonic()
    response = session.post(f"{url}/prompt", json={"prompt": graph}, timeout=30)
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]
    json_file(directory / "prompt.json", {"prompt_id": prompt_id})
    print(f"ComfyUI prompt {prompt_id}", flush=True)
    samples = []
    deadline = monotonic() + 900
    while monotonic() < deadline:
        try:
            response = session.get(f"{url}/history/{prompt_id}", timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"ComfyUI connection lost during prompt {prompt_id}. Restart ComfyUI then use --resume. {exc}") from exc
        history = response.json().get(prompt_id)
        used = gpu_memory_mb()
        if used is not None:
            samples.append(used)
        if history:
            json_file(directory / "history.json", history)
            if history["status"]["status_str"] != "success":
                raise ValueError(f"ComfyUI prompt {prompt_id} failed. Inspect history.json")
            saved = history["outputs"]["13"]["images"][0]
            response = session.get(f"{url}/view", params=saved, timeout=30)
            response.raise_for_status()
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (CONDITION_SIZE, CONDITION_SIZE, 3):
                raise ValueError("Expected a generated 512 x 512 image")
            elapsed = monotonic() - started
            return image, {
                "prompt_id": prompt_id, "request_seconds": round(elapsed, 3),
                "gpu_total_used_peak_mb": max(samples) if samples else None,
                "gpu_memory_samples": len(samples), "gpu_memory_scope": "whole_device_including_desktop",
            }
        sleep(2)
    raise ValueError(f"ComfyUI prompt {prompt_id} exceeded 15 minutes")


def paste_refinement(baseline, generated, full_mask, crop):
    restored = crop.restore(generated).astype(np.float32)
    weight = np.minimum(cv2.distanceTransform(full_mask, cv2.DIST_L2, 3) / 3, 1)[crop.region][..., None]
    result = baseline.copy()
    result[crop.region] = np.rint(
        restored * weight + baseline[crop.region].astype(np.float32) * (1 - weight)
    ).clip(0, 255).astype(np.uint8)
    return result


def verify_output(folder, baseline, edited, mask, anchor_lock):
    original = read_image(folder / "original.png")
    original_repair = read_image(folder / "original-repair-mask.png", mask=True) > 0
    baseline_edit = read_image(folder / "baseline-edit-mask.png", mask=True) > 0
    protected = read_image(folder / "protected-mask-full.png", mask=True) > 0
    changed = np.any(edited != baseline, axis=2)
    compared_to_original = np.any(edited != original, axis=2)
    checks = {
        "outside_seam_changed_pixels": int(np.count_nonzero(changed & (mask == 0))),
        "outside_all_edits_changed_pixels": int(np.count_nonzero(
            compared_to_original & ~(original_repair | baseline_edit | (mask > 0)))),
        "protected_changed_pixels": int(np.count_nonzero(compared_to_original & protected)),
        "anchor_lock_changed_pixels": int(np.count_nonzero(changed & (anchor_lock > 0))),
    }
    if any(checks.values()):
        raise ValueError(f"Output invariant failed: {checks}")
    checks["changed_pixels"] = int(np.count_nonzero(changed))
    checks["mean_abs_change_on_seam"] = round(float(np.abs(
        edited.astype(float) - baseline.astype(float))[mask > 0].mean()), 4)
    return checks


def contact_sheet(baseline, edited, difference, crop, *, result_label="ComfyUI refinement", difference_scale=8):
    panels = []
    for image, title in ((baseline[crop.region], "OpenCV"), (edited[crop.region], result_label),
                         (difference[crop.region], f"Difference x {difference_scale}")):
        panel = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
        panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(panel, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(panel)
    return cv2.hconcat(panels)


def main():
    parser = argparse.ArgumentParser(description="Compare fixed seeds for local masked seam refinement.")
    parser.add_argument("--bundle", type=Path, default=ROOT / "output-videos/whatif-defense-step5-1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases", help="Choose a bundle case name. Repeat as needed.")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--denoise", type=float, default=.22)
    parser.add_argument("--resume", action="store_true", help="Reuse completed jobs with identical inputs and settings")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8191")
    args = parser.parse_args()
    try:
        endpoint = urlsplit(args.comfy_url)
        if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path or endpoint.query:
            raise ValueError("Use an HTTP URL for a local ComfyUI instance")
        if not 0 < args.denoise <= 1:
            raise ValueError("Denoise must be in (0, 1]")
        output = args.output.resolve()
        if output.exists() and not args.resume:
            raise ValueError(f"Output exists. Choose another path or use --resume: {output}")
        specification = json.loads((args.bundle / "verification.json").read_text(encoding="utf-8"))
        available = {case["name"] for case in specification["cases"]}
        names = args.cases or [case["name"] for case in specification["cases"]]
        if set(names) - available:
            raise ValueError(f"Unknown input case: {set(names) - available}")
        seeds = args.seeds or [103, 104, 105]
        settings = {"workflow_version": 1, "denoise": args.denoise, "steps": 8, "cfg": 1.0,
                    "prompt": PROMPT, "anchor_lock_radius_px": 6, "feather_px": 3}
        records = []
        with requests.Session() as session:
            session.trust_env = False
            url = args.comfy_url.rstrip("/")
            check_comfy_ready(session, url)
            output.mkdir(parents=True, exist_ok=True)
            stats = session.get(f"{url}/system_stats", timeout=10)
            stats.raise_for_status()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            json_file(output / f"environment-{stamp}.json", stats.json())
            for name in names:
                folder = args.bundle / name
                source = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                baseline = read_image(folder / "opencv.png")
                crop_x, crop_y, crop_w, crop_h = source["crop"]["xywh"]
                if crop_w != crop_h:
                    raise ValueError("Expected square crop")
                crop = Crop(crop_x, crop_y, crop_w)
                full_mask, conditioning_mask, anchor_lock = refine_mask(folder, crop, source)
                conditioning = read_image(folder / "conditioning-image.png")
                if not np.array_equal(conditioning, crop.sample(baseline)):
                    raise ValueError("Conditioning image does not match the original baseline crop")
                digest = input_digest(folder)
                input_files = None
                for seed in seeds:
                    target = output / name / f"seed-{seed}"
                    identity = {**settings, "case": name, "input_sha256": digest, "seed": seed}
                    if target.exists():
                        saved = json.loads((target / "report.json").read_text(encoding="utf-8"))
                        if saved["identity"] != identity:
                            raise ValueError(f"Existing job inputs do not match: {target}")
                        if any(not (target / filename).is_file()
                               for filename in ("comfy-raw.png", "refined.png", "workflow-api.json")):
                            raise ValueError(f"Existing job is incomplete: {target}")
                        records.append(saved)
                        print(f"Reuse {name} seed {seed}", flush=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=target.parent, prefix=".refine-") as temp:
                        staging = Path(temp) / "result"
                        staging.mkdir()
                        if input_files is None:
                            upload_dir = "hockey-seam-refine/" + uuid4().hex
                            input_files = (
                                upload(session, url, upload_dir, "input.png", conditioning),
                                upload(session, url, upload_dir, "mask.png", conditioning_mask),
                                upload_dir,
                            )
                        image_input, mask_input, prefix = input_files
                        graph = repair_workflow(image_input, mask_input, prefix + f"/seed-{seed}", seed)
                        graph["6"]["inputs"]["text"] = PROMPT
                        graph["11"]["inputs"]["denoise"] = args.denoise
                        print(f"{name} seed {seed} denoise {args.denoise}", flush=True)
                        generated, timing = run_prompt(session, url, graph, staging)
                        post_started = monotonic()
                        edited = paste_refinement(baseline, generated, full_mask, crop)
                        checks = verify_output(folder, baseline, edited, full_mask, anchor_lock)
                        checks["raw_crop_outside_seam_changed_pixels"] = int(np.count_nonzero(
                            np.any(generated != conditioning, axis=2) & (conditioning_mask == 0)))
                        difference = np.clip(cv2.absdiff(edited, baseline).astype(np.uint16) * 8, 0, 255).astype(np.uint8)
                        record = {
                            "identity": identity, "source_frame": source["source_frame"], "track_id": source["track_id"],
                            "target_m": source["target_m"], "contact_frame_px": source["pose"]["contact_frame_px"],
                            "contact_policy": "preserve_baseline_6px_disk", "generation": timing, "verification": checks,
                            "quality_review": "pending_visual_review",
                        }
                        for filename, image in {
                            "comfy-raw.png": generated, "refined.png": edited, "refine-mask-full.png": full_mask,
                            "refine-mask.png": conditioning_mask, "anchor-lock.png": anchor_lock,
                            "difference-x8.png": difference, "comparison.png": contact_sheet(baseline, edited, difference, crop),
                        }.items():
                            write_png(staging / filename, image)
                        record["postprocess_seconds"] = round(monotonic() - post_started, 3)
                        json_file(staging / "report.json", record)
                        staging.rename(target)
                        records.append(record)
                        json_file(output / "summary.json", {"phase": "5.2", "jobs": records})
                        print(f"Finished in {timing['request_seconds']:.1f}s; changed {checks['changed_pixels']} pixels", flush=True)
            json_file(output / "summary.json", {"phase": "5.2", "jobs": records})
    except (KeyError, OSError, ValueError, requests.RequestException, cv2.error) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
