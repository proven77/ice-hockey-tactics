"""Run pose-conditioned D1 inpainting using the fixed experiment 5.1 inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from uuid import uuid4
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests

from prepare_player_redraw import Crop, json_file
from refine_player import read_image, upload, run_prompt, contact_sheet, paste_refinement
from what_if import ROOT, check_comfy_ready, repair_workflow, write_png


CONTROL_FILE = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"
CONTROL_SHA256 = "3ea098db9bd145be525c7e2366920b6d76c5ffd46b3d7aa8169bbc943fdaee35"
PROMPT = (
    "A realistic paused professional ice hockey television broadcast. One Toronto defender "
    "in the same white jersey with navy stripes and existing jersey markings, white helmet, "
    "navy gloves and shorts, white socks and black hockey skates. The defender bends his "
    "knees and leans his upper body forward into a low defensive stance matching the colored "
    "pose guide. His support skate stays planted on the ice at the specified ankle position. "
    "Two hands grip one hockey stick naturally, with the blade reaching forward. "
    "Keep the original player scale, uniform, equipment and broadcast camera perspective. "
    "Soft natural contact shadows on pale ice, matching the surrounding grain and motion blur."
)


def pose_workflow(image, mask, pose, prefix, seed, denoise, strength):
    graph = repair_workflow(image, mask, prefix, seed)
    graph["6"]["inputs"]["text"] = PROMPT
    graph["11"]["inputs"]["denoise"] = denoise
    graph["14"] = {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROL_FILE}}
    graph["15"] = {"class_type": "ZImageFunControlnet", "inputs": {
        "model": ["1", 0], "model_patch": ["14", 0], "vae": ["3", 0],
        "image": ["16", 0], "inpaint_image": ["4", 0], "mask": ["5", 0], "strength": strength,
    }}
    graph["16"] = {"class_type": "LoadImage", "inputs": {"image": pose}}
    graph["10"]["inputs"]["model"] = ["15", 0]
    return graph


def bundle_digest(folder):
    digest = sha256()
    for name in ("report.json", "opencv.png", "conditioning-image.png", "redraw-mask-full.png",
                 "pose-target.png", "pose-target.json", "stick-guide.png", "protected-mask-full.png"):
        digest.update(name.encode())
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def validate_pose_output(folder, baseline, edited, mask):
    original = read_image(folder / "original.png")
    background = read_image(folder / "repaired-background.png")
    original_repair = read_image(folder / "original-repair-mask.png", mask=True) > 0
    protected = read_image(folder / "protected-mask-full.png", mask=True) > 0
    changes_from_original = np.any(edited != original, axis=2)
    checks = {
        "outside_target_changed_pixels": int(np.count_nonzero(np.any(edited != background, axis=2) & (mask == 0))),
        "outside_all_edits_changed_pixels": int(np.count_nonzero(changes_from_original & ~original_repair & (mask == 0))),
        "protected_changed_pixels": int(np.count_nonzero(changes_from_original & protected)),
    }
    if any(checks.values()):
        raise ValueError(f"Pose paste verification failed: {checks}")
    checks["changed_from_opencv_pixels"] = int(np.count_nonzero(np.any(edited != baseline, axis=2)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "output-videos/whatif-defense-step5-1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--denoise", type=float, default=.65)
    parser.add_argument("--strength", type=float, default=.85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8191")
    args = parser.parse_args()
    try:
        endpoint = urlsplit(args.comfy_url)
        if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path or endpoint.query:
            raise ValueError("Expected a local ComfyUI HTTP URL")
        if not 0 < args.denoise <= 1 or not 0 < args.strength <= 2:
            raise ValueError("Denoise or control strength is out of range")
        output = args.output.resolve()
        if output.exists() and not args.resume:
            raise ValueError(f"Output exists: {output}. Choose a new path or use --resume")
        specification = json.loads((args.bundle / "verification.json").read_text(encoding="utf-8"))
        names = args.cases or [case["name"] for case in specification["cases"]]
        if set(names) - {case["name"] for case in specification["cases"]}:
            raise ValueError("Unknown bundle case")
        seeds = args.seeds or [103, 104, 105]
        settings = {
            "workflow_version": 1, "denoise": args.denoise, "control_strength": args.strength,
            "steps": 8, "cfg": 1.0, "prompt": PROMPT, "feather_px": 3,
            "control_model": CONTROL_FILE, "control_model_sha256": CONTROL_SHA256,
            "contact_policy": "generated_from_pose_without_pixel_lock",
        }
        records = []
        with requests.Session() as session:
            session.trust_env = False
            url = args.comfy_url.rstrip("/")
            check_comfy_ready(session, url)
            info = session.get(f"{url}/object_info", timeout=15)
            info.raise_for_status()
            nodes = info.json()
            if not {"ModelPatchLoader", "ZImageFunControlnet"} <= nodes.keys():
                raise ValueError("Required ComfyUI pose nodes are missing")
            if CONTROL_FILE not in nodes["ModelPatchLoader"]["input"]["required"]["name"][0]:
                raise ValueError(f"ComfyUI cannot find {CONTROL_FILE} in model_patches")
            output.mkdir(parents=True, exist_ok=True)
            environment = session.get(f"{url}/system_stats", timeout=10)
            environment.raise_for_status()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            json_file(output / f"environment-{stamp}.json", environment.json())
            for name in names:
                source_dir = args.bundle / name
                source = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
                x, y, w, h = source["crop"]["xywh"]
                if w != h:
                    raise ValueError("Expected a square conditioning crop")
                crop = Crop(x, y, w)
                baseline = read_image(source_dir / "opencv.png")
                conditioning = read_image(source_dir / "conditioning-image.png")
                mask = read_image(source_dir / "redraw-mask-full.png", mask=True)
                protected = read_image(source_dir / "protected-mask-full.png", mask=True) > 0
                if np.any((mask > 0) & protected):
                    raise ValueError("Redraw mask overlaps a protected object")
                if not np.array_equal(crop.sample(baseline), conditioning):
                    raise ValueError("Conditioning crop does not match the baseline")
                pose = read_image(source_dir / "pose-target.png")
                digest = bundle_digest(source_dir)
                inputs = None
                for seed in seeds:
                    destination = output / name / f"seed-{seed}"
                    identity = {**settings, "case": name, "seed": seed, "input_sha256": digest}
                    if destination.exists():
                        record = json.loads((destination / "report.json").read_text(encoding="utf-8"))
                        if record["identity"] != identity:
                            raise ValueError(f"Existing job identity mismatch: {destination}")
                        if any(not (destination / filename).is_file()
                               for filename in ("redrawn.png", "comfy-raw.png", "history.json")):
                            raise ValueError(f"Incomplete existing job: {destination}")
                        records.append(record)
                        print(f"Reuse {name} seed {seed}", flush=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=destination.parent, prefix=".pose-") as temporary:
                        staging = Path(temporary) / "result"
                        staging.mkdir()
                        if inputs is None:
                            prefix = "hockey-pose-redraw/" + uuid4().hex
                            inputs = (
                                upload(session, url, prefix, "input.png", conditioning),
                                upload(session, url, prefix, "mask.png", crop.sample(mask, mask=True)),
                                upload(session, url, prefix, "pose.png", pose), prefix,
                            )
                        image_input, mask_input, pose_input, prefix = inputs
                        graph = pose_workflow(image_input, mask_input, pose_input, prefix + f"/seed-{seed}",
                                              seed, args.denoise, args.strength)
                        print(f"{name} seed {seed}, denoise {args.denoise}, control {args.strength}", flush=True)
                        try:
                            generated, timing = run_prompt(session, url, graph, staging)
                        except (ValueError, requests.RequestException) as exc:
                            failure = output / "failures" / f"{name}-seed-{seed}-{uuid4().hex[:8]}"
                            failure.parent.mkdir(exist_ok=True)
                            staging.rename(failure)
                            raise ValueError(f"{exc}. Diagnostic files kept at {failure}") from exc
                        edited = paste_refinement(baseline, generated, mask, crop)
                        checks = validate_pose_output(source_dir, baseline, edited, mask)
                        difference = np.clip(cv2.absdiff(edited, baseline).astype(np.uint16) * 4, 0, 255).astype(np.uint8)
                        record = {
                            "identity": identity, "source_frame": source["source_frame"], "track_id": source["track_id"],
                            "target_m": source["target_m"], "requested_contact_frame_px": source["pose"]["contact_frame_px"],
                            "crop": source["crop"], "generation": timing, "verification": checks,
                            "quality_review": "pending_contact_and_pose_review",
                        }
                        for filename, image in {
                            "comfy-raw.png": generated, "redrawn.png": edited,
                            "difference-x4.png": difference,
                            "comparison.png": contact_sheet(baseline, edited, difference, crop,
                                                            result_label="Pose redraw", difference_scale=4),
                        }.items():
                            write_png(staging / filename, image)
                        json_file(staging / "report.json", record)
                        staging.rename(destination)
                        records.append(record)
                        json_file(output / "summary.json", {"phase": "5.3", "jobs": records})
                        print(f"Finished in {timing['request_seconds']:.1f}s", flush=True)
            json_file(output / "summary.json", {"phase": "5.3", "jobs": records})
    except (KeyError, OSError, ValueError, requests.RequestException, cv2.error) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
