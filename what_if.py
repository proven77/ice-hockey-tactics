import argparse
import csv
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
import requests

from tactical_view_converter import TacticalViewConverter


ROOT = Path(__file__).resolve().parent
RINK = ROOT / "images/nhl-hockey-rink.png"


def project(matrix, point):
    result = matrix @ np.array([*point, 1.0], dtype=float)
    if not np.isfinite(result).all() or abs(result[2]) < 1e-8:
        raise ValueError("Point cannot be projected through the rink calibration")
    return result[:2] / result[2]


def rink_frame(source, index):
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
            raise ValueError(f"Could not decode source frame {index}")
        return frame
    finally:
        capture.release()


def mask_from_polygons(shape, polygons):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("Each mask polygon needs at least three pixel coordinates")
        cv2.fillPoly(mask, [vertices], 255)
    return mask


def build_edit(frame, config, rink_size):
    image_to_ice = np.asarray(config["image_to_ice"], dtype=float)
    if image_to_ice.shape != (3, 3) or not np.isfinite(image_to_ice).all() or np.linalg.matrix_rank(image_to_ice) < 3:
        raise ValueError("Expected a valid 3 x 3 image-to-ice matrix")
    height, width = frame.shape[:2]
    source_px = np.asarray(config["source_foot_px"], dtype=float)
    target_m = np.asarray(config["target_m"], dtype=float)
    if source_px.shape != (2,) or not np.isfinite(source_px).all() or (source_px < 0).any() or (source_px >= [width, height]).any():
        raise ValueError("Source skate anchor is outside the frame")
    if target_m.shape != (2,) or not np.isfinite(target_m).all() or (target_m < 0).any() or (target_m > rink_size).any():
        raise ValueError("Target is outside the rink")

    source_m = project(image_to_ice, source_px)
    ice_to_image = np.linalg.inv(image_to_ice)
    target_px = project(ice_to_image, target_m)
    if (target_px < 0).any() or (target_px >= [width, height]).any():
        raise ValueError("The target skate anchor is outside the camera view")
    # The projective denominator gives a local scale approximation for a short move.
    scale = float((ice_to_image[2] @ [*source_m, 1.0]) / (ice_to_image[2] @ [*target_m, 1.0]))
    if not 0.75 <= scale <= 1.30:
        raise ValueError("Target requires a larger perspective change than this 2D edit supports")
    transform = np.column_stack((np.eye(2) * scale, target_px - scale * source_px))

    player = mask_from_polygons((height, width), config["player_polygons_px"])
    if not np.any(player):
        raise ValueError("Player mask is empty")
    x, y, w, h = cv2.boundingRect(player)
    corners = np.array([[x, y], [x + w - 1, y + h - 1]]) * scale + transform[:, 2]
    if (corners < 0).any() or (corners >= [width, height]).any():
        raise ValueError("The relocated player would be cut off by the frame")
    # Leave room for the feather entirely outside the player and stick silhouette.
    original = cv2.dilate(player, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    shadow = mask_from_polygons((height, width), config.get("shadow_polygons_px", []))
    original = cv2.bitwise_or(original, shadow)
    relocated = cv2.warpAffine(player, transform, (width, height), flags=cv2.INTER_LINEAR)
    protected = mask_from_polygons((height, width), config.get("protected_polygons_px", []))
    if np.any((protected > 0) & ((original > 0) | (relocated > 0))):
        raise ValueError("Edit intersects a protected object")
    return source_m, target_m, target_px, scale, transform, player, original, relocated


def contact_shadow(shape, foot_px, scale):
    mask = np.zeros(shape, dtype=np.uint8)
    center = tuple(np.rint(foot_px + np.array([2, 3]) * scale).astype(int))
    cv2.ellipse(mask, center, (round(20 * scale), round(5 * scale)), 10, 0, 360, 44, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 4 * scale)


def estimate_ice_color(crop, mask):
    # This case is on open ice. Fit a lighting gradient from the unedited perimeter.
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))) > mask
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    coordinates = np.stack((np.ones_like(xx), xx / mask.shape[1], yy / mask.shape[0]), axis=-1)
    coefficients = np.linalg.lstsq(coordinates[ring], crop[ring], rcond=None)[0]
    return (coordinates @ coefficients).clip(0, 255).astype(np.float32)


def comfy_repair(frame, mask, session, url, output, seed):
    x, y, w, h = cv2.boundingRect(mask)
    margin = 80
    left, top = max(0, x - margin), max(0, y - margin)
    right = min(frame.shape[1], x + w + margin)
    bottom = min(frame.shape[0], y + h + margin)
    region = np.s_[top:bottom, left:right]
    crop, crop_mask = frame[region], mask[region]
    # Seed the inpainting with the local ice tone, including where the shadow fell.
    ice_color = estimate_ice_color(crop, crop_mask)
    prefill = np.where(crop_mask[..., None] > 0, np.rint(ice_color), crop).astype(np.uint8)
    size = tuple(max(16, round(v * 512 / max(prefill.shape[:2]) / 16) * 16) for v in prefill.shape[1::-1])
    folder = f"hockey-what-if/{uuid4().hex}"
    uploaded = []
    for filename, image in (("ice.png", cv2.resize(prefill, size)),
                            ("mask.png", cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST))):
        cv2.imwrite(str(output / filename), image)
        _, encoded = cv2.imencode(".png", image)
        response = session.post(f"{url}/upload/image", files={"image": (filename, encoded.tobytes(), "image/png")},
                                data={"subfolder": folder, "type": "input"}, timeout=30)
        response.raise_for_status()
        result = response.json()
        uploaded.append(f"{result['subfolder']}/{result['name']}")

    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": uploaded[0]}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": uploaded[1], "channel": "red"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "Empty smooth white ice surface of a hockey rink from a television broadcast, subtle pale blue-grey lighting, matching the surrounding clean ice, empty ice, no objects or people."}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["8", 0], "mask": ["5", 0]}},
        "10": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["9", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 0.42}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": folder + "/repair"}},
    }
    (output / "comfy-workflow-api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    response = session.post(f"{url}/prompt", json={"prompt": graph}, timeout=30)
    if not response.ok:
        raise ValueError(f"ComfyUI rejected the graph: {response.text}")
    prompt_id = response.json()["prompt_id"]
    print(f"ComfyUI prompt: {prompt_id}", flush=True)
    deadline = monotonic() + 900
    while monotonic() < deadline:
        response = session.get(f"{url}/history/{prompt_id}", timeout=15)
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            if history["status"]["status_str"] != "success":
                (output / "comfy-error.json").write_text(json.dumps(history["status"], indent=2), encoding="utf-8")
                _, failure = history["status"]["messages"][-1]
                raise ValueError(f"ComfyUI {failure.get('node_type', 'job')} failed: {failure.get('exception_message', prompt_id)}")
            image = history["outputs"]["13"]["images"][0]
            response = session.get(f"{url}/view", params=image, timeout=30)
            response.raise_for_status()
            repaired = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if repaired is None:
                raise ValueError("Could not decode ComfyUI output")
            cv2.imwrite(str(output / "comfy-repair.png"), repaired)
            result = frame.copy()
            restored = cv2.resize(repaired, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
            texture = restored - cv2.GaussianBlur(restored, (0, 0), 8)
            restored = (ice_color + texture.clip(-8, 8)).clip(0, 255)
            feather = np.minimum(cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3) / 12, 1)[..., None]
            result[region] = np.rint(restored * feather + crop * (1 - feather)).astype(np.uint8)
            return result, prompt_id
        sleep(1)
    raise ValueError(f"ComfyUI prompt {prompt_id} did not finish within 15 minutes")


def composite(frame, background, transform, player, relocated, shadow):
    # Premultiplied alpha avoids carrying ice-coloured fringes to the new location.
    height, width = frame.shape[:2]
    foreground = frame.astype(np.float32) * (player[..., None] / 255)
    moved = cv2.warpAffine(foreground, transform, (width, height), flags=cv2.INTER_LINEAR)
    alpha = relocated[..., None] / 255
    background = background * (1 - shadow[..., None] / 255)
    return np.rint(moved + background * (1 - alpha)).clip(0, 255).astype(np.uint8)


def write_animation(path, config, rink_size, source_m, target_m):
    rink = cv2.resize(cv2.imread(str(RINK)), (900, 382), interpolation=cv2.INTER_AREA)
    palette = {0: (130, 130, 130), 1: (75, 126, 23), 2: (242, 244, 247)}
    with (ROOT / config["tracks_csv"]).open(newline="", encoding="utf-8") as stream:
        tracks = [row for row in csv.DictReader(stream)
                  if int(row["frame"]) == config["frame"] and row["x_m"] and row["y_m"]]

    def pixel(position):
        return tuple(np.rint(np.asarray(position) / rink_size * [899, 381] + [50, 84]).astype(int))

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1000, 550))
    if not writer.isOpened():
        raise ValueError("Could not open the tactical animation writer")
    try:
        for index in range(90):
            board = np.full((550, 1000, 3), (27, 24, 19), dtype=np.uint8)
            board[84:466, 50:950] = rink
            cv2.putText(board, "WHAT-IF  /  DEFENSIVE REPOSITIONING", (50, 54), 0, 1, (244, 239, 228), 2, cv2.LINE_AA)
            cv2.arrowedLine(board, pixel(source_m), pixel(target_m), (24, 112, 223), 2, cv2.LINE_AA, tipLength=0.19)
            for row in tracks:
                if int(row["track_id"]) == config["track_id"]:
                    continue
                point = pixel([float(row["x_m"]), float(row["y_m"])])
                cv2.circle(board, point, 11, (35, 37, 42), -1, cv2.LINE_AA)
                cv2.circle(board, point, 8, palette[int(row["team"])], -1, cv2.LINE_AA)
            fraction = min(1.0, max(0.0, (index - 12) / 55))
            progress = fraction * fraction * (3 - 2 * fraction)
            cv2.circle(board, pixel(target_m), 15, (24, 112, 223), 2, cv2.LINE_AA)
            cv2.circle(board, pixel(source_m + progress * (target_m - source_m)), 10, (24, 112, 223), -1, cv2.LINE_AA)
            text = (f"Track {config['track_id']}   |   {np.linalg.norm(target_m - source_m):.2f} m   |   "
                    f"Target ({target_m[0]:.2f}, {target_m[1]:.2f}) m   |   Illustrative path")
            cv2.putText(board, text, (50, 516), 0, 0.60, (227, 226, 218), 1, cv2.LINE_AA)
            writer.write(board)
    finally:
        writer.release()


def comparison(original, edited, index, output):
    panels = [cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA) for frame in (original, edited)]
    for label, panel in zip((f"ORIGINAL  /  FRAME {index}", "WHAT-IF  /  RELOCATED PLAYER"), panels):
        cv2.rectangle(panel, (0, 0), (800, 34), (27, 24, 19), -1)
        cv2.putText(panel, label, (16, 23), 0, 0.54, (245, 241, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), cv2.hconcat(panels))


def main():
    parser = argparse.ArgumentParser(description="Render a single hockey What-If edit and tactical trajectory.")
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8191")
    parser.add_argument("--seed", type=int, default=103)
    args = parser.parse_args()
    url = args.comfy_url.rstrip("/")
    endpoint = urlsplit(url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path or endpoint.query:
        parser.error("Use an HTTP URL for a local ComfyUI instance")

    try:
        config = json.loads(args.case.read_text(encoding="utf-8"))
        frame = rink_frame(ROOT / config["video"], config["frame"])
        rink_size = TacticalViewConverter(RINK).size_m
        source_m, target_m, target_px, scale, transform, player, repair, relocated = build_edit(frame, config, rink_size)
        shadow = contact_shadow(frame.shape[:2], target_px, scale)
        protected = mask_from_polygons(frame.shape[:2], config.get("protected_polygons_px", []))
        if np.any((shadow > 0) & (protected > 0)):
            raise ValueError("Contact shadow intersects a protected object")
        args.output.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(args.output / "original.png"), frame)
        allowed = (repair > 0) | (relocated > 0) | (shadow > 0)
        cv2.imwrite(str(args.output / "edit-mask.png"), allowed.astype(np.uint8) * 255)
        write_animation(args.output / "tactical-path.mp4", config, rink_size, source_m, target_m)
        with requests.Session() as session:
            session.trust_env = False
            started = monotonic()
            background, prompt_id = comfy_repair(frame, repair, session, url, args.output, args.seed)
        edited = composite(frame, background, transform, player, relocated, shadow)
        changed = np.any(edited != frame, axis=2)
        outside_changes = int(np.count_nonzero(changed & ~allowed))
        if outside_changes:
            raise ValueError("Pixels outside the edit mask changed")
        cv2.imwrite(str(args.output / "what-if.png"), edited)
        comparison(frame, edited, config["frame"], args.output / "comparison.png")
        actual_px = transform @ [*config["source_foot_px"], 1]
        error = float(np.linalg.norm(project(np.asarray(config["image_to_ice"]), actual_px) - target_m))
        report = {
            "source_video": config["video"], "source_frame": config["frame"], "track_id": config["track_id"],
            "calibration_inliers": config.get("calibration_inliers"), "source_m": source_m.tolist(),
            "target_m": target_m.tolist(), "target_foot_px": target_px.tolist(), "applied_foot_px": actual_px.tolist(),
            "anchor_roundtrip_error_m": error, "player_scale": scale, "outside_mask_changed_pixels": outside_changes,
            "changed_pixels": int(np.count_nonzero(changed)), "comfy_prompt_id": prompt_id,
            "comfy_seconds": round(monotonic() - started, 1), "seed": args.seed,
            "animation_fps": 30, "animation_frames": 90,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (KeyError, ValueError, OSError, requests.RequestException) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
