import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, sleep
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
import requests

from tactical_animation import animation_info, write_animation
from tactical_view_converter import TacticalViewConverter


ROOT = Path(__file__).resolve().parent
RINK = ROOT / "images/nhl-hockey-rink.png"
RINK_CORNER_RADIUS_M = 4.4  # Rounded corners of the tactical board artwork.


class PlacementError(ValueError):
    def __init__(self, message, user_message):
        super().__init__(message)
        self.user_message = user_message


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


@dataclass
class PreparedPlayer:
    image_to_ice: np.ndarray
    ice_to_image: np.ndarray
    source_px: np.ndarray
    source_m: np.ndarray
    repair: np.ndarray
    protected: np.ndarray
    origin_px: np.ndarray
    alpha: np.ndarray
    body_alpha: np.ndarray
    foreground: np.ndarray


@dataclass
class Placement:
    target_m: np.ndarray
    target_px: np.ndarray
    scale: float
    applied_foot_px: np.ndarray
    transform: np.ndarray
    region: tuple
    alpha: np.ndarray
    shadow: np.ndarray
    clipped_player_pixels: int
    clipped_shadow_pixels: int


def prepare_player(frame, config):
    image_to_ice = np.asarray(config["image_to_ice"], dtype=float)
    if image_to_ice.shape != (3, 3) or not np.isfinite(image_to_ice).all() or np.linalg.matrix_rank(image_to_ice) < 3:
        raise ValueError("Expected a valid 3 x 3 image-to-ice matrix")
    height, width = frame.shape[:2]
    source_px = np.asarray(config["source_foot_px"], dtype=float)
    if source_px.shape != (2,) or not np.isfinite(source_px).all() or (source_px < 0).any() or (source_px >= [width, height]).any():
        raise ValueError("Source skate anchor is outside the frame")
    source_m = project(image_to_ice, source_px)
    ice_to_image = np.linalg.inv(image_to_ice)
    body = mask_from_polygons((height, width), config["player_polygons_px"])
    stick = mask_from_polygons((height, width), config.get("stick_polygons_px", []))
    player = cv2.bitwise_or(body, stick)
    if not np.any(player):
        raise ValueError("Player mask is empty")
    x, y, w, h = cv2.boundingRect(player)
    # Leave room for the feather entirely outside the player and stick silhouette.
    repair = cv2.dilate(player, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    shadow = mask_from_polygons((height, width), config.get("shadow_polygons_px", []))
    repair = cv2.bitwise_or(repair, shadow)
    protected = mask_from_polygons((height, width), config.get("protected_polygons_px", [])) > 0
    if np.any((repair > 0) & protected):
        raise ValueError("Source repair intersects a protected object")
    # Keep a one-pixel transparent border for bilinear sampling at the edges.
    left, top = max(0, x - 1), max(0, y - 1)
    crop = np.s_[top:y + h + 1, left:x + w + 1]
    alpha = player[crop].copy()
    foreground = frame[crop].astype(np.float32) * (alpha[..., None].astype(np.float32) / 255)
    return PreparedPlayer(image_to_ice, ice_to_image, source_px, source_m, repair, protected,
                          np.array([left, top]), alpha, body[crop].copy(), foreground)


def place_player(player, target, rink_size):
    height, width = player.repair.shape
    target_m = np.asarray(target, dtype=float)
    if target_m.shape != (2,) or not np.isfinite(target_m).all() or (target_m < 0).any() or (target_m > rink_size).any():
        raise PlacementError("Target is outside the rink", "請將 D1 放在冰場內。")
    radius = RINK_CORNER_RADIUS_M
    center = np.clip(target_m, radius, np.asarray(rink_size) - radius)
    if np.linalg.norm(target_m - center) > radius + 1e-8:
        raise PlacementError("Target is outside the rounded boards", "請將 D1 放在冰場圓角擋板內。")
    try:
        target_px = project(player.ice_to_image, target_m)
    except ValueError as exc:
        raise PlacementError("Target cannot be projected", "此落點無法投影到目前鏡頭，請移近原位。") from exc
    if (target_px < 0).any() or (target_px >= [width, height]).any():
        raise PlacementError("The target skate anchor is outside the camera view", "此落點在目前鏡頭外，請移近原位。")
    # The projective denominator gives a local scale approximation for a short move.
    depth = player.ice_to_image[2]
    scale = float((depth @ [*player.source_m, 1.0]) / (depth @ [*target_m, 1.0]))
    if not 0.75 <= scale <= 1.30:
        raise PlacementError("Target requires a larger perspective change than this 2D edit supports",
                             "此落點的透視變化太大，請移近原位。")
    offset = target_px - scale * player.source_px
    x, y, w, h = cv2.boundingRect(player.alpha)
    corners = (np.array([[x, y], [x + w - 1, y + h - 1]]) + player.origin_px) * scale + offset
    if (corners < 0).any() or (corners >= [width, height]).any():
        raise PlacementError("The relocated player would be cut off by the frame",
                             "D1 的輪廓會超出畫面，請往畫面內移。")

    # Cover the transformed cutout and the full support of the blurred skate shadow.
    start = np.maximum(0, np.floor(np.minimum(corners[0] - 2, target_px - 56)).astype(int))
    end = np.minimum([width, height], np.ceil(np.maximum(corners[1] + 3, target_px + 56)).astype(int))
    region = np.s_[start[1]:end[1], start[0]:end[0]]
    size = tuple(end - start)
    transform = np.column_stack((np.eye(2) * scale, scale * player.origin_px + offset - start))
    alpha = cv2.warpAffine(player.alpha, transform, size, flags=cv2.INTER_LINEAR)
    shadow = contact_shadow(size[::-1], target_px - start, scale)
    protected = player.protected[region]
    clipped_player = int(np.count_nonzero(protected & (alpha > 0)))
    clipped_shadow = int(np.count_nonzero(protected & (shadow > 0)))
    if clipped_player:
        body = cv2.warpAffine(player.body_alpha, transform, size, flags=cv2.INTER_LINEAR) >= 128
        stick = (alpha >= 128) & ~body
        # Let the original object cover a small part of the cutout, while keeping
        # the skates and the main silhouette visible.
        foot_x, foot_y = np.clip(np.rint(target_px - start).astype(int), 0, end - start - 1)
        if protected[foot_y, foot_x] or np.count_nonzero(protected & body) > max(4, np.count_nonzero(body) * .01):
            raise PlacementError("D1 body or skate anchor overlaps a protected object",
                                 "D1 的身體或冰刀與其他球員、冰球或球門重疊，請稍微移開。")
        if np.count_nonzero(protected & stick) > np.count_nonzero(stick) * .25:
            raise PlacementError("D1 stick substantially overlaps a protected object",
                                 "D1 的球桿會被遮住太多，請稍微移開。")
    alpha[protected] = 0
    shadow[protected] = 0
    applied_foot_px = transform @ [*(player.source_px - player.origin_px), 1] + start
    return Placement(target_m, target_px, scale, applied_foot_px, transform, region, alpha, shadow,
                     clipped_player, clipped_shadow)


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


def repair_reference(config, frame):
    reference = config.get("repair_reference")
    if reference is None:
        return None
    matrix = np.asarray(reference["donor_to_source"], dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        raise ValueError("Invalid repair reference alignment")
    donor = rink_frame(ROOT / config["video"], reference["frame"])
    aligned = cv2.warpAffine(donor, matrix, frame.shape[1::-1])
    region = mask_from_polygons(frame.shape[:2], reference["polygons_px"])
    valid = cv2.warpAffine(np.full(donor.shape[:2], 255, np.uint8), matrix, frame.shape[1::-1])
    if np.any((region > 0) & (valid != 255)):
        raise ValueError("Repair reference extends beyond the donor frame")
    weights = np.minimum(cv2.distanceTransform(region, cv2.DIST_L2, 3) / 10, 1)
    return aligned, weights


def comfy_repair(frame, mask, session, url, output, seed, reference=None):
    x, y, w, h = cv2.boundingRect(mask)
    margin = 80
    left, top = max(0, x - margin), max(0, y - margin)
    right = min(frame.shape[1], x + w + margin)
    bottom = min(frame.shape[0], y + h + margin)
    region = np.s_[top:bottom, left:right]
    crop, crop_mask = frame[region], mask[region]
    # Seed the inpainting with the local ice tone, including where the shadow fell.
    ice_color = estimate_ice_color(crop, crop_mask)
    if reference is not None:
        aligned, weights = reference
        weight = weights[region][..., None]
        ice_color = ice_color * (1 - weight) + aligned[region] * weight
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

    graph = repair_workflow(*uploaded, folder + "/repair", seed)
    (output / "comfy-workflow-api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    response = session.post(f"{url}/prompt", json={"prompt": graph}, timeout=30)
    if not response.ok:
        raise ValueError(f"ComfyUI rejected the graph: {response.text}")
    prompt_id = response.json()["prompt_id"]
    print(f"ComfyUI prompt: {prompt_id}", flush=True)
    deadline = monotonic() + 900
    try:
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
                if reference is not None:
                    texture *= 1 - weight
                restored = (ice_color + texture.clip(-8, 8)).clip(0, 255)
                feather = np.minimum(cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3) / 12, 1)[..., None]
                result[region] = np.rint(restored * feather + crop * (1 - feather)).astype(np.uint8)
                return result, prompt_id
            sleep(1)
    except requests.RequestException as exc:
        raise ValueError(
            f"ComfyUI connection was lost while prompt {prompt_id} was running. "
            "Check that ComfyUI is still running, then rerun the command. "
            f"Details: {exc}"
        ) from exc
    raise ValueError(f"ComfyUI prompt {prompt_id} did not finish within 15 minutes")


def check_comfy_ready(session, url):
    try:
        response = session.get(f"{url}/system_stats", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(
            f"Cannot connect to ComfyUI at {url}. Start ComfyUI, wait until its GUI/API is ready, "
            f"then rerun this command. Details: {exc}"
        ) from exc


def repair_workflow(ice_input, mask_input, output_prefix, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": ice_input}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": mask_input, "channel": "red"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "Empty smooth white ice surface of a hockey rink from a television broadcast, subtle pale blue-grey lighting, matching the surrounding clean ice, empty ice, no objects or people."}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["8", 0], "mask": ["5", 0]}},
        "10": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "11": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["9", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 0.42}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": output_prefix}},
    }


def composite(background, player, placement):
    moved = cv2.warpAffine(player.foreground, placement.transform, placement.alpha.shape[::-1], flags=cv2.INTER_LINEAR)
    moved[player.protected[placement.region]] = 0
    alpha = placement.alpha[..., None].astype(np.float32) / 255
    shadow = placement.shadow[..., None].astype(np.float32) / 255
    patch = moved + background[placement.region] * (1 - shadow) * (1 - alpha)
    edited = background.copy()
    edited[placement.region] = np.rint(patch).clip(0, 255).astype(np.uint8)
    return edited


def case_tracks(config, image_to_ice):
    if "other_players" in config:
        return [
            {"position": project(image_to_ice, track["foot_px"]).tolist(), "team": track["team"]}
            for track in config["other_players"] if track["track_id"] != config["track_id"]
        ]
    with (ROOT / config["tracks_csv"]).open(newline="", encoding="utf-8") as stream:
        return [
            {"position": [float(row["x_m"]), float(row["y_m"])], "team": int(row["team"])}
            for row in csv.DictReader(stream)
            if int(row["frame"]) == config["frame"] and int(row["track_id"]) != config["track_id"]
            and row["x_m"] and row["y_m"]
        ]


def tactical_report(config, rink_size, player, placement, defense):
    source = player.source_m.tolist()
    target = placement.target_m.tolist()
    return {
        "source_video": config["video"], "source_frame": config["frame"], "track_id": config["track_id"],
        "label": "D1", "source_m": source, "target_m": target,
        "defense": defense.measure(target), "animation": animation_info(rink_size, source, target),
    }


def comparison(original, edited, index, output):
    panels = [cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA) for frame in (original, edited)]
    for label, panel in zip((f"ORIGINAL  /  FRAME {index}", "WHAT-IF  /  RELOCATED PLAYER"), panels):
        cv2.rectangle(panel, (0, 0), (800, 34), (27, 24, 19), -1)
        cv2.putText(panel, label, (16, 23), 0, 0.54, (245, 241, 230), 1, cv2.LINE_AA)
    write_png(output, cv2.hconcat(panels))


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def placement_report(frame, edited, player, placement):
    allowed = player.repair > 0
    allowed[placement.region] |= (placement.alpha > 0) | (placement.shadow > 0)
    changed = np.any(edited != frame, axis=2)
    outside_changes = int(np.count_nonzero(changed & ~allowed))
    if outside_changes:
        raise ValueError("Pixels outside the edit mask changed")
    protected_changes = int(np.count_nonzero(changed & player.protected))
    if protected_changes:
        raise ValueError("Pixels inside the protected mask changed")
    actual_px = placement.applied_foot_px
    error = float(np.linalg.norm(project(player.image_to_ice, actual_px) - placement.target_m))
    return allowed, {
        "target_m": placement.target_m.tolist(), "target_foot_px": placement.target_px.tolist(),
        "applied_foot_px": actual_px.tolist(), "anchor_roundtrip_error_m": error,
        "distance_m": float(np.linalg.norm(placement.target_m - player.source_m)),
        "player_scale": placement.scale, "outside_mask_changed_pixels": outside_changes,
        "protected_changed_pixels": protected_changes, "clipped_player_pixels": placement.clipped_player_pixels,
        "clipped_shadow_pixels": placement.clipped_shadow_pixels,
        "changed_pixels": int(np.count_nonzero(changed)),
    }


def render_placement(frame, background, player, placement, config, rink_size, output, tracks, defense):
    started = perf_counter()
    edited = composite(background, player, placement)
    composite_ms = (perf_counter() - started) * 1000
    allowed, report = placement_report(frame, edited, player, placement)
    write_png(output / "edit-mask.png", allowed.astype(np.uint8) * 255)
    write_png(output / "what-if.png", edited)
    comparison(frame, edited, config["frame"], output / "comparison.png")
    report.update(tactical_report(config, rink_size, player, placement, defense))
    coverage_at = defense.lane.coverage if defense.lane is not None else None
    write_animation(output / "tactical-path.mp4", RINK, rink_size, tracks, report, coverage_at)
    return {**report, "composite_ms": round(composite_ms, 2),
            "animation_fps": report["animation"]["fps"], "animation_frames": report["animation"]["frames"]}


def main():
    from shot_lane import DefenseMeasurement

    parser = argparse.ArgumentParser(description="Render hockey What-If edits using one repair per selected player.")
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", nargs=2, type=float, action="append", metavar=("X_M", "Y_M"),
                        help="Override the case target in ice metres. Repeat to reuse the repair across targets.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8191")
    parser.add_argument("--seed", type=int, default=103)
    args = parser.parse_args()
    args.output = args.output.resolve()
    url = args.comfy_url.rstrip("/")
    endpoint = urlsplit(url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path or endpoint.query:
        parser.error("Use an HTTP URL for a local ComfyUI instance")

    staging = None
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}. Choose a new output path or move the previous run folder.")
        config = json.loads(args.case.read_text(encoding="utf-8"))
        frame = rink_frame(ROOT / config["video"], config["frame"])
        rink_size = TacticalViewConverter(RINK).size_m
        player = prepare_player(frame, config)
        tracks = case_tracks(config, player.image_to_ice)
        defense = DefenseMeasurement(config, player, rink_size, tracks)
        targets = args.target if args.target else [config["target_m"]]
        placements = []
        for index, target in enumerate(targets, 1):
            try:
                placements.append(place_player(player, target, rink_size))
            except ValueError as exc:
                raise ValueError(f"Target {index} {target}: {exc}") from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with requests.Session() as session:
            session.trust_env = False
            check_comfy_ready(session, url)
            staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
            write_png(staging / "original.png", frame)
            started = monotonic()
            background, prompt_id = comfy_repair(
                frame, player.repair, session, url, staging, args.seed, repair_reference(config, frame)
            )
        repair_seconds = round(monotonic() - started, 1)
        write_png(staging / "repaired-background.png", background)
        write_png(staging / "repair-mask.png", player.repair)
        report = {
            "source_video": config["video"], "source_frame": config["frame"], "track_id": config["track_id"],
            "calibration_inliers": config.get("calibration_inliers"), "source_m": player.source_m.tolist(),
            "comfy_prompt_id": prompt_id, "comfy_seconds": repair_seconds, "seed": args.seed,
        }
        if "repair_reference" in config:
            report["repair_reference"] = config["repair_reference"]
        if len(placements) == 1:
            report.update(render_placement(frame, background, player, placements[0], config, rink_size, staging,
                                           tracks, defense))
        else:
            report["placements"] = []
            for index, placement in enumerate(placements, 1):
                output = staging / f"target-{index:02d}"
                output.mkdir()
                result = render_placement(frame, background, player, placement, config, rink_size, output,
                                          tracks, defense)
                report["placements"].append({"directory": output.name, **result})
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(args.output)
        staging = None
        print(json.dumps(report, indent=2))
    except (KeyError, ValueError, OSError, requests.RequestException) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
