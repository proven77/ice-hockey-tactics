import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import cv2
import numpy as np
import requests

from shot_lane import DefenseMeasurement
from what_if import (RINK, RINK_CORNER_RADIUS_M, ROOT, PlacementError, TacticalViewConverter, case_tracks, check_comfy_ready, composite, comfy_repair,
                     place_player, placement_report, prepare_player, repair_reference, repair_workflow,
                     rink_frame)


def png(image):
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError("無法編碼 PNG")
    return encoded.tobytes()


class Review:
    def __init__(self, case, repair_run, comfy_port):
        self.case = case
        self.comfy_url = f"http://127.0.0.1:{comfy_port}"
        source = ROOT / case["video"]
        self.frame = rink_frame(source, case["frame"])
        self.reference = rink_frame(source, case["frame"] - 1)
        self.player = prepare_player(self.frame, case)
        self.rink_size = TacticalViewConverter(RINK).size_m
        self.background = self.cached_background(repair_run) if repair_run else None
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        tracks = case_tracks(case, self.player.image_to_ice)
        self.defense = DefenseMeasurement(case, self.player, self.rink_size, tracks)
        self.metadata = {
            "frame": case["frame"], "reference_frame": case["frame"] - 1, "rink_size": self.rink_size,
            "image_size": self.frame.shape[1::-1], "source_m": self.player.source_m.tolist(),
            "source_px": self.player.source_px.tolist(), "ice_to_image": self.player.ice_to_image.tolist(),
            "tracks": tracks, "defense": self.defense.measure(self.player.source_m),
            "rink_corner_radius_m": RINK_CORNER_RADIUS_M,
        }

    def cached_background(self, path):
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        graph = json.loads((path / "comfy-workflow-api.json").read_text(encoding="utf-8"))
        expected = repair_workflow(graph["4"]["inputs"]["image"], graph["5"]["inputs"]["image"],
                                   graph["13"]["inputs"]["filename_prefix"], 103)
        if (report["source_video"], report["source_frame"], report["track_id"], report["seed"]) != (
            self.case["video"], self.case["frame"], self.case["track_id"], 103
        ) or graph != expected or report.get("repair_reference") != self.case.get("repair_reference"):
            raise ValueError("修補快取的案例、seed、參考幀或工作流不符")
        original = cv2.imread(str(path / "original.png"))
        mask = cv2.imread(str(path / "repair-mask.png"), cv2.IMREAD_GRAYSCALE)
        background = cv2.imread(str(path / "repaired-background.png"))
        if not np.array_equal(original, self.frame) or not np.array_equal(mask, self.player.repair):
            raise ValueError("修補快取的原圖或遮罩不符")
        if background is None or background.shape != self.frame.shape:
            raise ValueError("修補快取缺少背景或尺寸不符")
        if np.any(background[mask == 0] != self.frame[mask == 0]):
            raise ValueError("修補快取在遮罩外有變更")
        return background, report["comfy_prompt_id"]

    def repair(self):
        cache_root = ROOT / "all-runs"
        cache_root.mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="defense-review-", dir=cache_root))
        with requests.Session() as session:
            session.trust_env = False
            check_comfy_ready(session, self.comfy_url)
            reference = repair_reference(self.case, self.frame)
            return comfy_repair(self.frame, self.player.repair, session, self.comfy_url, output, 103, reference)


def handler(review):
    web = ROOT / "defense_review"
    assets = {"/": ("text/html; charset=utf-8", (web / "index.html").read_bytes()),
              "/review.css": ("text/css; charset=utf-8", (web / "review.css").read_bytes()),
              "/review.js": ("text/javascript; charset=utf-8", (web / "review.js").read_bytes()),
              "/rink.png": ("image/png", RINK.read_bytes()),
              "/original.png": ("image/png", png(review.frame)),
              "/reference.png": ("image/png", png(review.reference)),
              "/case": ("application/json", json.dumps(review.metadata).encode())}

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, status, content_type, data, report=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if report is not None:
                self.send_header("X-Placement", json.dumps(report, separators=(",", ":")))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, status, value):
            self.send_data(status, "application/json", json.dumps(value).encode())

        def do_GET(self):
            asset = assets.get(self.path)
            if asset:
                self.send_data(200, *asset)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "需要 JSON 請求"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2048:
                    raise ValueError("請求大小不符")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("需要 JSON 物件")
                if self.path == "/prepare":
                    self.prepare(payload.get("retry") is True)
                elif self.path == "/measure":
                    placement = place_player(review.player, payload["target"], review.rink_size)
                    self.send_json(200, review.defense.measure(placement.target_m))
                elif self.path == "/render":
                    self.render(payload["target"])
                else:
                    self.send_error(404)
            except PlacementError as exc:
                self.send_json(400, {"error": exc.user_message, "invalid_placement": True})
            except (KeyError, TypeError, ValueError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OSError:
                self.send_json(500, {"error": "無法處理影像"})

        def prepare(self, retry):
            with self.server.prepare_lock:
                if review.background is not None:
                    self.send_json(200, {"state": "ready"})
                    return
                if review.pending is None or (retry and review.pending.done()):
                    review.pending = review.worker.submit(review.repair)
                if not review.pending.done():
                    self.send_json(202, {"state": "pending"})
                    return
                try:
                    review.background = review.pending.result()
                except (KeyError, OSError, ValueError, requests.RequestException) as exc:
                    self.send_json(503, {"error": f"修補失敗。{exc}"})
                    return
                self.send_json(200, {"state": "ready"})

        def render(self, target):
            started = perf_counter()
            placement = place_player(review.player, target, review.rink_size)
            prepared = review.background
            if prepared is None:
                self.send_json(409, {"error": "修補背景尚未就緒"})
                return
            background, prompt_id = prepared
            edited = composite(background, review.player, placement)
            _, report = placement_report(review.frame, edited, review.player, placement)
            report.update(source_frame=review.case["frame"], track_id=review.case["track_id"],
                          label="D1", source_m=review.player.source_m.tolist(), comfy_prompt_id=prompt_id,
                          defense=review.defense.measure(placement.target_m))
            data = png(edited)
            report["server_ms"] = round((perf_counter() - started) * 1000, 2)
            self.send_data(200, "image/png", data, report)

    return Handler


def main():
    from threading import Lock

    parser = argparse.ArgumentParser(description="Serve a local defensive What-If review.")
    parser.add_argument("--case", type=Path, default=ROOT / "what_if_case.json")
    parser.add_argument("--repair-run", type=Path, help="Reuse a verified What-If experiment output")
    parser.add_argument("--measure-only", action="store_true", help="Print geometry and exit without generating an image")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--comfy-port", type=int, default=8191)
    args = parser.parse_args()
    try:
        case = json.loads(args.case.read_text(encoding="utf-8"))
        if case["frame"] < 1 or not 1 <= args.comfy_port <= 65535:
            raise ValueError("案例需要前一幀，且 ComfyUI 連接埠必須有效")
        review = Review(case, args.repair_run, args.comfy_port)
        if args.measure_only:
            suggestion = review.defense.metadata.get("suggestion", {})
            target = suggestion.get("target_m", review.player.source_m)
            print(json.dumps(review.defense.measure(target), indent=2))
            return
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler(review)) as server:
            server.prepare_lock = Lock()
            print(f"http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
