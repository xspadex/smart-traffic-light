"""
Detection adapter for Demo/demo.html.

Run:
    python3 代码/detection_server.py

Then open:
    http://127.0.0.1:5001/Demo/demo.html?detect=local
"""
import argparse
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "代码" / "yolov8n.pt"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5001

CLASS_MAP = {
    2: "car",
    5: "bus",
    7: "truck",
}


def decode_data_url(data_url):
    if not data_url:
        raise ValueError("missing image")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid image")
    return frame


def get_mem_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0


def center_in_roi(det, roi):
    cx = det["x"] + det["w"] / 2
    cy = det["y"] + det["h"] / 2
    return roi["x"] <= cx <= roi["x"] + roi["w"] and roi["y"] <= cy <= roi["y"] + roi["h"]


def roi_to_pixels(roi, frame_w, frame_h):
    x1 = int(max(0, min(frame_w - 1, roi["x"] / 100 * frame_w)))
    y1 = int(max(0, min(frame_h - 1, roi["y"] / 100 * frame_h)))
    x2 = int(max(0, min(frame_w, (roi["x"] + roi["w"]) / 100 * frame_w)))
    y2 = int(max(0, min(frame_h, (roi["y"] + roi["h"]) / 100 * frame_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class LocalYoloProvider:
    def __init__(self, model_path):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def detect(self, frame, rois=None, conf=0.25):
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        detections = []
        for roi in rois or []:
            bounds = roi_to_pixels(roi, w, h)
            if not bounds:
                continue

            rx1, ry1, rx2, ry2 = bounds
            crop = frame[ry1:ry2, rx1:rx2]
            crop_h, crop_w = crop.shape[:2]
            results = self.model(crop, conf=conf, classes=list(CLASS_MAP.keys()), verbose=False)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in CLASS_MAP:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                    abs_x1 = rx1 + max(0, min(crop_w, x1))
                    abs_y1 = ry1 + max(0, min(crop_h, y1))
                    abs_x2 = rx1 + max(0, min(crop_w, x2))
                    abs_y2 = ry1 + max(0, min(crop_h, y2))
                    detections.append({
                        "type": CLASS_MAP[cls_id],
                        "conf": float(box.conf[0]),
                        "x": max(0, abs_x1 / w * 100),
                        "y": max(0, abs_y1 / h * 100),
                        "w": max(0, (abs_x2 - abs_x1) / w * 100),
                        "h": max(0, (abs_y2 - abs_y1) / h * 100),
                        "roiId": roi["id"],
                    })

        infer_ms = (time.perf_counter() - t0) * 1000

        return {
            "detections": detections,
            "stats": {
                "provider": "local",
                "model": self.model_path.name,
                "infer_ms": round(infer_ms, 1),
                "mem_mb": round(get_mem_mb(), 1),
            },
        }


def create_app(provider):
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "provider": "local", "model": provider.model_path.name})

    @app.route("/api/detections", methods=["POST", "OPTIONS"])
    def detections():
        if request.method == "OPTIONS":
            return ("", 204)
        payload = request.get_json(force=True)
        rois = payload.get("rois") or []
        if not rois:
            return jsonify({
                "detections": [],
                "stats": {
                    "provider": "local",
                    "model": provider.model_path.name,
                    "infer_ms": 0,
                    "mem_mb": round(get_mem_mb(), 1),
                },
            })
        frame = decode_data_url(payload.get("image"))
        conf = float(payload.get("conf") or 0.25)
        return jsonify(provider.detect(frame, rois=rois, conf=conf))

    @app.get("/")
    def index():
        return send_from_directory(ROOT / "Demo", "demo.html")

    @app.get("/<path:path>")
    def files(path):
        return send_from_directory(ROOT, path)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    provider = LocalYoloProvider(args.model)
    app = create_app(provider)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
