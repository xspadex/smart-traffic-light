"""
traffic_demo — WSL 端主程序

架构:
  - 视频播放线程: 12fps 流畅播放，叠加最新检测结果
  - 推理线程: 本地GPU 或 板端NPU 推理，更新检测结果 + FSM + GPIO
  - Flask + SocketIO: 推送帧到浏览器

用法:
    python3 traffic_demo.py --local                  # 本地GPU推理 (无需板子)
    python3 traffic_demo.py --local --board localhost:8765  # 本地推理 + 板子GPIO控灯
    python3 traffic_demo.py --board localhost:8765   # 板端推理 (默认)
    python3 traffic_demo.py --no-board               # 离线模式 (无推理)
"""
import argparse
import base64
import logging
import os
import socket
import struct
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np

# ============= Configuration =============

# Original ROI on 4096x2160
ROI_ORIG = [275, 103, 1298, 649]
ORIG_W, ORIG_H = 4096, 2160
# Display resolution
DISPLAY_W, DISPLAY_H = 1920, 1080
INPUT_SIZE = 640
VIDEO_PATH = os.path.expanduser("~/traffic/traffic.mp4")
BOARD_HOST = "localhost"
BOARD_PORT = 8765

# Scale ROI to display resolution
SCALE_X = DISPLAY_W / ORIG_W
SCALE_Y = DISPLAY_H / ORIG_H
ROI = [int(ROI_ORIG[0] * SCALE_X), int(ROI_ORIG[1] * SCALE_Y),
       int(ROI_ORIG[2] * SCALE_X), int(ROI_ORIG[3] * SCALE_Y)]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("demo")


# ============= Board TCP Client =============

class BoardClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            log.info(f"已连接板端: {self.host}:{self.port}")
            return True
        except Exception as e:
            log.error(f"连接板端失败: {e}")
            self.sock = None
            return False

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def ping(self):
        with self.lock:
            try:
                self.sock.sendall(b"PING\n")
                return self._recv_line() == "PONG"
            except Exception:
                return False

    def send_gpio(self, r, y, g):
        with self.lock:
            try:
                self.sock.sendall(f"GPIO {r} {y} {g}\n".encode())
                return self._recv_line() == "OK"
            except Exception as e:
                log.error(f"GPIO 命令失败: {e}")
                return False

    def infer(self, raw_data):
        with self.lock:
            try:
                self.sock.sendall(f"INFER {len(raw_data)}\n".encode())
                self.sock.sendall(raw_data)
                result = {}
                detections = []
                while True:
                    line = self._recv_line()
                    if line is None or line == "END":
                        break
                    if line.startswith("RESULT|"):
                        for part in line.split("|")[1:]:
                            k, v = part.split("=", 1)
                            result[k] = v
                    elif line.startswith("DET|"):
                        parts = line.split("|")
                        if len(parts) >= 7:
                            detections.append({
                                "class": parts[1],
                                "conf": float(parts[2]),
                                "x1": float(parts[3]),
                                "y1": float(parts[4]),
                                "x2": float(parts[5]),
                                "y2": float(parts[6]),
                            })
                    elif line.startswith("ERROR|"):
                        log.error(f"板端错误: {line}")
                        break
                return result, detections
            except Exception as e:
                log.error(f"推理请求失败: {e}")
                return None, []

    def _recv_line(self):
        buf = b""
        while True:
            c = self.sock.recv(1)
            if not c:
                return None
            if c == b"\n":
                return buf.decode("utf-8", errors="replace").strip()
            buf += c


# ============= Local GPU Inference =============

class LocalInfer:
    """Local YOLOv8 inference using ultralytics (GPU/CPU)."""

    VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, model_path="yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        log.info(f"本地模型加载完成: {model_path}")

    def infer(self, frame, roi=None):
        """Run inference on a frame. Returns (result_dict, detections_list).
        frame: BGR numpy array (display resolution)
        roi: [x1, y1, x2, y2] crop region on display frame
        """
        import psutil
        t0 = time.time()

        # Crop ROI if specified
        if roi:
            x1, y1, x2, y2 = roi
            img = frame[y1:y2, x1:x2]
        else:
            img = frame

        results = self.model(img, verbose=False)
        infer_ms = (time.time() - t0) * 1000

        detections = []
        vehicle_count = 0
        person_count = 0
        h_img, w_img = img.shape[:2]

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()

                # Normalize to 640x640 space (for compatibility with annotate_frame)
                det = {
                    "class": self.model.names[cls_id],
                    "conf": conf,
                    "x1": float(bx1 / w_img * 640),
                    "y1": float(by1 / h_img * 640),
                    "x2": float(bx2 / w_img * 640),
                    "y2": float(by2 / h_img * 640),
                }
                detections.append(det)

                if cls_id in self.VEHICLE_CLASSES:
                    vehicle_count += 1
                if cls_id == 0:
                    person_count += 1

        proc = psutil.Process()
        mem_kb = proc.memory_info().rss // 1024

        result = {
            "vehicles": str(vehicle_count),
            "persons": str(person_count),
            "infer_ms": str(infer_ms),
            "mem_rss_kb": str(mem_kb),
        }
        return result, detections


# ============= HdcGpio: GPIO control via hdc shell =============

class HdcGpioClient:
    """Control board GPIO via hdc shell commands (for --local mode)."""

    def __init__(self, hdc_path, pins, active_low=False):
        self.hdc = hdc_path
        self.pins = pins  # {"red": N, "yellow": N, "green": N}
        self.active_low = active_low
        self.initialized = False

    def init(self):
        """Export and configure GPIO pins."""
        import subprocess
        for name, pin in self.pins.items():
            if pin < 0:
                continue
            try:
                cmd = (f'{self.hdc} shell "'
                       f'echo {pin} > /sys/class/gpio/export 2>/dev/null; '
                       f'echo out > /sys/class/gpio/gpio{pin}/direction"')
                subprocess.run(cmd, shell=True, timeout=5, capture_output=True)
            except Exception as e:
                log.error(f"GPIO init pin {pin} ({name}) 失败: {e}")
                return False
        self.initialized = True
        log.info(f"HdcGpio 初始化完成: {self.pins}")
        return True

    def set(self, r, y, g):
        """Set GPIO values. r/y/g are 0 or 1."""
        if not self.initialized:
            return
        import subprocess
        if self.active_low:
            r, y, g = 1 - r, 1 - y, 1 - g
        vals = {"red": r, "yellow": y, "green": g}
        parts = []
        for name, val in vals.items():
            pin = self.pins.get(name, -1)
            if pin >= 0:
                parts.append(f"echo {val} > /sys/class/gpio/gpio{pin}/value")
        if parts:
            cmd = f'{self.hdc} shell "{"; ".join(parts)}"'
            try:
                subprocess.run(cmd, shell=True, timeout=3, capture_output=True)
            except Exception:
                pass


# ============= Modbus RTU: BSM relay control via RS232 =============

class ModbusLightClient:
    """Control BSM relay via TCP bridge (bsm_bridge.py on Windows)."""

    def __init__(self, host="localhost", port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()

    def init(self):
        """Connect to BSM bridge TCP server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(2)
            log.info(f"Modbus Bridge 已连接: {self.host}:{self.port}")
            self.set(0, 0, 0)
            return True
        except Exception as e:
            log.error(f"Modbus Bridge 连接失败: {e}")
            self.sock = None
            return False

    def set(self, r, y, g):
        """Set traffic light: r/y/g are 0 or 1."""
        if not self.sock:
            return
        with self.lock:
            try:
                self.sock.sendall(f"GPIO {r} {y} {g}\n".encode())
                self.sock.recv(64)
            except Exception as e:
                log.error(f"Modbus Bridge 发送失败: {e}")

    def close(self):
        """Turn off all and close connection."""
        if self.sock:
            try:
                self.set(0, 0, 0)
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# ============= Signal Controller (FSM) =============

class LightState(Enum):
    RED = auto()
    GREEN = auto()
    YELLOW = auto()


class SignalController:
    YELLOW_TIME = 3.0
    MIN_RED_TIME = 5.0

    def __init__(self):
        self.state = LightState.RED
        self.state_start = time.time()
        self.green_duration = 0
        self.vehicle_count = 0
        self.countdown = 0

    def get_green_duration(self, count):
        if count == 0:
            return 0
        elif count <= 10:
            return 15
        elif count <= 15:
            return 30
        else:
            return 45

    def update(self, vehicle_count):
        self.vehicle_count = vehicle_count
        now = time.time()
        elapsed = now - self.state_start

        if self.state == LightState.RED:
            # Minimum red time before allowing green
            if elapsed < self.MIN_RED_TIME:
                self.countdown = max(0, self.MIN_RED_TIME - elapsed)
                return self.state, 1, 0, 0
            dur = self.get_green_duration(vehicle_count)
            if dur > 0:
                self.state = LightState.GREEN
                self.green_duration = dur
                self.state_start = now
                self.countdown = dur
                log.info(f"RED → GREEN ({dur}s), 车辆数={vehicle_count}")
            else:
                self.countdown = 0
        elif self.state == LightState.GREEN:
            remaining = self.green_duration - elapsed
            self.countdown = max(0, remaining)
            if remaining <= 0:
                self.state = LightState.YELLOW
                self.state_start = now
                self.countdown = self.YELLOW_TIME
                log.info("GREEN → YELLOW")
        elif self.state == LightState.YELLOW:
            remaining = self.YELLOW_TIME - elapsed
            self.countdown = max(0, remaining)
            if remaining <= 0:
                self.state = LightState.RED
                self.state_start = now
                self.countdown = 0
                log.info("YELLOW → RED")

        gpio = {
            LightState.RED:    (1, 0, 0),
            LightState.GREEN:  (0, 0, 1),
            LightState.YELLOW: (0, 1, 0),
        }
        r, y, g = gpio[self.state]
        return self.state, r, y, g

    def tick(self):
        """Update countdown without changing vehicle count (for display thread)."""
        now = time.time()
        elapsed = now - self.state_start
        if self.state == LightState.GREEN:
            self.countdown = max(0, self.green_duration - elapsed)
        elif self.state == LightState.YELLOW:
            self.countdown = max(0, self.YELLOW_TIME - elapsed)
        elif self.state == LightState.RED:
            pass  # countdown stays 0 until next green
        return self.state, self.countdown


# ============= Shared State =============

class SharedState:
    """Thread-safe shared state between video and inference threads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.vehicle_count = 0
        self.infer_ms = 0.0
        self.mem_kb = 0
        self.infer_fps = 0.0  # board inference FPS
        self._infer_times = []  # timestamps of recent inferences

    def update_detections(self, detections, vehicle_count, infer_ms, mem_kb):
        with self.lock:
            self.detections = detections
            self.vehicle_count = vehicle_count
            self.infer_ms = infer_ms
            self.mem_kb = mem_kb
            now = time.time()
            self._infer_times.append(now)
            # Keep last 10 timestamps for FPS calc
            self._infer_times = [t for t in self._infer_times if now - t < 30]
            if len(self._infer_times) >= 2:
                span = self._infer_times[-1] - self._infer_times[0]
                self.infer_fps = (len(self._infer_times) - 1) / span if span > 0 else 0
            else:
                self.infer_fps = 0

    def get(self):
        with self.lock:
            return (list(self.detections), self.vehicle_count,
                    self.infer_ms, self.mem_kb, self.infer_fps)


# ============= Frame Preprocessing =============

def preprocess_frame(frame, roi=ROI, size=INPUT_SIZE):
    """Crop ROI from display-res frame, resize to 640x640, normalize → raw bytes."""
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    img = resized[:, :, ::-1].astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img.tobytes()


# ============= Frame Annotation =============

def annotate_frame(frame, detections, signal_state, countdown, vehicle_count,
                   infer_ms, infer_fps, roi=ROI):
    """Draw detections, ROI, and info overlay on frame."""
    display = frame.copy()
    x1, y1, x2, y2 = roi
    roi_w, roi_h = x2 - x1, y2 - y1

    # Draw ROI
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(display, f"Detection Zone ({vehicle_count} vehicles)",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw detections (scale from 640x640 back to ROI on display frame)
    for det in detections:
        dx1 = int(det["x1"] / 640 * roi_w) + x1
        dy1 = int(det["y1"] / 640 * roi_h) + y1
        dx2 = int(det["x2"] / 640 * roi_w) + x1
        dy2 = int(det["y2"] / 640 * roi_h) + y1

        color = (0, 255, 0) if det["class"] in ("car", "bus", "truck") else (255, 200, 0)
        cv2.rectangle(display, (dx1, dy1), (dx2, dy2), color, 2)
        label = f'{det["class"]} {det["conf"]:.2f}'
        cv2.putText(display, label, (dx1, dy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Virtual traffic light
    draw_traffic_light(display, signal_state, 30, 30, countdown)

    # Info bar at top
    info = f"Board: {infer_ms:.0f}ms ({infer_fps:.2f} fps) | Vehicles: {vehicle_count}"
    cv2.putText(display, info, (130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return display


def draw_traffic_light(frame, state, x, y, countdown):
    cv2.rectangle(frame, (x, y), (x + 50, y + 135), (50, 50, 50), -1)
    c_red = (0, 0, 100)
    c_yellow = (0, 100, 100)
    c_green = (0, 100, 0)
    if state == LightState.RED:
        c_red = (0, 0, 255)
    elif state == LightState.YELLOW:
        c_yellow = (0, 255, 255)
    elif state == LightState.GREEN:
        c_green = (0, 255, 0)
    cv2.circle(frame, (x + 25, y + 25), 15, c_red, -1)
    cv2.circle(frame, (x + 25, y + 60), 15, c_yellow, -1)
    cv2.circle(frame, (x + 25, y + 95), 15, c_green, -1)
    cv2.putText(frame, f"{countdown:.0f}s", (x + 3, y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ============= Flask + SocketIO Web Server =============

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>RK3568 Smart Traffic Light Demo</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', sans-serif;
    background: #1a1a2e;
    color: #eee;
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.header {
    background: #16213e;
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #0f3460;
}
.header h1 { font-size: 1.2em; color: #e94560; }
.status { font-size: 0.9em; color: #888; }
.status.connected { color: #4ecca3; }
.main {
    display: flex;
    flex: 1;
    overflow: hidden;
}
.video-panel {
    flex: 3;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #0f0f23;
    padding: 10px;
}
.video-panel img {
    max-width: 100%;
    max-height: 100%;
    border: 2px solid #333;
    border-radius: 4px;
}
.side-panel {
    flex: 1;
    min-width: 220px;
    max-width: 300px;
    background: #16213e;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    border-left: 2px solid #0f3460;
}
.traffic-light {
    background: #222;
    border-radius: 15px;
    padding: 15px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}
.light {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    transition: all 0.3s;
}
.light.red { background: #3a0000; box-shadow: none; }
.light.red.on { background: #ff0000; box-shadow: 0 0 30px #ff0000, 0 0 60px #ff000088; }
.light.yellow { background: #3a3a00; box-shadow: none; }
.light.yellow.on { background: #ffdd00; box-shadow: 0 0 30px #ffdd00, 0 0 60px #ffdd0088; }
.light.green { background: #003a00; box-shadow: none; }
.light.green.on { background: #00ff00; box-shadow: 0 0 30px #00ff00, 0 0 60px #00ff0088; }
.countdown {
    font-size: 2em;
    font-weight: bold;
    color: #fff;
    text-align: center;
}
.info-card {
    background: #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
    width: 100%;
}
.info-card h3 { font-size: 0.8em; color: #888; margin-bottom: 5px; }
.info-card .value { font-size: 1.4em; font-weight: bold; }
.footer {
    background: #16213e;
    padding: 10px 20px;
    border-top: 2px solid #0f3460;
    display: flex;
    gap: 20px;
    align-items: center;
    font-size: 0.85em;
}
.chart-container {
    flex: 1;
    height: 80px;
    position: relative;
}
canvas {
    width: 100%;
    height: 100%;
}
</style>
</head>
<body>
<div class="header">
    <h1>RK3568 Smart Traffic Light</h1>
    <span class="status" id="status">Disconnected</span>
</div>
<div class="main">
    <div class="video-panel">
        <img id="video-frame" src="" alt="Waiting for video...">
    </div>
    <div class="side-panel">
        <div class="traffic-light">
            <div class="light red" id="light-red"></div>
            <div class="light yellow" id="light-yellow"></div>
            <div class="light green" id="light-green"></div>
        </div>
        <div class="countdown" id="countdown">--</div>
        <div class="info-card">
            <h3>Vehicles Detected</h3>
            <div class="value" id="vehicles">0</div>
        </div>
        <div class="info-card">
            <h3>Board Inference</h3>
            <div class="value" id="infer-time">-- ms</div>
        </div>
        <div class="info-card">
            <h3>Board FPS</h3>
            <div class="value" id="board-fps">-- fps</div>
        </div>
        <div class="info-card">
            <h3>Board Memory</h3>
            <div class="value" id="mem">-- MB</div>
        </div>
        <div class="info-card">
            <h3>Video FPS</h3>
            <div class="value" id="fps">-- fps</div>
        </div>
    </div>
</div>
<div class="footer">
    <span>Vehicle History:</span>
    <div class="chart-container">
        <canvas id="chart"></canvas>
    </div>
</div>

<script>
const socket = io();
const statusEl = document.getElementById('status');
const frameEl = document.getElementById('video-frame');
const countdownEl = document.getElementById('countdown');
const vehiclesEl = document.getElementById('vehicles');
const inferEl = document.getElementById('infer-time');
const boardFpsEl = document.getElementById('board-fps');
const memEl = document.getElementById('mem');
const fpsEl = document.getElementById('fps');
const lightRed = document.getElementById('light-red');
const lightYellow = document.getElementById('light-yellow');
const lightGreen = document.getElementById('light-green');
const chart = document.getElementById('chart');
const ctx = chart.getContext('2d');

let vehicleHistory = [];
const MAX_HISTORY = 100;
let lastFrameTime = 0;
let fpsSamples = [];

socket.on('connect', () => {
    statusEl.textContent = 'Connected';
    statusEl.className = 'status connected';
});
socket.on('disconnect', () => {
    statusEl.textContent = 'Disconnected';
    statusEl.className = 'status';
});

socket.on('frame', (data) => {
    frameEl.src = 'data:image/jpeg;base64,' + data.image;

    const state = data.state;
    lightRed.className = 'light red' + (state === 'RED' ? ' on' : '');
    lightYellow.className = 'light yellow' + (state === 'YELLOW' ? ' on' : '');
    lightGreen.className = 'light green' + (state === 'GREEN' ? ' on' : '');

    countdownEl.textContent = data.countdown.toFixed(0) + 's';
    countdownEl.style.color = state === 'RED' ? '#ff4444' :
                              state === 'YELLOW' ? '#ffdd00' : '#44ff44';
    vehiclesEl.textContent = data.vehicles;
    inferEl.textContent = data.infer_ms.toFixed(0) + ' ms';
    boardFpsEl.textContent = data.board_fps.toFixed(2) + ' fps';
    memEl.textContent = (data.mem_kb / 1024).toFixed(1) + ' MB';

    // Video FPS
    const now = performance.now();
    if (lastFrameTime > 0) {
        fpsSamples.push(1000 / (now - lastFrameTime));
        if (fpsSamples.length > 30) fpsSamples.shift();
        const avgFps = fpsSamples.reduce((a, b) => a + b) / fpsSamples.length;
        fpsEl.textContent = avgFps.toFixed(1) + ' fps';
    }
    lastFrameTime = now;

    // Vehicle history (only update on detection changes)
    if (data.det_updated) {
        vehicleHistory.push(data.vehicles);
        if (vehicleHistory.length > MAX_HISTORY) vehicleHistory.shift();
        drawChart();
    }
});

function drawChart() {
    const w = chart.width = chart.offsetWidth * 2;
    const h = chart.height = chart.offsetHeight * 2;
    ctx.clearRect(0, 0, w, h);
    if (vehicleHistory.length < 2) return;
    const maxV = Math.max(15, ...vehicleHistory);
    const step = w / (MAX_HISTORY - 1);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
        const y = h - (i / 3) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.strokeStyle = '#4ecca3';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const offset = MAX_HISTORY - vehicleHistory.length;
    for (let i = 0; i < vehicleHistory.length; i++) {
        const x = (offset + i) * step;
        const y = h - (vehicleHistory[i] / maxV) * h * 0.9;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo((offset + vehicleHistory.length - 1) * step, h);
    ctx.lineTo(offset * step, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(78, 204, 163, 0.1)';
    ctx.fill();
}
</script>
</body>
</html>"""


def create_app(demo_state):
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO

    app = Flask(__name__)
    app.config["SECRET_KEY"] = "traffic_demo_2024"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    demo_state["socketio"] = socketio

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    return app, socketio


# ============= Inference Thread =============

def inference_loop(demo_state, shared, controller, args):
    """Separate thread: grabs latest frame, runs inference (local or board), updates shared state."""
    local_infer = demo_state.get("local_infer")
    board = demo_state.get("board")
    gpio = demo_state.get("gpio")
    modbus = demo_state.get("modbus")

    if not local_infer and (not board or not board.sock):
        log.warning("推理线程: 无推理后端, 退出")
        return

    mode = "本地GPU" if local_infer else "板端"
    interval = 1.0 / args.infer_fps
    log.info(f"推理线程启动: {mode}, 目标 {args.infer_fps} fps (间隔 {interval:.1f}s)")

    while demo_state.get("running", True):
        frame = demo_state.get("latest_frame")
        if frame is None:
            time.sleep(0.05)
            continue

        t0 = time.time()
        result = None
        detections = []

        if local_infer:
            # Local GPU inference
            result, detections = local_infer.infer(frame, roi=ROI)
        elif board and board.sock:
            # Board inference via TCP
            raw_data = preprocess_frame(frame)
            result, detections = board.infer(raw_data)

        if result:
            infer_ms = float(result.get("infer_ms", 0))
            mem_kb = int(result.get("mem_rss_kb", 0))
            vehicle_count = int(result.get("vehicles", 0))
            shared.update_detections(detections, vehicle_count, infer_ms, mem_kb)

            # Update FSM
            state, gpio_r, gpio_y, gpio_g = controller.update(vehicle_count)

            # Send GPIO (via board TCP, hdc shell, or Modbus BSM)
            if board and board.sock:
                board.send_gpio(gpio_r, gpio_y, gpio_g)
            elif modbus:
                modbus.set(gpio_r, gpio_y, gpio_g)
            elif gpio:
                gpio.set(gpio_r, gpio_y, gpio_g)

        # Maintain target interval
        elapsed = time.time() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    log.info("推理线程退出")


# ============= Video Playback Thread =============

def video_loop(demo_state, shared, controller, args):
    """Reads video at native FPS, annotates with latest detections, pushes to WebSocket."""
    socketio = demo_state["socketio"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error(f"无法打开视频: {args.video}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = min(video_fps, args.video_fps)
    frame_time = 1.0 / target_fps

    log.info(f"视频: {orig_w}x{orig_h} @ {video_fps:.1f}fps → 显示 {DISPLAY_W}x{DISPLAY_H} @ {target_fps:.0f}fps")

    while demo_state.get("running", True):
        t0 = time.time()

        ret, frame_raw = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize to display resolution
        frame = cv2.resize(frame_raw, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

        # Store latest frame for inference thread
        demo_state["latest_frame"] = frame

        # Get latest detection results
        detections, vehicle_count, infer_ms, mem_kb, infer_fps = shared.get()

        # Tick FSM countdown (no state change, just update timer)
        state, countdown = controller.tick()

        # Annotate
        annotated = annotate_frame(frame, detections, state, countdown,
                                   vehicle_count, infer_ms, infer_fps)

        # Encode JPEG
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

        socketio.emit("frame", {
            "image": jpg_b64,
            "state": state.name,
            "countdown": countdown,
            "vehicles": vehicle_count,
            "infer_ms": infer_ms,
            "mem_kb": mem_kb,
            "board_fps": infer_fps,
            "det_updated": False,  # chart updates on detection events only
        })

        # Maintain target FPS
        elapsed = time.time() - t0
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    log.info("视频线程退出")


# ============= Main =============

def main():
    parser = argparse.ArgumentParser(description="RK3568 Smart Traffic Light Demo")
    parser.add_argument("--video", default=VIDEO_PATH, help="Video file path")
    parser.add_argument("--local", action="store_true",
                        help="Use local GPU inference (ultralytics YOLOv8)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path (for --local)")
    parser.add_argument("--board", default=f"{BOARD_HOST}:{BOARD_PORT}",
                        help="Board daemon address (host:port)")
    parser.add_argument("--no-board", action="store_true", help="Offline mode (no inference)")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--video-fps", type=float, default=12.0,
                        help="Video playback FPS for web display")
    parser.add_argument("--infer-fps", type=float, default=1.0,
                        help="Target inference FPS")
    # GPIO via hdc (for --local mode without board daemon)
    parser.add_argument("--hdc", default="", help="Path to hdc.exe (enables GPIO via hdc shell)")
    parser.add_argument("--gpio-red", type=int, default=0, help="GPIO pin for red light")
    parser.add_argument("--gpio-yellow", type=int, default=3, help="GPIO pin for yellow light")
    parser.add_argument("--gpio-green", type=int, default=18, help="GPIO pin for green light")
    parser.add_argument("--active-low", action="store_true", help="Relay is active-low")
    # Modbus BSM relay control via TCP bridge (bsm_bridge.py)
    parser.add_argument("--modbus", default="", help="BSM bridge address (host:port, e.g. localhost:5555)")
    args = parser.parse_args()

    demo_state = {"running": True, "latest_frame": None}
    shared = SharedState()
    controller = SignalController()

    # Setup local inference if requested
    if args.local:
        demo_state["local_infer"] = LocalInfer(args.model)
    else:
        demo_state["local_infer"] = None

    # Connect to board daemon (for board inference + GPIO, or just GPIO)
    board = None
    if not args.no_board and not args.local:
        host, port = args.board.split(":")
        board = BoardClient(host, int(port))
        if not board.connect():
            log.warning("板端未连接, 以离线模式运行")
            board = None
        elif board.ping():
            log.info("板端 PING 成功")
        else:
            log.warning("板端 PING 失败")
    demo_state["board"] = board

    # Setup GPIO via hdc shell (for --local mode)
    gpio = None
    if args.hdc:
        pins = {"red": args.gpio_red, "yellow": args.gpio_yellow, "green": args.gpio_green}
        gpio = HdcGpioClient(args.hdc, pins, args.active_low)
        gpio.init()
    demo_state["gpio"] = gpio

    # Setup Modbus BSM relay control via TCP bridge
    modbus = None
    if args.modbus:
        parts = args.modbus.split(":")
        host = parts[0] if parts[0] else "localhost"
        port = int(parts[1]) if len(parts) > 1 else 5555
        modbus = ModbusLightClient(host=host, port=port)
        modbus.init()
    demo_state["modbus"] = modbus

    # Create Flask app
    app, socketio = create_app(demo_state)

    # Start video playback thread
    video_thread = threading.Thread(target=video_loop,
                                    args=(demo_state, shared, controller, args), daemon=True)
    video_thread.start()

    # Start inference thread
    infer_thread = threading.Thread(target=inference_loop,
                                    args=(demo_state, shared, controller, args), daemon=True)
    infer_thread.start()

    mode = "本地GPU" if args.local else ("板端" if board else "离线")
    gpio_str = f"Modbus BSM({args.modbus})" if modbus else (f"hdc GPIO({args.gpio_red}/{args.gpio_yellow}/{args.gpio_green})" if gpio else ("板端GPIO" if board else "无"))
    log.info(f"推理模式: {mode} | GPIO: {gpio_str}")
    log.info(f"Web 服务启动: http://{args.host}:{args.port}")
    try:
        socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        demo_state["running"] = False
        if board:
            board.close()
        if modbus:
            modbus.close()
        log.info("Demo 退出")


if __name__ == "__main__":
    main()
