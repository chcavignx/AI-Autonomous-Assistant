#!/usr/bin/env python3
"""Simultaneous Face Recognition & Object Detection - Web Interface.

Real-time browser-based dashboard running unified YOLO object detection
and ArcFace face recognition with asynchronous decoupled tracking.

Access via browser:
    http://<device-ip>:5003
    http://localhost:5003

Usage:
    # Start web server on port 5003:
    python examples/vision/simultaneous_vision_web.py

    # Run automated verification test of web endpoints and video streaming:
    python examples/vision/simultaneous_vision_web.py --test

    # Custom port and host:
    python examples/vision/simultaneous_vision_web.py --host 0.0.0.0 --port 5003
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, Response, jsonify, render_template_string  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.sysutils import get_cpu_temperature_c, get_cpu_usage_percent  # noqa: E402
from src.vision.video_capture import VideoCapture  # noqa: E402

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

app = Flask(__name__)


class WebSimultaneousVision:
    """Manager for Web-based Simultaneous Vision (Object Detection + Face Recognition)."""

    def __init__(self) -> None:
        """Initialize the simultaneous vision manager and load video capture pipeline."""
        self.cfg = load_config()
        self.cfg.vision.enable_object_detection = True
        self.cfg.vision.enable_face_detection = True
        self.cfg.vision.enable_face_recognition = True
        self.cfg.vision.async_face_recognition = True

        self.video_cap = VideoCapture(self.cfg)
        self.running: bool = False
        self.lock = threading.Lock()

        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frame_count: int = 0
        self.fps_start_time: float = time.time()
        self.latest_annotated: np.ndarray | None = None
        self.detected_objects: list[str] = []
        self.detected_faces: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start the video capture pipeline."""
        if not self.running:
            self.running = True
            self.video_cap.start()
            logger.info("Simultaneous vision web pipeline started")

    def stop(self) -> None:
        """Stop video capture and cleanup resources."""
        self.running = False
        self.video_cap.stop()
        logger.info("Simultaneous vision web pipeline stopped")

    def _extract_objects(self) -> list[str]:
        """Extract labels from current YOLO results."""
        current_objects: list[str] = []
        if self.video_cap.latest_results is not None:
            boxes = getattr(self.video_cap.latest_results, "boxes", None)
            names_dict = getattr(self.video_cap.latest_results, "names", {})
            if boxes is not None and hasattr(boxes, "cls"):
                for cls_id in boxes.cls:
                    cid = int(cls_id.item() if hasattr(cls_id, "item") else cls_id)
                    current_objects.append(names_dict.get(cid, f"class_{cid}"))
            elif hasattr(self.video_cap.latest_results, "detections") and isinstance(
                self.video_cap.latest_results.detections, (list, tuple)
            ):
                current_objects.extend(
                    d.get("label", "object")
                    for d in self.video_cap.latest_results.detections
                    if isinstance(d, dict)
                )
        return current_objects

    def _process_stream_frame(self) -> tuple[np.ndarray | None, float]:
        """Capture and process a single frame from the camera."""
        t0 = time.perf_counter()
        frame = self.video_cap.capture_frame()

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(20, 80, 640, dtype=np.uint8)
            frame[:, :, 2] = np.linspace(80, 20, 640, dtype=np.uint8)
            cv2.putText(
                frame,
                "Simultaneous Vision Feed",
                (140, 240),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 255),
                1,
            )
            time.sleep(0.04)

        annotated = self.video_cap.process_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return annotated, dt_ms

    def _update_stats(self, annotated: np.ndarray, dt_ms: float) -> None:
        """Update live statistics under lock."""
        current_objects = self._extract_objects()
        current_faces = self.video_cap.active_tracks

        with self.lock:
            self.latest_annotated = annotated
            self.detected_objects = current_objects
            self.detected_faces = current_faces
            self.latency_ms = dt_ms
            self.frame_count += 1
            if self.frame_count % 15 == 0:
                elapsed = time.time() - self.fps_start_time
                if elapsed > 0:
                    self.fps = 15.0 / elapsed
                self.fps_start_time = time.time()

    def _encode_frame_jpeg(self, annotated: np.ndarray) -> bytes | None:
        """Encode frame to JPEG bytes."""
        ret, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg_buf.tobytes() if ret else None

    def _next_mjpeg_chunk(self) -> bytes | None:
        """Generate next MJPEG chunk."""
        annotated, dt_ms = self._process_stream_frame()
        if annotated is None:
            return None
        self._update_stats(annotated, dt_ms)
        raw_bytes = self._encode_frame_jpeg(annotated)
        if raw_bytes is None:
            return None
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + raw_bytes + b"\r\n"

    def generate_mjpeg_frames(self) -> Any:
        """Yield multipart MJPEG frames for live browser streaming."""
        while self.running:
            try:
                chunk = self._next_mjpeg_chunk()
                if chunk is not None:
                    yield chunk
            except Exception as e:
                logger.debug("Error in MJPEG frame generator: %s", e)
                time.sleep(0.05)


_vision_tool_instance: WebSimultaneousVision | None = None


def get_vision_tool() -> WebSimultaneousVision:
    """Retrieve or lazily initialize the singleton WebSimultaneousVision instance."""
    global _vision_tool_instance  # noqa: PLW0603
    if _vision_tool_instance is None:
        _vision_tool_instance = WebSimultaneousVision()
    return _vision_tool_instance


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simultaneous Vision - Object Detection & Face Recognition</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0b0b0f;
            --bg-card: #14141c;
            --bg-card-hover: #1a1a24;
            --border-color: #262636;
            --accent-violet: #8b5cf6;
            --accent-emerald: #10b981;
            --accent-cyan: #06b6d4;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-main);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }

        h1 {
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-badge {
            font-size: 13px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .layout-grid {
            display: grid;
            grid-template-columns: 1.8fr 1fr;
            gap: 24px;
        }

        @media (max-width: 992px) {
            .layout-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--text-main);
        }

        .video-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .video-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        @media (max-width: 600px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .stat-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }

        .stat-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .stat-val {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-main);
        }

        .controls-panel {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .toggle-btn {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-card-hover);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }

        .toggle-btn:hover {
            border-color: var(--accent-violet);
            transform: translateY(-1px);
        }

        .toggle-state {
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
        }

        .toggle-state.on {
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-emerald);
            border: 1px solid var(--accent-emerald);
        }

        .toggle-state.off {
            background: rgba(244, 63, 94, 0.2);
            color: var(--accent-rose);
            border: 1px solid var(--accent-rose);
        }

        .action-btn {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
            color: #fff;
            border: none;
            padding: 14px;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-family: inherit;
        }

        .action-btn:hover {
            filter: brightness(1.1);
            transform: translateY(-1px);
        }

        .entities-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 220px;
            overflow-y: auto;
            margin-top: 12px;
        }

        .entity-tag {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
        }

        .entity-name {
            font-weight: 600;
            color: var(--accent-cyan);
        }

        .entity-score {
            font-size: 12px;
            color: var(--accent-emerald);
            font-weight: 700;
        }

        #alert-box {
            display: none;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 14px;
            font-weight: 600;
        }
        .alert-success { background: rgba(16, 185, 129, 0.2); border: 1px solid var(--accent-emerald); color: var(--accent-emerald); }
        .alert-error { background: rgba(244, 63, 94, 0.2); border: 1px solid var(--accent-rose); color: var(--accent-rose); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ Simultaneous Vision Dashboard</h1>
            <div class="status-badge">
                <div class="status-dot"></div>
                LIVE INFERENCE
            </div>
        </header>

        <div id="alert-box"></div>

        <div class="layout-grid">
            <!-- Left Column: Video Stream & Main Metrics -->
            <div class="card">
                <div class="card-header">
                    <span>Live Camera Feed</span>
                    <span id="resolution-badge" style="font-size: 12px; color: var(--text-muted);">640 × 480</span>
                </div>
                <div class="video-wrapper">
                    <img src="/video_feed" alt="Simultaneous Vision Video Feed">
                </div>

                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Pipeline FPS</div>
                        <div class="stat-val" id="stat-fps" style="color: var(--accent-emerald);">0.0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Latency</div>
                        <div class="stat-val" id="stat-latency" style="color: var(--accent-cyan);">0 ms</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">CPU Load</div>
                        <div class="stat-val" id="stat-cpu">0%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Temperature</div>
                        <div class="stat-val" id="stat-temp" style="color: var(--accent-amber);">--°C</div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Interactive Deck & Entities -->
            <div style="display: flex; flex-direction: column; gap: 24px;">
                <div class="card">
                    <div class="card-header">
                        <span>Pipeline Control Deck</span>
                    </div>
                    <div class="controls-panel">
                        <button class="toggle-btn" onclick="toggleFeature('objects')">
                            <span>📦 YOLO Object Detection</span>
                            <span class="toggle-state on" id="state-objects">ON</span>
                        </button>
                        <button class="toggle-btn" onclick="toggleFeature('faces')">
                            <span>👤 Face Detection</span>
                            <span class="toggle-state on" id="state-faces">ON</span>
                        </button>
                        <button class="toggle-btn" onclick="toggleFeature('recognition')">
                            <span>🎯 ArcFace Recognition</span>
                            <span class="toggle-state on" id="state-recognition">ON</span>
                        </button>
                        <button class="toggle-btn" onclick="toggleFeature('async')">
                            <span>⚡ Async Worker Decoupling</span>
                            <span class="toggle-state on" id="state-async">ON</span>
                        </button>
                        <button class="action-btn" onclick="capturePhoto()">
                            📷 Capture Annotated Snapshot
                        </button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span>Live Entities Detected</span>
                    </div>
                    <div>
                        <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;">
                            Recognized Faces (<span id="count-faces">0</span>)
                        </div>
                        <div class="entities-list" id="faces-list">
                            <div style="color: var(--text-muted); font-size: 13px;">No faces currently tracked</div>
                        </div>

                        <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-top: 14px; margin-bottom: 6px;">
                            Detected Objects (<span id="count-objects">0</span>)
                        </div>
                        <div class="entities-list" id="objects-list">
                            <div style="color: var(--text-muted); font-size: 13px;">No objects detected</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showAlert(text, isError = false) {
            const box = document.getElementById('alert-box');
            box.textContent = text;
            box.className = isError ? 'alert-error' : 'alert-success';
            box.style.display = 'block';
            setTimeout(() => { box.style.display = 'none'; }, 3500);
        }

        function toggleFeature(feature) {
            fetch(`/toggle/${feature}`, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, !data.success);
                    updateStats();
                })
                .catch(err => showAlert('Error toggling feature', true));
        }

        function capturePhoto() {
            fetch('/capture_photo', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    showAlert(data.message, !data.success);
                })
                .catch(err => showAlert('Capture error', true));
        }

        function updateStats() {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('stat-fps').textContent = data.fps.toFixed(1);
                    document.getElementById('stat-latency').textContent = data.latency_ms.toFixed(0) + ' ms';
                    document.getElementById('stat-cpu').textContent = data.cpu_percent.toFixed(1) + '%';
                    document.getElementById('stat-temp').textContent = data.temp_c ? (data.temp_c.toFixed(1) + '°C') : 'N/A';

                    const setToggle = (id, val) => {
                        const el = document.getElementById(id);
                        el.textContent = val ? 'ON' : 'OFF';
                        el.className = 'toggle-state ' + (val ? 'on' : 'off');
                    };
                    setToggle('state-objects', data.enable_objects);
                    setToggle('state-faces', data.enable_faces);
                    setToggle('state-recognition', data.enable_recognition);
                    setToggle('state-async', data.async_recognition);

                    document.getElementById('count-faces').textContent = data.faces.length;
                    const facesContainer = document.getElementById('faces-list');
                    if (data.faces.length === 0) {
                        facesContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 13px;">No faces currently tracked</div>';
                    } else {
                        facesContainer.innerHTML = data.faces.map(f => `
                            <div class="entity-tag">
                                <span class="entity-name">👤 ${f.name}</span>
                                <span class="entity-score">sim: ${f.similarity.toFixed(2)}</span>
                            </div>
                        `).join('');
                    }

                    document.getElementById('count-objects').textContent = data.objects.length;
                    const objsContainer = document.getElementById('objects-list');
                    if (data.objects.length === 0) {
                        objsContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 13px;">No objects detected</div>';
                    } else {
                        objsContainer.innerHTML = data.objects.map(obj => `
                            <div class="entity-tag">
                                <span class="entity-name">📦 ${obj}</span>
                                <span class="entity-score">detected</span>
                            </div>
                        `).join('');
                    }
                })
                .catch(err => console.debug('Stats poll error', err));
        }

        setInterval(updateStats, 1000);
        updateStats();
    </script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    """Render the dashboard UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed() -> Response:
    """Video streaming route providing multipart MJPEG stream."""
    tool = get_vision_tool()
    return Response(
        tool.generate_mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats() -> Response:
    """Return JSON statistics of the simultaneous pipeline."""
    tool = get_vision_tool()
    vc = tool.video_cap
    with tool.lock:
        data = {
            "fps": tool.fps,
            "latency_ms": tool.latency_ms,
            "frame_count": tool.frame_count,
            "cpu_percent": get_cpu_usage_percent(),
            "temp_c": get_cpu_temperature_c(),
            "enable_objects": vc.enable_object_detection,
            "enable_faces": vc.enable_face_detection,
            "enable_recognition": vc.enable_face_recognition,
            "async_recognition": vc.async_face_recognition,
            "objects": tool.detected_objects,
            "faces": tool.detected_faces,
        }
    return jsonify(data)


@app.route("/toggle/<feature>", methods=["POST"])
def toggle_feature(feature: str) -> Response:
    """Toggle pipeline feature (objects, faces, recognition, async)."""
    tool = get_vision_tool()
    vc = tool.video_cap
    if feature == "objects":
        vc.enable_object_detection = not vc.enable_object_detection
        val = vc.enable_object_detection
        return jsonify({"success": True, "message": f"Object detection {'enabled' if val else 'disabled'}"})
    if feature == "faces":
        vc.enable_face_detection = not vc.enable_face_detection
        val = vc.enable_face_detection
        return jsonify({"success": True, "message": f"Face detection {'enabled' if val else 'disabled'}"})
    if feature == "recognition":
        vc.enable_face_recognition = not vc.enable_face_recognition
        val = vc.enable_face_recognition
        return jsonify({"success": True, "message": f"Face recognition {'enabled' if val else 'disabled'}"})
    if feature == "async":
        vc.async_face_recognition = not vc.async_face_recognition
        val = vc.async_face_recognition
        return jsonify({"success": True, "message": f"Async recognition mode {'enabled' if val else 'disabled'}"})

    return jsonify({"success": False, "message": f"Unknown feature '{feature}'"})


@app.route("/capture_photo", methods=["POST"])
def capture_photo() -> Response:
    """Save an annotated snapshot from the video capture instance."""
    tool = get_vision_tool()
    success, msg = tool.video_cap.capture_photo()
    return jsonify({"success": success, "message": msg})


def _test_client_routes(client: Any) -> None:
    """Test standard HTTP routes on Flask test client."""
    resp_index = client.get("/")
    if resp_index.status_code != 200 or b"Simultaneous Vision Dashboard" not in resp_index.data:
        msg = "Index route test failed"
        raise RuntimeError(msg)
    logger.info("✓ Index page rendered successfully (HTTP 200)")

    resp_stats = client.get("/stats")
    if resp_stats.status_code != 200 or "enable_objects" not in resp_stats.get_json():
        msg = "Stats route test failed"
        raise RuntimeError(msg)
    logger.info("✓ Stats endpoint operational (HTTP 200, JSON valid)")

    resp_toggle = client.post("/toggle/objects")
    if resp_toggle.status_code != 200 or not resp_toggle.get_json().get("success"):
        msg = "Toggle objects endpoint failed"
        raise RuntimeError(msg)

    resp_toggle_rec = client.post("/toggle/recognition")
    if resp_toggle_rec.status_code != 200 or not resp_toggle_rec.get_json().get("success"):
        msg = "Toggle recognition endpoint failed"
        raise RuntimeError(msg)
    logger.info("✓ Feature toggle endpoints functional")


def _test_mjpeg_stream(tool: WebSimultaneousVision) -> None:
    """Test MJPEG stream generator chunks."""
    gen = tool.generate_mjpeg_frames()
    first_chunk = next(gen, None)
    if first_chunk is None or not isinstance(first_chunk, bytes):
        msg = "MJPEG stream generator did not yield valid frame chunk"
        raise RuntimeError(msg)
    if b"Content-Type: image/jpeg" not in first_chunk:
        msg = "MJPEG chunk missing image/jpeg content type"
        raise RuntimeError(msg)
    logger.info("✓ Live MJPEG stream generated valid multipart JPEG chunk")


def _run_test_client_suite(tool: WebSimultaneousVision) -> None:
    """Run tests on Flask test client and stream."""
    with app.test_client() as client:
        _test_client_routes(client)
        _test_mjpeg_stream(tool)


def run_automated_web_test() -> bool:
    """Run automated verification test of Flask endpoints and Simultaneous Vision streaming."""
    logger.info("============================================================")
    logger.info("  STARTING AUTOMATED SIMULTANEOUS VISION WEB TEST")
    logger.info("============================================================")

    test_passed = True
    tool = get_vision_tool()
    tool.start()

    try:
        _run_test_client_suite(tool)
        logger.info("============================================================")
        logger.info("  ALL WEB ENDPOINT TESTS PASSED SUCCESSFULLY!")
        logger.info("============================================================")
    except Exception as e:
        logger.exception("✗ Web test failed: %s", e)
        test_passed = False
    finally:
        tool.stop()

    return test_passed


def main() -> int:
    """Execute main web application."""
    parser = argparse.ArgumentParser(description="Simultaneous Vision Web Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")  # noqa: S104
    parser.add_argument("--port", type=int, default=5003, help="Port number for web dashboard")
    parser.add_argument("--test", action="store_true", help="Run automated test suite and exit")
    args = parser.parse_args()

    if args.test:
        success = run_automated_web_test()
        return 0 if success else 1

    logger.info("============================================================")
    logger.info("  Simultaneous Vision Dashboard - Web Interface")
    logger.info("============================================================")

    tool = get_vision_tool()
    tool.start()
    atexit.register(tool.stop)

    ip_addr = "<raspberry-pi-ip>"
    with contextlib.suppress(Exception):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
        s.close()

    logger.info("Access the web dashboard in your browser:")
    logger.info("  Local:   http://localhost:%d", args.port)
    logger.info("  Network: http://%s:%d", ip_addr, args.port)
    logger.info("============================================================")

    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
