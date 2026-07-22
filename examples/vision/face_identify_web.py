#!/usr/bin/env python3
"""
Face Identify Tool - Web Interface Version
Access via browser: http://<raspberry-pi-ip>:5001
Works over SSH - no X11 forwarding needed
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, render_template, Response, request, jsonify, render_template_string # pyright: ignore[reportUnusedImport]
import cv2
import logging
import os
import time
from datetime import datetime # pyright: ignore[reportUnusedImport]
import numpy as np
import threading

from src.utils.sysutils import detect_raspberry_pi_model # pyright: ignore[reportUnusedImport]
from src.utils.config import config
from src.utils.camera import discover_pi_cameras
from src.vision.face_insight_pipeline import FaceInsightPipeline

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

app = Flask(__name__)

OUTPUT_DIR = "face_dataset"
output_dir = str(project_root / "data" / OUTPUT_DIR)

class WebFaceIdentify:
    def __init__(self):
        self.cfg = config
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None

        # Load dataset path from config or fallback
        if hasattr(self.cfg.vision, "face_dataset_path") and self.cfg.vision.face_dataset_path:
            self.dataset_dir = self.cfg.vision.face_dataset_path
        else:
            self.dataset_dir = output_dir

        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.latest_frame = None
        self.faces_detected = 0
        self.running = False
        self.lock = threading.Lock()

        # Initialize recognizer pipeline
        self.recognizer = FaceInsightPipeline(self.cfg)
        self.registered_faces = 0

        # Load known faces
        self.load_dataset()

        self.initialize_cameras()

    def load_dataset(self):
        """Load known faces from dataset directory"""
        print(f"Loading dataset from: {self.dataset_dir}")
        if not os.path.exists(self.dataset_dir):
            print(f"Warning: Dataset directory {self.dataset_dir} does not exist.")
            return

        for person_name in os.listdir(self.dataset_dir):
            person_path = os.path.join(self.dataset_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            # Use full frames for better context detection, limit to 10 for speed
            full_frames = [f for f in os.listdir(person_path) if "_full" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not full_frames:
                # Fallback to any images
                full_frames = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for idx, img_name in enumerate(full_frames[:10]):
                img_path = os.path.join(person_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Append index to ID
                        face_id = f"{person_name}_{idx}"
                        self.recognizer.register_face(face_id, img)
                        self.registered_faces += 1
                except Exception as e:
                    print(f"Failed to register face {img_path}: {e}")

        print(f"Loaded {self.registered_faces} face templates.")

    def initialize_cameras(self):
        """Initialize available cameras"""
        for cam in discover_pi_cameras():
            self.cameras.append({
                'index': cam.index,
                'name': f'Camera {cam.index}',
                'camera': cam,
                'info': cam.get_info_str()
            })
            logger.info(f"✓ Camera {cam.index} detected")

        if not self.cameras:
            logger.warning("No cameras available. Camera streaming will be disabled.")

    def start_camera(self, idx=0):
        """Start selected camera"""
        if not self.cameras:
            logger.error("No camera available to start.")
            return

        if self.current_camera:
            try:
                self.current_camera['camera'].stop()
            except:
                pass

        self.current_camera = self.cameras[idx]
        self.current_camera['camera'].start(
            self.cfg.vision.camera.frame_width,
            self.cfg.vision.camera.frame_height,
            self.cfg.vision.camera.format
        )
        time.sleep(1)

    def generate_frames(self):
        """Generate video frames for streaming"""
        while self.running:
            try:
                if not self.current_camera:
                    # If no camera, yield a blank black placeholder frame periodically
                    blank_frame = np.zeros((self.cfg.vision.camera.frame_height, self.cfg.vision.camera.frame_width, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "No Camera Available", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', blank_frame) # pyright: ignore[reportUnusedVariable]
                    yield (b'--frame\r\n' # pyright: ignore[reportImplicitStringConcatenation]
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.5)
                    continue

                frame = self.current_camera['camera'].capture_frame()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Recognize faces
                faces = self.recognizer.recognize(frame_bgr, thresh=0.4)

                with self.lock:
                    self.faces_detected = len(faces)
                    self.latest_frame = frame_bgr.copy()

                # Draw faces and info
                for f in faces:
                    bbox = [int(val) for val in f.bbox]
                    raw_id = f.identity or "unknown"

                    if raw_id != "unknown":
                        person_name = raw_id.rsplit('_', 1)[0]
                    else:
                        person_name = "unknown"

                    sim = f.similarity or 0.0

                    color = (0, 255, 0) if person_name != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                    label = f"{person_name} ({sim:.2f})"
                    cv2.putText(
                        frame_bgr,
                        label,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                # Info overlay
                info_text = [
                    f"Camera: {self.current_camera['name']}",
                    f"Faces Detected: {len(faces)}",
                    f"DB Size: {self.registered_faces} faces"
                ]

                y_pos = 25
                for text in info_text:
                    cv2.putText(frame_bgr, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y_pos += 25

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.fps = 30 / elapsed
                    self.fps_start_time = time.time()

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame_bgr) # pyright: ignore[reportUnusedVariable]
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n' # pyright: ignore[reportImplicitStringConcatenation]
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(0.1)

# Global instance
identify_tool = None

@app.route('/')
def index():
    """Main page"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>SOTA Face Identification</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #0b0b0f;
                --bg-card: #14141b;
                --bg-control: #1d1d26;
                --border-color: #2b2b3a;
                --accent-primary: #8b5cf6; /* Violet */
                --accent-secondary: #10b981; /* Emerald */
                --accent-danger: #ef4444; /* Red */
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
            }

            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg-main);
                color: var(--text-main);
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }

            .container {
                max-width: 1000px;
                width: 100%;
                margin: 0 auto;
            }

            h1 {
                text-align: center;
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }

            #video-container {
                position: relative;
                background: #000;
                border: 2px solid var(--border-color);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                aspect-ratio: 4/3;
                max-height: 480px;
                margin: 0 auto 25px auto;
                max-width: 640px;
            }

            #video-stream {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }

            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }

            .stat-box {
                background: var(--bg-card);
                padding: 20px;
                border-radius: 14px;
                border: 1px solid var(--border-color);
                transition: transform 0.2s;
            }

            .stat-box:hover {
                transform: translateY(-2px);
            }

            .stat-label {
                color: var(--text-muted);
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }

            .stat-value {
                color: var(--accent-secondary);
                font-size: 28px;
                font-weight: 700;
            }

            .controls {
                background: var(--bg-card);
                padding: 25px;
                border-radius: 16px;
                border: 1px solid var(--border-color);
                margin-bottom: 25px;
                text-align: center;
            }

            button {
                background: var(--accent-primary);
                color: #fff;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                font-family: inherit;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }

            button:hover {
                filter: brightness(1.1);
                transform: translateY(-1px);
            }

            .btn-outline {
                background: transparent;
                border: 1px solid var(--border-color);
                color: var(--text-main);
            }

            .btn-outline:hover {
                background: rgba(255, 255, 255, 0.05);
            }

            #message {
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 25px;
                display: none;
                font-weight: 600;
            }
            .success { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent-secondary); color: var(--accent-secondary); }
            .error { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--accent-danger); color: var(--accent-danger); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Live Face Identification</h1>

            <div id="message"></div>

            <div id="video-container">
                <img id="video-stream" src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Faces Detected</div>
                    <div class="stat-value" id="faces-count">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Database Size</div>
                    <div class="stat-value" id="db-size">{{ db_size }} templates</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fps">0</div>
                </div>
            </div>

            <div class="controls">
                <button class="btn-outline" onclick="switchCamera()">🔄 Switch Camera</button>
            </div>
        </div>

        <script>
            function showMessage(text, isError = false) {
                const msgDiv = document.getElementById('message');
                msgDiv.textContent = text;
                msgDiv.className = isError ? 'error' : 'success';
                msgDiv.style.display = 'block';
                setTimeout(() => {
                    msgDiv.style.display = 'none';
                }, 4000);
            }

            function switchCamera() {
                fetch('/switch_camera')
                .then(response => response.json())
                .then(data => {
                    showMessage(data.message);
                });
            }

            function updateStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('faces-count').textContent = data.faces_detected;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                });
            }

            setInterval(updateStats, 2000);
            updateStats();
        </script>
    </body>
    </html>
    """
    return render_template_string(html, db_size=identify_tool.registered_faces)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(identify_tool.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera')
def switch_camera():
    """Switch to next camera"""
    if len(identify_tool.cameras) > 1:
        next_idx = (identify_tool.current_camera_idx + 1) % len(identify_tool.cameras)
        identify_tool.current_camera_idx = next_idx
        identify_tool.start_camera(next_idx)
        return jsonify({'success': True, 'message': f'Switched to {identify_tool.cameras[next_idx]["name"]}'})
    return jsonify({'success': False, 'message': 'Only one camera available'})

@app.route('/stats')
def stats():
    """Get current statistics"""
    return jsonify({
        'faces_detected': identify_tool.faces_detected,
        'fps': identify_tool.fps
    })

if __name__ == '__main__':
    print("="*60)
    print("  Face Identify Tool - Web Interface")
    print("="*60)
    print("\nInitializing cameras...")

    identify_tool = WebFaceIdentify()
    identify_tool.start_camera(0)
    identify_tool.running = True

    print("\n✓ Camera initialized")
    print("\nAccess the interface at:")
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
        s.close()
    except Exception:
        ip_addr = "<raspberry-pi-ip>"

    # Use port 5001 to avoid conflicting with face_capture_web.py if it's running
    port = 5001
    print(f"  http://{ip_addr}:{port}")
    print(f"  or")
    print(f"  http://localhost:{port} (if accessing locally)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=port, threaded=True)
