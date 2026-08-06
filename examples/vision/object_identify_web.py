#!/usr/bin/env python3
"""
Object Identify & Detection Tool - Web Interface Version
Access via browser: http://<raspberry-pi-ip>:5002
Works over SSH - no X11 forwarding needed
"""

import atexit
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from src.utils.camera import discover_pi_cameras
from src.utils.config import config
from src.vision.video_capture import VideoCapture

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

app = Flask(__name__)


class WebObjectIdentify:
    """Manager for Web-based Object Detection & Identification."""

    def __init__(self):
        self.cfg = config
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None

        self.initialize_cameras()

        self.video_cap = VideoCapture(self.cfg)
        self.processor = self.video_cap.object_processor
        self.conf_threshold = self.cfg.vision.object_recognition_threshold

        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.latest_frame = None
        self.objects_detected = 0
        self.detected_labels = {}
        self.running = False
        self.lock = threading.Lock()

    def initialize_cameras(self):
        """Initialize available cameras."""
        for cam in discover_pi_cameras():
            self.cameras.append({
                'index': cam.index,
                'name': f'Camera {cam.index}',
                'camera': cam,
                'info': cam.get_info_str()
            })
            logger.info(f"✓ Camera {cam.index} detected")

        if not self.cameras:
            # Fallback probe for OpenCV VideoCapture devices (e.g. USB webcams)
            for idx in [0, 1]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cameras.append({
                            'index': idx,
                            'name': f'Camera {idx}',
                            'camera': None,
                            'info': f'OpenCV Camera {idx}'
                        })
                        logger.info(f"✓ OpenCV Camera {idx} detected")
                    cap.release()

        if self.cameras:
            self.current_camera_idx = 0
            self.current_camera = self.cameras[0]
        else:
            logger.warning("No cameras available.")

    def start_camera(self, idx=0):
        """Start the selected camera."""
        if self.cameras:
            idx = idx % len(self.cameras)
            self.current_camera_idx = idx
            self.current_camera = self.cameras[idx]
            cam_index = self.cameras[idx]['index']
        else:
            self.current_camera_idx = idx
            cam_index = idx

        self.cfg.vision.camera.camera_index = cam_index
        if self.video_cap.running:
            self.video_cap.stop()
        self.video_cap._initialize_camera()
        self.video_cap.start()

    def stop(self):
        """Stop camera capture and terminate frame generator loop."""
        self.running = False
        if hasattr(self, "video_cap") and self.video_cap.running:
            self.video_cap.stop()

    def set_threshold(self, threshold: float):
        """Update detection confidence threshold dynamically."""
        self.conf_threshold = max(0.01, min(1.0, threshold))
        self.cfg.vision.object_recognition_threshold = self.conf_threshold
        if hasattr(self.processor, "detector") and hasattr(self.processor.detector, "conf_thres"):
            self.processor.detector.conf_thres = self.conf_threshold

    def generate_frames(self):
        """Generate annotated video frames for MJPEG streaming."""
        while self.running:
            try:
                frame_bgr = self.video_cap.capture_frame()
                if frame_bgr is None:
                    time.sleep(0.01)
                    continue

                metadata = getattr(self.video_cap, "latest_metadata", None)

                # Sync dynamic confidence threshold with detector
                if hasattr(self.processor, "detector") and hasattr(self.processor.detector, "conf_thres"):
                    self.processor.detector.conf_thres = self.conf_threshold

                # Run object detection
                annotated_frame, detections, _ = self.processor.process_frame(
                    frame_bgr, draw=True, metadata=metadata
                )

                # Filter detections by dynamic confidence threshold
                filtered_detections = [d for d in detections if d.get('score', 0.0) >= self.conf_threshold]

                # Update stats
                label_counts = {}
                for d in filtered_detections:
                    lbl = d.get('label', 'unknown')
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                with self.lock:
                    self.objects_detected = len(filtered_detections)
                    self.detected_labels = label_counts
                    self.latest_frame = annotated_frame.copy()

                # Add status overlay
                info_text = [
                    f"Model: {self.cfg.vision.object_model_type} ({self.cfg.vision.object_model_name})",
                    f"Objects: {len(filtered_detections)}",
                    f"Conf Thresh: {self.conf_threshold:.2f}"
                ]

                y_pos = 25
                for text in info_text:
                    cv2.putText(annotated_frame, text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    y_pos += 25

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.fps = 30 / elapsed if elapsed > 0 else 0.0
                    self.fps_start_time = time.time()

                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'

            except Exception as e:
                logger.error(f"Frame generation error: {e}")
                time.sleep(0.1)


# Global instance
object_tool = None


@app.route('/')
def index():
    """Main dashboard page."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Object Identification & Detection</title>
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
                --accent-primary: #3b82f6; /* Blue */
                --accent-secondary: #10b981; /* Emerald */
                --accent-warning: #f59e0b; /* Amber */
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
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            .control-group {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 15px;
                flex-wrap: wrap;
            }

            label {
                font-weight: 600;
                color: var(--text-main);
            }

            input[type=range] {
                flex-grow: 1;
                max-width: 300px;
                accent-color: var(--accent-primary);
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

            .detected-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }
            .tag {
                background: var(--bg-control);
                border: 1px solid var(--border-color);
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 14px;
                color: var(--accent-warning);
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📦 Live Object Identification & Detection</h1>

            <div id="message"></div>

            <div id="video-container">
                <img id="video-stream" src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Objects Detected</div>
                    <div class="stat-value" id="objects-count">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Detector Model</div>
                    <div class="stat-value" style="font-size: 18px;" id="model-type">{{ model_type }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fps">0</div>
                </div>
            </div>

            <div class="controls">
                <div class="control-group">
                    <button class="btn-outline" onclick="switchCamera()">🔄 Switch Camera</button>
                </div>

                <div class="control-group">
                    <label for="thresh-slider">Confidence Threshold: <span id="thresh-val">{{ threshold }}</span></label>
                    <input type="range" id="thresh-slider" min="0.05" max="0.95" step="0.05" value="{{ threshold }}" onchange="updateThreshold(this.value)" oninput="document.getElementById('thresh-val').textContent = parseFloat(this.value).toFixed(2)">
                </div>

                <div>
                    <label>Active Detections:</label>
                    <div class="detected-tags" id="detected-tags">
                        <span class="tag">None</span>
                    </div>
                </div>
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

            function updateThreshold(val) {
                fetch('/update_config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({threshold: parseFloat(val)})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage('Confidence threshold updated to ' + parseFloat(val).toFixed(2));
                    }
                });
            }

            function updateStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('objects-count').textContent = data.objects_detected;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);

                    const tagsDiv = document.getElementById('detected-tags');
                    tagsDiv.innerHTML = '';
                    const labels = data.detected_labels;
                    const keys = Object.keys(labels);
                    if (keys.length === 0) {
                        tagsDiv.innerHTML = '<span class="tag">None</span>';
                    } else {
                        keys.forEach(key => {
                            const tag = document.createElement('span');
                            tag.className = 'tag';
                            tag.textContent = `${key} (${labels[key]})`;
                            tagsDiv.appendChild(tag);
                        });
                    }
                });
            }

            setInterval(updateStats, 1500);
            updateStats();
        </script>
    </body>
    </html>
    """
    return render_template_string(
        html,
        model_type=f"{object_tool.cfg.vision.object_model_type} ({object_tool.cfg.vision.object_model_name})",
        threshold=f"{object_tool.conf_threshold:.2f}"
    )


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint."""
    return Response(
        object_tool.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/switch_camera')
def switch_camera():
    """Switch camera source."""
    if len(object_tool.cameras) > 1:
        next_idx = (object_tool.current_camera_idx + 1) % len(object_tool.cameras)
        object_tool.start_camera(next_idx)
        return jsonify({'success': True, 'message': f'Switched to {object_tool.cameras[next_idx]["name"]}'})
    elif object_tool.current_camera:
        return jsonify({'success': False, 'message': 'Only one camera source active'})
    return jsonify({'success': False, 'message': 'No active camera'})


@app.route('/update_config', methods=['POST'])
def update_config():
    """Update runtime configuration settings."""
    data = request.get_json() or {}
    if 'threshold' in data:
        try:
            thresh = float(data['threshold'])
            object_tool.set_threshold(thresh)
            return jsonify({'success': True, 'threshold': object_tool.conf_threshold})
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid threshold value'}), 400
    return jsonify({'success': False, 'error': 'No threshold parameter provided'}), 400


@app.route('/stats')
def stats():
    """Get live detection statistics."""
    with object_tool.lock:
        return jsonify({
            'objects_detected': object_tool.objects_detected,
            'detected_labels': object_tool.detected_labels,
            'fps': object_tool.fps
        })


if __name__ == '__main__':
    print("=" * 60)
    print("  Object Identification & Detection Tool - Web Interface")
    print("=" * 60)
    print("\nInitializing cameras...")

    object_tool = WebObjectIdentify()
    object_tool.start_camera(0)
    object_tool.running = True
    atexit.register(object_tool.stop)

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

    # Use port 5002 to avoid conflicting with face_capture_web (5000) & face_identify_web (5001)
    port = 5002
    print(f"  http://{ip_addr}:{port}")
    print(f"  or")
    print(f"  http://localhost:{port} (if accessing locally)")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        app.run(host='0.0.0.0', port=port, threaded=True)
    finally:
        object_tool.stop()
