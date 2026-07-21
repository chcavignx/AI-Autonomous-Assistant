#!/usr/bin/env python3
"""
Face Capture Tool - Web Interface Version
Access via browser: http://<raspberry-pi-ip>:5000
Works over SSH - no X11 forwarding needed
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, render_template, Response, request, jsonify
import cv2
import logging
import os
import time
from datetime import datetime
import numpy as np
import threading

from src.utils.config import config
from src.utils.camera import discover_pi_cameras
from src.vision.face_in_frame import FaceInFrame

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

app = Flask(__name__)

output_dir = config.vision.face_dataset_path
os.makedirs(output_dir, exist_ok=True)

class WebFaceCapture:
    def __init__(self):
        self.cfg = config
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None
        self.current_person_name = "unknown"
        self.capture_count = 0
        self.output_dir = output_dir
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.latest_frame = None
        self.faces_detected = 0
        self.running = False
        self.lock = threading.Lock()

        # Initialize configured modular face detector
        detector_type = getattr(self.cfg.vision, "face_detector_type", "cascade")
        self.face_processor = FaceInFrame(self.cfg, detector_type=detector_type)

        os.makedirs(self.output_dir, exist_ok=True)
        self.initialize_cameras()

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
            logger.warning("No cameras available. Camera streaming and capture will be disabled.")

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

    def detect_faces(self, frame):
        """Detect faces in the frame"""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, recognized = self.face_processor.process_frame(frame_bgr, draw=False)
        faces = [(f["box"][0], f["box"][1], f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) for f in recognized]
        return faces

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

                # Detect faces
                faces = self.detect_faces(frame)

                with self.lock:
                    self.faces_detected = len(faces)
                    self.latest_frame = frame.copy()

                # Draw faces and info
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, self.current_person_name, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Info overlay
                info_text = [
                    f"Camera: {self.current_camera['name']}",
                    f"Person: {self.current_person_name}",
                    f"Captured: {self.capture_count}",
                    f"Faces: {len(faces)}"
                ]

                y_pos = 25
                for text in info_text:
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.fps = 30 / elapsed
                    self.fps_start_time = time.time()

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # pyright: ignore[reportUnusedVariable]
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n' # pyright: ignore[reportImplicitStringConcatenation]
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(0.1)

    def capture_photo(self):
        """Capture current frame"""
        with self.lock:
            if self.latest_frame is None:
                return False, "No frame available"

            frame = self.latest_frame.copy()

        # Detect faces
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return False, "No face detected"

        # Save images
        person_dir = os.path.join(self.output_dir, self.current_person_name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Full frame
        full_path = os.path.join(person_dir, f"{timestamp}_full.jpg")
        cv2.imwrite(full_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Face crops
        for idx, (x, y, w, h) in enumerate(faces):
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_img = frame[y1:y2, x1:x2]
            face_path = os.path.join(person_dir, f"{timestamp}_face{idx}.jpg")
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        self.capture_count += 1
        return True, f"Captured {len(faces)} face(s)"

    def process_directory_dataset(self, src_dir: str, person_name: str) -> dict:
        """Process all images in a directory, detect faces and save crops for dataset."""
        if not os.path.isdir(src_dir):
            return {"success": False, "message": f"Source directory '{src_dir}' does not exist."}

        person_name = person_name.strip()
        if not person_name:
            return {"success": False, "message": "Person name cannot be empty."}

        target_dir = os.path.join(self.output_dir, person_name)
        os.makedirs(target_dir, exist_ok=True)

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = []
        for root, _, files in os.walk(src_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_exts:
                    image_files.append(os.path.join(root, file))

        if not image_files:
            return {"success": False, "message": f"No supported images found in '{src_dir}'."}

        processed_count = 0
        detected_faces_count = 0

        for img_path in image_files:
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                _, recognized = self.face_processor.process_frame(frame, draw=False)
                faces = [(f["box"][0], f["box"][1], f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) for f in recognized]

                if len(faces) == 0:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                # Save full frame
                full_path = os.path.join(target_dir, f"{timestamp}_{base_name}_full.jpg")
                cv2.imwrite(full_path, frame)

                # Save crops
                for idx, (x, y, w, h) in enumerate(faces):
                    margin = 20
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)

                    face_img = frame[y1:y2, x1:x2]
                    face_path = os.path.join(target_dir, f"{timestamp}_{base_name}_face{idx}.jpg")
                    cv2.imwrite(face_path, face_img)
                    detected_faces_count += 1

                processed_count += 1
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        return {
            "success": True,
            "message": f"Processed {processed_count}/{len(image_files)} images. Extracted {detected_faces_count} faces.",
            "processed_images": processed_count,
            "total_images": len(image_files),
            "faces_extracted": detected_faces_count
        }

    def process_directory_identify(self, src_dir: str) -> dict:
        """Identify faces in images in a directory using FaceInsightPipeline."""
        if not os.path.isdir(src_dir):
            return {"success": False, "message": f"Source directory '{src_dir}' does not exist."}

        # Check if we have registered faces in dataset
        if not os.path.exists(self.output_dir):
            return {"success": False, "message": "No registered face dataset found. Please capture/process some faces first."}

        # Use the recognizer from the unified FaceInFrame processor
        try:
            recognizer = self.face_processor.face_recognizer
            if not recognizer:
                from src.vision.face_insight_pipeline import FaceInsightPipeline
                recognizer = FaceInsightPipeline(self.cfg)
        except Exception as e:
            return {"success": False, "message": f"Failed to initialize FaceInsightPipeline: {e}"}

        # Scan face_dataset and register all known faces (using full frames for better context detection)
        registered_count = 0
        for person_name in os.listdir(self.output_dir):
            person_path = os.path.join(self.output_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            full_frames = [f for f in os.listdir(person_path) if "_full" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for idx, full_frame in enumerate(full_frames[:10]):  # Limit to 10 photos per person for registration speed
                img_path = os.path.join(person_path, full_frame)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        recognizer.register_face(f"{person_name}_{idx}", img)
                        registered_count += 1
                except Exception as e:
                    print(f"Failed to register face {img_path}: {e}")

        if registered_count == 0:
            return {"success": False, "message": "No face images found in the dataset folder to match against."}

        # Scan source directory for images to identify
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = []
        for root, _, files in os.walk(src_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_exts:
                    image_files.append(os.path.join(root, file))

        if not image_files:
            return {"success": False, "message": f"No supported images found in '{src_dir}' to identify."}

        results = []
        results_dir = os.path.join(GEN_DATA_DIR, "identification_results") # pyright: ignore[reportUndefinedVariable]
        os.makedirs(results_dir, exist_ok=True)

        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = recognizer.recognize(img, thresh=0.4)
                detected_names = []
                annotated_img = img.copy()

                for f in faces:
                    person_name = f.identity or "unknown"
                    best_sim = f.similarity or 0.0
                    bbox = [int(val) for val in f.bbox]
                    # Draw bbox
                    color = (0, 255, 0) if person_name != "unknown" else (0, 0, 255)
                    cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    label = f"{person_name} ({best_sim:.2f})"
                    cv2.putText(annotated_img, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    detected_names.append({"name": person_name, "score": best_sim})

                # Save annotated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(results_dir, f"identified_{timestamp}_{os.path.basename(img_path)}")
                cv2.imwrite(save_path, annotated_img)

                results.append({
                    "filename": os.path.basename(img_path),
                    "faces_detected": len(faces),
                    "identities": detected_names,
                    "annotated_path": os.path.relpath(save_path, GEN_DATA_DIR) # pyright: ignore[reportUndefinedVariable]
                })
            except Exception as e:
                print(f"Error identifying faces in {img_path}: {e}")

        return {
            "success": True,
        "message": f"Successfully processed {len(results)} images.",
            "results": results
        }

    def scan_directory_for_catalog(self, src_dir: str) -> dict:
        """Scan a directory for images containing faces and return candidates with parsed name suggestions."""
        if not os.path.isdir(src_dir):
            return {"success": False, "message": f"Source directory '{src_dir}' does not exist."}

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = []
        for root, _, files in os.walk(src_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_exts:
                    image_files.append(os.path.join(root, file))

        if not image_files:
            return {"success": False, "message": f"No supported images found in '{src_dir}'."}

        candidates = []
        for img_path in image_files:
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                _, recognized = self.face_processor.process_frame(frame, draw=False)
                faces = [(f["box"][0], f["box"][1], f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) for f in recognized]

                if len(faces) > 0:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    import re
                    # Remove numbers, underscores, and dashes, clean up spaces
                    suggestion = re.sub(r'[\d_-]+', ' ', base).strip()
                    suggestion = suggestion.title()
                    if not suggestion:
                        suggestion = "unknown"

                    candidates.append({
                        "path": img_path,
                        "filename": os.path.basename(img_path),
                        "suggestion": suggestion,
                        "faces_detected": len(faces)
                    })
            except Exception as e:
                print(f"Error scanning image {img_path}: {e}")

        return {
            "success": True,
            "message": f"Scanned {len(image_files)} images, found {len(candidates)} with faces.",
            "candidates": candidates
        }

    def save_cataloged_face(self, img_path: str, person_name: str) -> dict:
        """Process a single image path, crop detected faces, and save them under person_name."""
        if not os.path.exists(img_path):
            return {"success": False, "message": f"Image file '{img_path}' does not exist."}

        person_name = person_name.strip()
        if not person_name:
            return {"success": False, "message": "Person name cannot be empty."}

        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return {"success": False, "message": "Failed to read image."}

            _, recognized = self.face_processor.process_frame(frame, draw=False)
            faces = [(f["box"][0], f["box"][1], f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) for f in recognized]

            if len(faces) == 0:
                return {"success": False, "message": "No faces detected in the image."}

            target_dir = os.path.join(self.output_dir, person_name)
            os.makedirs(target_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            # Save full frame
            full_path = os.path.join(target_dir, f"{timestamp}_{base_name}_full.jpg")
            cv2.imwrite(full_path, frame)

            # Save crops
            for idx, (x, y, w, h) in enumerate(faces):
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_img = frame[y1:y2, x1:x2]
                face_path = os.path.join(target_dir, f"{timestamp}_{base_name}_face{idx}.jpg")
                cv2.imwrite(face_path, face_img)

            return {
                "success": True,
                "message": f"Successfully cataloged {len(faces)} face(s) for '{person_name}'."
            }
        except Exception as e:
            return {"success": False, "message": f"Error cataloging face: {e}"}

# Global instance
capture_tool = None

@app.route('/')
def index():
    """Main page"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>SOTA Face Capture & Identification</title>
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

            .tabs {
                display: flex;
                margin-bottom: 25px;
                background: var(--bg-card);
                padding: 6px;
                border-radius: 12px;
                border: 1px solid var(--border-color);
            }

            .tab-btn {
                flex: 1;
                background: transparent;
                color: var(--text-muted);
                border: none;
                padding: 14px 20px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }

            .tab-btn.active {
                background: linear-gradient(135deg, var(--accent-primary), #6d28d9);
                color: #fff;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25);
            }

            .tab-btn:hover:not(.active) {
                color: #fff;
                background: rgba(255, 255, 255, 0.05);
            }

            .tab-content {
                display: none;
                animation: fadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }

            .tab-content.active {
                display: block;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
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
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            }

            .control-group {
                margin-bottom: 20px;
            }

            .control-group:last-child {
                margin-bottom: 0;
            }

            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: var(--text-muted);
                font-size: 14px;
            }

            input[type="text"] {
                width: 100%;
                padding: 12px 16px;
                background: var(--bg-control);
                border: 1px solid var(--border-color);
                color: #fff;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
                font-family: inherit;
                transition: all 0.3s;
            }

            input[type="text"]:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15);
            }

            .btn-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 15px;
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

            button:active {
                transform: translateY(1px);
            }

            .btn-secondary {
                background: var(--accent-secondary);
            }

            .btn-accent {
                background: linear-gradient(135deg, var(--accent-primary), #6d28d9);
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
                animation: slideDown 0.3s ease;
            }

            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .success {
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid var(--accent-secondary);
                color: var(--accent-secondary);
            }

            .error {
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid var(--accent-danger);
                color: var(--accent-danger);
            }

            .instructions {
                background: var(--bg-card);
                padding: 25px;
                border-radius: 16px;
                border: 1px solid var(--border-color);
            }

            .instructions h3 {
                color: var(--accent-primary);
                margin-top: 0;
                font-size: 18px;
                margin-bottom: 15px;
            }

            .instructions ul {
                margin: 0;
                padding-left: 20px;
                line-height: 1.8;
                color: var(--text-muted);
            }

            /* Table styling */
            table {
                width: 100%;
                border-collapse: collapse;
                text-align: left;
                margin-top: 15px;
            }

            th, td {
                padding: 14px;
                border-bottom: 1px solid var(--border-color);
            }

            th {
                color: var(--accent-primary);
                font-weight: 600;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            td {
                font-size: 15px;
            }

            code {
                background: var(--bg-control);
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
                color: var(--accent-secondary);
            }

            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            .badge-success {
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent-secondary);
            }
            .badge-danger {
                background: rgba(239, 68, 68, 0.15);
                color: var(--accent-danger);
            }

            /* Interactive Cataloger styling */
            .catalog-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            }
            .catalog-card:hover {
                border-color: var(--accent-primary);
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(0,0,0,0.25);
            }
            .catalog-img-container {
                width: 100%;
                aspect-ratio: 4/3;
                overflow: hidden;
                background: #000;
                border-bottom: 1px solid var(--border-color);
            }
            .catalog-img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            .catalog-info {
                padding: 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .catalog-filename {
                font-size: 13px;
                color: var(--text-muted);
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .catalog-input-group {
                display: flex;
                gap: 6px;
            }
            .catalog-input-group input {
                flex: 1;
                padding: 8px;
                background: var(--bg-control);
                border: 1px solid var(--border-color);
                color: #fff;
                border-radius: 6px;
                font-size: 14px;
            }
            .catalog-input-group button {
                padding: 8px 12px;
                font-size: 14px;
            }
            .catalog-status {
                font-size: 12px;
                font-weight: 600;
                text-align: center;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 SOTA Face Capture & Identification</h1>

            <div class="tabs">
                <button class="tab-btn active" id="btn-camera" onclick="switchTab('camera')">
                    📹 Live Camera Capture
                </button>
                <button class="tab-btn" id="btn-directory" onclick="switchTab('directory')">
                    📁 Directory Processing
                </button>
            </div>

            <div id="message"></div>

            <!-- Tab content: Live Camera -->
            <div id="tab-camera" class="tab-content active">
                <div id="video-container">
                    <img id="video-stream" src="{{ url_for('video_feed') }}" alt="Video Stream">
                </div>

                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-label">Photos Captured</div>
                        <div class="stat-value" id="capture-count">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Faces Detected</div>
                        <div class="stat-value" id="faces-count">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Current Person</div>
                        <div class="stat-value" id="current-person">unknown</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">FPS</div>
                        <div class="stat-value" id="fps">0</div>
                    </div>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <label for="person-name">Person Name:</label>
                        <div style="display:flex; gap:10px;">
                            <input type="text" id="person-name" placeholder="Enter person name" value="unknown">
                            <button onclick="setPersonName()">Set Name</button>
                        </div>
                    </div>

                    <div class="btn-row">
                        <button class="btn-accent" onclick="capturePhoto()">📷 Capture Photo</button>
                        <button class="btn-outline" onclick="switchCamera()">🔄 Switch Camera</button>
                    </div>
                </div>

                <div class="instructions">
                    <h3>📝 Camera Instructions</h3>
                    <ul>
                        <li>Enter the person's name and click <strong>"Set Name"</strong></li>
                        <li>Position the person in front of the camera</li>
                        <li>Wait for face detection (green rectangle)</li>
                        <li>Click <strong>"Capture Photo"</strong> to save images</li>
                        <li>Photos are saved to <code>face_dataset/&lt;person_name&gt;/</code></li>
                    </ul>
                </div>
            </div>

            <!-- Tab content: Directory Processing -->
            <div id="tab-directory" class="tab-content">
                <div class="controls">
                    <h3>📁 Process Images from Directory</h3>

                    <div class="control-group">
                        <label for="dir-src-path">Source Directory Path:</label>
                        <input type="text" id="dir-src-path" placeholder="e.g. /home/cca/my_photos">
                    </div>

                    <div class="control-group">
                        <label for="dir-person-name">Person Name (only for dataset builder):</label>
                        <input type="text" id="dir-person-name" placeholder="Enter person name">
                    </div>

                    <div class="btn-row">
                        <button class="btn-secondary" onclick="runBatchDataset()">🛠️ Build Dataset (Crop Faces)</button>
                        <button class="btn-accent" onclick="runBatchIdentify()">🔍 Identify Faces (Recognition)</button>
                        <button class="btn-outline" onclick="runInteractiveCataloger()">🏷️ Interactive Database Cataloger</button>
                    </div>
                </div>

                <div class="instructions" id="dir-results-container" style="display: none; margin-bottom: 25px;">
                    <h3>📊 Directory Processing Results</h3>
                    <div id="dir-results-summary" style="font-weight: 600; color: var(--accent-secondary); margin-bottom: 15px;"></div>
                    <div style="max-height: 400px; overflow-y: auto;">
                        <table id="dir-results-table">
                            <thead>
                                <tr>
                                    <th>Image Name</th>
                                    <th>Faces</th>
                                    <th>Identified People / Matches</th>
                                    <th>Annotated Output</th>
                                </tr>
                            </thead>
                            <tbody id="dir-results-body"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Interactive Cataloger Grid Section -->
                <div class="instructions" id="catalog-container" style="display: none; margin-bottom: 25px;">
                    <h3>🏷️ Interactive Database Cataloger</h3>
                    <div id="catalog-summary" style="font-weight: 600; color: var(--accent-secondary); margin-bottom: 15px;"></div>

                    <div class="btn-row" style="margin-bottom: 20px;">
                        <button class="btn-secondary" onclick="saveAllCataloged()">💾 Save All Pending Faces</button>
                    </div>

                    <div id="catalog-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; max-height: 500px; overflow-y: auto; padding: 10px;">
                        <!-- Dynamically filled with catalog cards -->
                    </div>
                </div>

                <div class="instructions">
                    <h3>📝 Directory Instructions</h3>
                    <ul>
                        <li><strong>Process Directory</strong> mode: Scans a folder of images, detects faces, crops them, and registers them under a single person's name. Good for processing large datasets in bulk.</li>
                        <li><strong>Identify Faces</strong> mode: Loads registered faces from <code>face_dataset/</code>, runs SOTA <code>FaceInsightPipeline</code> face recognition on files in the target folder, and identifies the individuals.</li>
                        <li><strong>Interactive Database Cataloger</strong> mode: Scans the folder for images with faces, renders them in the UI, parses suggested names from the filenames, and lets you name/save them individually or all at once.</li>
                        <li>Annotated outputs for the identified faces are saved to <code>data/identification_results/</code>.</li>
                    </ul>
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

            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

                document.getElementById('tab-' + tab).classList.add('active');
                document.getElementById('btn-' + tab).classList.add('active');
            }

            function setPersonName() {
                const name = document.getElementById('person-name').value.trim();
                if (!name) {
                    showMessage('Please enter a valid name', true);
                    return;
                }

                fetch('/set_person', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: name})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage('Person set to: ' + name);
                        document.getElementById('current-person').textContent = name;
                    }
                });
            }

            function capturePhoto() {
                fetch('/capture')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(data.message);
                        updateStats();
                    } else {
                        showMessage(data.message, true);
                    }
                });
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
                    document.getElementById('capture-count').textContent = data.capture_count;
                    document.getElementById('faces-count').textContent = data.faces_detected;
                    document.getElementById('current-person').textContent = data.current_person;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                });
            }

            function runBatchDataset() {
                const srcDir = document.getElementById('dir-src-path').value.trim();
                const name = document.getElementById('dir-person-name').value.trim();

                if (!srcDir || !name) {
                    showMessage('Please enter both source directory and person name', true);
                    return;
                }

                showMessage('Running batch dataset extraction... Please wait.');

                fetch('/process_directory_dataset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({src_dir: srcDir, name: name})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(data.message);
                        document.getElementById('dir-results-container').style.display = 'block';
                        document.getElementById('dir-results-summary').textContent = data.message;
                        document.getElementById('dir-results-body').innerHTML = `
                            <tr>
                                <td colspan="4" style="text-align: center; color: var(--text-muted);">
                                    Dataset crops successfully saved under <code>face_dataset/${name}/</code>
                                </td>
                            </tr>
                        `;
                        updateStats();
                    } else {
                        showMessage(data.message, true);
                    }
                })
                .catch(err => showMessage('Error: ' + err, true));
            }

            function runBatchIdentify() {
                const srcDir = document.getElementById('dir-src-path').value.trim();

                if (!srcDir) {
                    showMessage('Please enter a source directory', true);
                    return;
                }

                showMessage('Initializing InsightFace & scanning directory... (this can take a few seconds)');

                fetch('/process_directory_identify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({src_dir: srcDir})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage('Processing complete!');
                        document.getElementById('dir-results-container').style.display = 'block';
                        document.getElementById('catalog-container').style.display = 'none';
                        document.getElementById('dir-results-summary').textContent = data.message;

                        const tbody = document.getElementById('dir-results-body');
                        tbody.innerHTML = '';

                        data.results.forEach(res => {
                            const tr = document.createElement('tr');

                            let idText = '';
                            if (res.identities.length === 0) {
                                idText = '<span class="badge badge-danger">No face detected</span>';
                            } else {
                                idText = res.identities.map(id => {
                                    const badgeClass = id.name === 'unknown' ? 'badge-danger' : 'badge-success';
                                    return `<span class="badge ${badgeClass}">${id.name} (${(id.score * 100).toFixed(0)}%)</span>`;
                                }).join(' ');
                            }

                            tr.innerHTML = `
                                <td>${res.filename}</td>
                                <td>${res.faces_detected}</td>
                                <td>${idText}</td>
                                <td><code>data/${res.annotated_path}</code></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    } else {
                        showMessage(data.message, true);
                    }
                })
                .catch(err => showMessage('Error: ' + err, true));
            }

            let catalogCandidates = [];

            function runInteractiveCataloger() {
                const srcDir = document.getElementById('dir-src-path').value.trim();

                if (!srcDir) {
                    showMessage('Please enter a source directory', true);
                    return;
                }

                showMessage('Scanning directory for images with faces...');

                fetch('/scan_directory_catalog', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({src_dir: srcDir})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(data.message);
                        catalogCandidates = data.candidates;

                        document.getElementById('catalog-container').style.display = 'block';
                        document.getElementById('dir-results-container').style.display = 'none';
                        document.getElementById('catalog-summary').textContent = data.message;

                        const grid = document.getElementById('catalog-grid');
                        grid.innerHTML = '';

                        if (catalogCandidates.length === 0) {
                            grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 20px;">No images with faces found.</div>';
                            return;
                        }

                        catalogCandidates.forEach((cand, idx) => {
                            const card = document.createElement('div');
                            card.className = 'catalog-card';
                            card.id = 'catalog-card-' + idx;

                            card.innerHTML = `
                                <div class="catalog-img-container">
                                    <img class="catalog-img" src="/get_local_image?path=${encodeURIComponent(cand.path)}" alt="Face Candidate">
                                </div>
                                <div class="catalog-info">
                                    <div class="catalog-filename" title="${cand.filename}">${cand.filename}</div>
                                    <div style="font-size:12px; color:var(--accent-secondary); margin-bottom:4px;">Faces: ${cand.faces_detected}</div>
                                    <div class="catalog-input-group">
                                        <input type="text" id="catalog-input-${idx}" value="${cand.suggestion}" placeholder="Person name">
                                        <button class="btn-accent" onclick="saveCatalogedFace(${idx})">Save</button>
                                    </div>
                                    <div class="catalog-status" id="catalog-status-${idx}"></div>
                                </div>
                            `;
                            grid.appendChild(card);
                        });
                    } else {
                        showMessage(data.message, true);
                    }
                })
                .catch(err => showMessage('Error: ' + err, true));
            }

            function saveCatalogedFace(idx) {
                const cand = catalogCandidates[idx];
                const name = document.getElementById('catalog-input-' + idx).value.trim();

                if (!name) {
                    showMessage('Please enter a name for the face', true);
                    return;
                }

                const statusDiv = document.getElementById('catalog-status-' + idx);
                statusDiv.style.display = 'block';
                statusDiv.style.color = 'var(--text-muted)';
                statusDiv.textContent = 'Saving...';

                fetch('/save_catalog_face', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image_path: cand.path, name: name})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusDiv.style.color = 'var(--accent-secondary)';
                        statusDiv.textContent = '✓ Saved!';
                        setTimeout(() => {
                            const card = document.getElementById('catalog-card-' + idx);
                            if (card) {
                                card.style.opacity = '0.3';
                                card.querySelectorAll('input, button').forEach(el => el.disabled = true);
                            }
                        }, 1000);
                        updateStats();
                    } else {
                        statusDiv.style.color = 'var(--accent-danger)';
                        statusDiv.textContent = 'Error: ' + data.message;
                    }
                })
                .catch(err => {
                    statusDiv.style.color = 'var(--accent-danger)';
                    statusDiv.textContent = 'Error: ' + err;
                });
            }

            function saveAllCataloged() {
                const cards = document.querySelectorAll('.catalog-card');
                let count = 0;
                cards.forEach((card, idx) => {
                    const input = document.getElementById('catalog-input-' + idx);
                    if (input && !input.disabled) {
                        setTimeout(() => {
                            saveCatalogedFace(idx);
                        }, count * 200);
                        count++;
                    }
                });
                if (count === 0) {
                    showMessage('No pending faces to save.', true);
                } else {
                    showMessage(`Saving all ${count} pending faces...`);
                }
            }

            // Update stats every 2 seconds
            setInterval(updateStats, 2000);
            updateStats();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/process_directory_dataset', methods=['POST'])
def process_directory_dataset():
    """Batch process a directory to build dataset (crop faces)"""
    data = request.json
    src_dir = data.get('src_dir', '').strip()
    name = data.get('name', '').strip()

    if not src_dir or not name:
        return jsonify({'success': False, 'message': 'Missing directory path or person name.'})

    result = capture_tool.process_directory_dataset(src_dir, name)
    return jsonify(result)

@app.route('/process_directory_identify', methods=['POST'])
def process_directory_identify():
    """Batch process a directory to recognize/identify faces"""
    data = request.json
    src_dir = data.get('src_dir', '').strip()

    if not src_dir:
        return jsonify({'success': False, 'message': 'Missing directory path.'})

    result = capture_tool.process_directory_identify(src_dir)
    return jsonify(result)

@app.route('/get_local_image')
def get_local_image():
    """Route to serve local image preview safely"""
    path = request.args.get('path', '')
    if not path or not os.path.exists(path):
        return "Image not found", 404

    # Ensure it's a valid image file type
    ext = os.path.splitext(path)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.bmp'}:
        return "Forbidden file type", 403

    from flask import send_file
    return send_file(path)

@app.route('/scan_directory_catalog', methods=['POST'])
def scan_directory_catalog():
    """Route to scan directory for photos containing faces"""
    data = request.json
    src_dir = data.get('src_dir', '').strip()
    if not src_dir:
        return jsonify({'success': False, 'message': 'Missing directory path.'})

    result = capture_tool.scan_directory_for_catalog(src_dir)
    return jsonify(result)

@app.route('/save_catalog_face', methods=['POST'])
def save_catalog_face():
    """Route to crop and save a face to the dataset database under a specific name"""
    data = request.json
    img_path = data.get('image_path', '').strip()
    name = data.get('name', '').strip()

    if not img_path or not name:
        return jsonify({'success': False, 'message': 'Missing image path or name.'})

    result = capture_tool.save_cataloged_face(img_path, name)
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(capture_tool.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    """Capture photo endpoint"""
    success, message = capture_tool.capture_photo()
    return jsonify({'success': success, 'message': message})

@app.route('/set_person', methods=['POST'])
def set_person():
    """Set current person name"""
    data = request.json
    name = data.get('name', '').strip()
    if name:
        capture_tool.current_person_name = name
        return jsonify({'success': True, 'name': name})
    return jsonify({'success': False})

@app.route('/switch_camera')
def switch_camera():
    """Switch to next camera"""
    if len(capture_tool.cameras) > 1:
        next_idx = (capture_tool.current_camera_idx + 1) % len(capture_tool.cameras)
        capture_tool.current_camera_idx = next_idx
        capture_tool.start_camera(next_idx)
        return jsonify({'success': True, 'message': f'Switched to {capture_tool.cameras[next_idx]["name"]}'})
    return jsonify({'success': False, 'message': 'Only one camera available'})

@app.route('/stats')
def stats():
    """Get current statistics"""
    return jsonify({
        'capture_count': capture_tool.capture_count,
        'faces_detected': capture_tool.faces_detected,
        'current_person': capture_tool.current_person_name,
        'fps': capture_tool.fps
    })

if __name__ == '__main__':
    print("="*60)
    print("  Face Capture Tool - Web Interface")
    print("="*60)
    print("\nInitializing cameras...")

    capture_tool = WebFaceCapture()
    capture_tool.start_camera(0)
    capture_tool.running = True

    print("\nâœ“ Camera initialized")
    print("\nAccess the interface at:")
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
        s.close()
    except Exception:
        ip_addr = "<raspberry-pi-ip>"

    print(f"  http://{ip_addr}:5000")
    print("  or")
    print("  http://localhost:5000 (if accessing locally)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, threaded=True)
