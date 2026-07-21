#!/usr/bin/env python3
"""
Face Capture Tool - Headless/Terminal Version
Works entirely in SSH terminal without GUI
Uses keyboard input and saves frames automatically
"""

import cv2
import logging
import os
import time
import sys
import select
import termios
import tty
from datetime import datetime
from src.utils.camera import discover_pi_cameras

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.vision.face_detector_cascade import CascadeFaceDetector
from src.vision.yolo_cpu import YoloCpuDetector
from src.vision.face_detector import InsightFaceDetector

GEN_DATA_DIR  =  str(project_root / "data")
OUTPUT_DIR = "face_dataset"
# Target directory
output_dir = os.path.join(GEN_DATA_DIR, OUTPUT_DIR)


app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

class USBCameraWrapper:
    def __init__(self, index):
        self.index = index
        self.cap = None
        self.camera_info = {"Model": f"USB Webcam {index}"}

    def start(self, width, height, format):
        import cv2
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_frame(self):
        if not self.cap: return None
        ret, frame = self.cap.read()
        if ret:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_info_str(self):
        return f"Standard USB Webcam (Index {self.index})"

    def stop(self):
        if self.cap:
            self.cap.release()

    def close(self):
        self.stop()


class HeadlessFaceCapture:
    def __init__(self):
        self.cfg = load_config()
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None
        self.current_person_name = "unknown"
        self.capture_count = 0
        self.output_dir = output_dir
        self.running = False
        self.auto_capture_mode = False
        self.auto_capture_interval = getattr(self.cfg.vision, "auto_capture_interval", 2.0)  # seconds
        self.last_auto_capture = 0
        self.margin = getattr(self.cfg.vision, "face_capture_margin", 20)

        # Initialize configured modular face detector
        detector_type = getattr(self.cfg.vision, "face_detector_type", "cascade")
        if detector_type == "cascade":
            self.face_detector = CascadeFaceDetector(self.cfg)
        elif detector_type == "yolo":
            self.face_detector = YoloCpuDetector(self.cfg)
        elif detector_type == "insightface":
            self.face_detector = InsightFaceDetector(self.cfg)
        else:
            self.face_detector = CascadeFaceDetector(self.cfg)

        os.makedirs(self.output_dir, exist_ok=True)
        self.initialize_cameras()

    def initialize_cameras(self):
        """Initialize available cameras with fallback to standard USB webcams"""
        print("Initializing cameras...")
        for cam in discover_pi_cameras():
            name = getattr(cam, "camera_info", {}).get("Model", f"Camera {cam.index}")
            self.cameras.append({
                'index': cam.index,
                'name': name,
                'camera': cam,
                'info': cam.get_info_str()
            })
            print(f"✓ Camera {cam.index} detected: {cam.get_info_str()}")

        # Fallback to standard USB camera if no Pi cameras are present
        if not self.cameras:
            import cv2
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                test_cap.release()
                usb_cam = USBCameraWrapper(0)
                self.cameras.append({
                    'index': 0,
                    'name': usb_cam.camera_info["Model"],
                    'camera': usb_cam,
                    'info': usb_cam.get_info_str()
                })
                print(f"✓ USB Camera 0 detected")

        if not self.cameras:
            raise RuntimeError("No cameras available (Pi or USB)")

        print(f"\nFound {len(self.cameras)} camera(s)")

    def start_camera(self, idx=0):
        """Start selected camera"""
        if self.current_camera:
            try:
                self.current_camera['camera'].stop()
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")

        self.current_camera = self.cameras[idx]
        self.current_camera_idx = idx

        self.current_camera['camera'].start(
            self.cfg.vision.camera.frame_width,
            self.cfg.vision.camera.frame_height,
            self.cfg.vision.camera.format
        )
        print(f"Started: {self.current_camera['name']}")
        # Allow camera to stabilize and wait for first valid frame (up to 2 seconds)
        for _ in range(20):
            try:
                frame = self.current_camera["camera"].capture_frame()
                if frame is not None and frame.size > 0:
                    break
            except Exception:
                pass
            time.sleep(0.1)

    def detect_and_capture(self):
        """Detect faces and capture if found"""
        if not self.current_camera or not self.current_camera.get('camera'):
            return False, 0

        frame = self.current_camera['camera'].capture_frame()

        # Detect faces
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detections = self.face_detector.detect(frame_bgr)
        faces = [(det["box"][0], det["box"][1], det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]) for det in detections]

        if len(faces) == 0:
            return False, 0

        # Save images
        person_dir = os.path.join(self.output_dir, self.current_person_name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Full frame
        full_path = os.path.join(person_dir, f"{timestamp}_full.jpg")
        cv2.imwrite(full_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Face crops
        for idx, (x, y, w, h) in enumerate(faces):
            margin = self.margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_img = frame[y1:y2, x1:x2]
            face_path = os.path.join(person_dir, f"{timestamp}_face{idx}.jpg")
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        self.capture_count += 1
        return True, len(faces)

    def get_key_press(self):
        """Non-blocking keyboard input"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def print_status(self):
        """Print current status"""
        os.system('clear' if os.name == 'posix' else 'cls') # pyright: ignore[reportDeprecated]
        print("="*70)
        print("  FACE CAPTURE TOOL - HEADLESS MODE")
        print("="*70)
        camera_name = self.current_camera['name'] if self.current_camera else "None"
        print(f"\nCamera: {camera_name}")
        print(f"Person: {self.current_person_name}")
        print(f"Captured: {self.capture_count} photos")
        print(f"Auto-capture: {'ON' if self.auto_capture_mode else 'OFF'}")
        print(f"Output: {os.path.abspath(self.output_dir)}")
        print("\n" + "="*70)
        print("COMMANDS:")
        print("  SPACE  - Capture photo now")
        print("  a      - Toggle auto-capture mode (every 2 seconds)")
        print("  n      - Change person name")
        print("  s      - Switch camera")
        print("  i      - Show info")
        print("  q      - Quit")
        print("="*70)

        if self.auto_capture_mode:
            print("\n[!]  AUTO-CAPTURE MODE ACTIVE - capturing every 2 seconds...")

    def run(self):
        """Main loop"""
        # Set terminal to raw mode for character input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            self.start_camera(0)
            self.running = True
            self.print_status()

            while self.running:
                # Check for keyboard input
                key = self.get_key_press()

                if key == 'q':
                    print("\n\nQuitting...")
                    break

                elif key == ' ':  # Space
                    print("\nCapturing...", end='', flush=True)
                    success, num_faces = self.detect_and_capture()
                    if success:
                        print(f" [OK] Captured {num_faces} face(s) (Total: {self.capture_count})")
                    else:
                        print(" [ERROR] No face detected")
                    time.sleep(0.5)

                elif key == 'a':
                    self.auto_capture_mode = not self.auto_capture_mode
                    self.print_status()

                elif key == 'n':
                    # Restore terminal to get input
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    print("\n" + "-"*50)
                    name = input("Enter person name: ").strip()
                    if name:
                        self.current_person_name = name
                        print(f"[OK] Person set to: {name}")
                    else:
                        print("[!]  Name cannot be empty")
                    print("-"*50)
                    input("Press Enter to continue...")
                    tty.setcbreak(sys.stdin.fileno())
                    self.print_status()

                elif key == 's':
                    if len(self.cameras) > 1:
                        next_idx = (self.current_camera_idx + 1) % len(self.cameras)
                        self.start_camera(next_idx)
                        self.print_status()
                    else:
                        print("\n[!]  Only one camera available")
                        time.sleep(1)

                elif key == 'i':
                    self.print_status()

                # Auto-capture logic
                if self.auto_capture_mode:
                    current_time = time.time()
                    if current_time - self.last_auto_capture >= self.auto_capture_interval:
                        success, num_faces = self.detect_and_capture()
                        if success:
                            print(f"\r[AUTO] Captured {num_faces} face(s) - Total: {self.capture_count}      ", end='', flush=True)
                        self.last_auto_capture = current_time

                time.sleep(0.1)

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            # Cleanup
            if self.current_camera:
                try:
                    self.current_camera['camera'].stop()
                    self.current_camera['camera'].close()
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")

            # Print summary
            print("\n\n" + "="*70)
            print("  CAPTURE SESSION SUMMARY")
            print("="*70)
            print(f"Total photos captured: {self.capture_count}")
            print(f"Output directory: {os.path.abspath(self.output_dir)}")

            if os.path.exists(self.output_dir):
                persons = [d for d in os.listdir(self.output_dir)
                          if os.path.isdir(os.path.join(self.output_dir, d))]

                if persons:
                    print(f"\nCaptured faces for {len(persons)} person(s):")
                    for person in persons:
                        person_path = os.path.join(self.output_dir, person)
                        photo_count = len([f for f in os.listdir(person_path)
                                         if f.endswith('.jpg')])
                        print(f"  - {person}: {photo_count} photos")

            print("\n[OK] Session complete!")
            print("="*70)

def main():
    try:
        tool = HeadlessFaceCapture()
        tool.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure cameras are properly configured and connected.")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
