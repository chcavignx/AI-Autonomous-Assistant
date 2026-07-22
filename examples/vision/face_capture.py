#!/usr/bin/env python3
"""
Dual Camera Face Recognition Capture System
Displays live video feed and captures photos for face recognition training
Supports both IMX219 and AI Camera on Raspberry Pi 5

Controls:
- SPACE: Capture photo for current person
- 'n': Enter new person name
- 's': Switch between cameras
- 'q': Quit application
- 'h': Show help
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import logging
import os
import time
from datetime import datetime

from src.utils.config import load_config
from src.vision.face_in_frame import FaceInFrame
from src.utils.camera import discover_pi_cameras

GEN_DATA_DIR = str(project_root / "data")
OUTPUT_DIR = "face_dataset"
output_dir = str(project_root / "data" / OUTPUT_DIR)

# configurations logger
app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

class FaceCaptureTool:
    def __init__(self):
        """Initialize the face capture tool"""
        self.cfg = load_config()
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None
        self.current_person_name = "unknown"
        self.capture_count = 0
        self.output_dir = self.cfg.vision.face_dataset_path or output_dir
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.margin = getattr(self.cfg.vision, "face_capture_margin", 20)
        self.fps_update_interval = getattr(self.cfg.vision, "fps_update_interval", 30)

        # Initialize configured modular face detector
        detector_type = getattr(self.cfg.vision, "face_detector_type", "cascade")
        self.face_processor = FaceInFrame(self.cfg, detector_type=detector_type)

        # Initialize cameras
        self.initialize_cameras()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_cameras(self):
        """Initialize available cameras"""
        print("Initializing cameras...")

        for cam in discover_pi_cameras():
            name = cam.camera_info.get("Model", f"Camera {cam.index}")
            self.cameras.append(
                {
                    "index": cam.index,
                    "name": name,
                    "camera": cam,
                    "info": cam.get_info_str(),
                }
            )
            print(f"✓ Camera {cam.index} detected: {cam.get_info_str()}")

        if not self.cameras:
            raise RuntimeError("No cameras available! Check connections.")

        print(f"\n{len(self.cameras)} camera(s) available")

    def start_camera(self, camera_idx=0):
        """Start the selected camera"""
        if not self.cameras:
            return

        if camera_idx >= len(self.cameras):
            camera_idx = 0

        # Stop previous camera if running
        if self.current_camera:
            try:
                self.current_camera["camera"].stop()
            except Exception:
                pass

        self.current_camera_idx = camera_idx
        self.current_camera = self.cameras[camera_idx]

        if not self.current_camera:
            return

        # Configure camera for video streaming
        self.current_camera["camera"].start(
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

    def switch_camera(self):
        """Switch to next available camera"""
        next_idx = (self.current_camera_idx + 1) % len(self.cameras)
        self.start_camera(next_idx)

    def detect_faces(self, frame):
        """Detect faces in the frame"""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, recognized = self.face_processor.process_frame(frame_bgr, draw=False)
        faces = [(f["box"][0], f["box"][1], f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) for f in recognized]
        return faces

    def draw_interface(self, frame, faces):
        """Draw UI elements on the frame"""
        h, w = frame.shape[:2]

        # Draw faces
        for x, y, w_face, h_face in faces:
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
            cv2.putText(
                frame,
                self.current_person_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Info panel background
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 255, 0), 2)

        # Camera info
        cv2.putText(
            frame,
            f"Camera: {self.current_camera['name'] if self.current_camera else 'Unknown'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Current person
        cv2.putText(
            frame,
            f"Person: {self.current_person_name}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Capture count
        cv2.putText(
            frame,
            f"Captured: {self.capture_count} photos",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Faces detected
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (w - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Controls help (bottom)
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "SPACE: Capture | N: New person | S: Switch cam | Q: Quit | H: Help",
            (10, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    def capture_face(self, frame, faces):
        """Capture and save face images"""
        if len(faces) == 0:
            print("[!]  No face detected in frame")
            return False

        # Create person directory
        person_dir = os.path.join(self.output_dir, self.current_person_name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save full frame
        full_filename = os.path.join(person_dir, f"{timestamp}_full.jpg")
        cv2.imwrite(full_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Save each detected face
        for idx, (x, y, w, h) in enumerate(faces):
            # Add margin around face
            margin = self.margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_img = frame[y1:y2, x1:x2]

            face_filename = os.path.join(person_dir, f"{timestamp}_face{idx}.jpg")
            cv2.imwrite(face_filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        self.capture_count += 1
        print(
            f"[OK] Captured {len(faces)} face(s) for '{self.current_person_name}' (Total: {self.capture_count})"
        )

        return True

    def show_help(self):
        """Display help information"""
        help_text = """
+â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•+
|          FACE RECOGNITION CAPTURE TOOL - HELP              |
-â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•+
|                                                            |
|  KEYBOARD CONTROLS:                                        |
|  -----------------                                         |
|  SPACE    : Capture photo of detected face(s)             |
|  N        : Enter new person name                          |
|  S        : Switch between available cameras               |
|  Q        : Quit application                               |
|  H        : Show this help                                 |
|                                                            |
|  USAGE:                                                    |
|  -----                                                     |
|  1. Press 'N' to enter the person's name                   |
|  2. Position the person in front of the camera             |
|  3. Press SPACE to capture multiple photos                 |
|  4. Capture 10-20 photos with different angles/expressions|
|  5. Repeat for additional people                           |
|                                                            |
|  TIPS:                                                     |
|  ----                                                      |
|  - Ensure good lighting                                    |
|  - Capture various angles (front, slight left/right)       |
|  - Include different expressions                           |
|  - Maintain consistent distance (~50cm to 1m)              |
|  - Green rectangle shows detected face                     |
|                                                            |
|  OUTPUT:                                                   |
|  ------                                                    |
|  Photos saved to: face_dataset/<person_name>/              |
|  - Full frame images: *_full.jpg                           |
|  - Cropped faces: *_face0.jpg, *_face1.jpg, etc.          |
|                                                            |
+â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•+
        """
        print(help_text)

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= self.fps_update_interval:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def run(self):
        """Main application loop"""
        print("\n" + "=" * 60)
        print("  FACE RECOGNITION CAPTURE TOOL")
        print("=" * 60)
        print("\nPress 'H' at any time to show help")
        print("Press 'N' to enter person name before capturing")
        print("\nStarting camera...")

        # Start configured camera
        self.start_camera(self.cfg.vision.camera.camera_index)

        # Show initial help
        time.sleep(1)
        self.show_help()

        print("\n[OK] Ready! Camera window opened.\n")

        try:
            while True:
                # Capture frame
                if not self.current_camera:
                    print("Error: No active camera.")
                    break
                frame = self.current_camera["camera"].capture_frame()

                # Detect faces
                faces = self.detect_faces(frame)

                # Draw interface
                display_frame = self.draw_interface(frame.copy(), faces)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow("Face Capture Tool", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nQuitting...")
                    break

                elif key == ord(" "):  # Space bar
                    self.capture_face(frame, faces)

                elif key == ord("n") or key == ord("N"):
                    # Get new person name without blocking the video feed
                    if not hasattr(self, "input_thread") or not self.input_thread.is_alive():
                        import threading
                        def get_name():
                            print("\n" + "-" * 40)
                            name = input("Enter person name: ").strip()
                            if name:
                                self.current_person_name = name
                                print(f"[OK] Person set to: {self.current_person_name}")
                            else:
                                print("[!] Name cannot be empty")
                            print("-" * 40 + "\n")

                        self.input_thread = threading.Thread(target=get_name, daemon=True)
                        self.input_thread.start()

                elif key == ord("s") or key == ord("S"):
                    if len(self.cameras) > 1:
                        self.switch_camera()
                    else:
                        print("[!]  Only one camera available")

                elif key == ord("h") or key == ord("H"):
                    self.show_help()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            print("\nCleaning up...")
            cv2.destroyAllWindows()

            if self.current_camera:
                try:
                    self.current_camera["camera"].stop()
                    self.current_camera["camera"].close()
                except Exception:
                    pass

            # Print summary
            print("\n" + "=" * 60)
            print("  CAPTURE SESSION SUMMARY")
            print("=" * 60)
            print(f"Total photos captured: {self.capture_count}")
            print(f"Output directory: {os.path.abspath(self.output_dir)}")

            # List captured persons
            if os.path.exists(self.output_dir):
                persons = [
                    d
                    for d in os.listdir(self.output_dir)
                    if os.path.isdir(os.path.join(self.output_dir, d))
                ]

                if persons:
                    print(f"\nCaptured faces for {len(persons)} person(s):")
                    for person in persons:
                        person_path = os.path.join(self.output_dir, person)
                        photo_count = len(
                            [f for f in os.listdir(person_path) if f.endswith(".jpg")]
                        )
                        print(f"  - {person}: {photo_count} photos")

            print("\n[OK] Session complete!")
            print("=" * 60)


def main():
    """Main entry point"""
    try:
        tool = FaceCaptureTool()
        tool.run()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nMake sure:")
        print("  - Cameras are properly connected")
        print("  - Camera overlays are configured in /boot/firmware/config.txt")
        print("  - System has been rebooted after configuration")
        print("  - Run 'python3 dual_camera_verification.py' to check setup")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
