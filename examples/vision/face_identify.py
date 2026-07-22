#!/usr/bin/env python3
"""
Dual Camera Face Recognition Identify System
Displays live video feed and identifies faces using the captured dataset
Supports both IMX219 and AI Camera on Raspberry Pi 5

Controls:
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
import numpy as np # pyright: ignore[reportUnusedImport]

from src.utils.sysutils import detect_raspberry_pi_model # pyright: ignore[reportUnusedImport]
from src.utils.config import load_config
from src.vision.face_insight_pipeline import FaceInsightPipeline
from src.utils.camera import discover_pi_cameras

OUTPUT_DIR = "face_dataset"
output_dir = str(project_root / "data" / OUTPUT_DIR)

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

class FaceIdentifyTool:
    def __init__(self):
        """Initialize the face identify tool"""
        self.cfg = load_config()
        self.cameras = []
        self.current_camera_idx = 0
        self.current_camera = None

        # Load dataset path from config or fallback
        # Note: self.cfg.vision.face_dataset_path might be relative or absolute. Handle gracefully.
        if hasattr(self.cfg.vision, "face_dataset_path") and self.cfg.vision.face_dataset_path:
            self.dataset_dir = self.cfg.vision.face_dataset_path
        else:
            self.dataset_dir = output_dir

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Initialize recognizer pipeline
        self.recognizer = FaceInsightPipeline(self.cfg)
        self.registered_faces = 0

        # Load known faces
        self.load_dataset()

        # Initialize available cameras
        self.initialize_cameras()

    def load_dataset(self):
        """Load known faces from dataset directory"""
        print(f"Loading dataset from: {self.dataset_dir}")
        if not os.path.exists(self.dataset_dir):
            print(f"Warning: Dataset directory {self.dataset_dir} does not exist.")
            print("Please run face_capture.py first to register faces.")
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
                        # Append index to ID to allow multiple registrations for same person
                        face_id = f"{person_name}_{idx}"
                        self.recognizer.register_face(face_id, img)
                        self.registered_faces += 1
                except Exception as e:
                    print(f"Failed to register face {img_path}: {e}")

        print(f"Loaded {self.registered_faces} face templates.")

    def initialize_cameras(self):
        """Initialize available cameras"""
        print("Initializing cameras...")

        for cam in discover_pi_cameras():
            name = "IMX219 (Camera V2)" if cam.index == 0 else "AI Camera (IMX500)"
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
        if camera_idx >= len(self.cameras):
            camera_idx = 0

        # Stop previous camera if running
        if self.current_camera:
            try:
                self.current_camera["camera"].stop()
            except:
                pass

        self.current_camera_idx = camera_idx
        self.current_camera = self.cameras[camera_idx]

        # Configure camera for video streaming
        self.current_camera["camera"].start(
            self.cfg.vision.camera.frame_width,
            self.cfg.vision.camera.frame_height,
            self.cfg.vision.camera.format
        )

        print(f"Started: {self.current_camera['name']}")
        time.sleep(1)  # Allow camera to stabilize

    def switch_camera(self):
        """Switch to next available camera"""
        next_idx = (self.current_camera_idx + 1) % len(self.cameras)
        self.start_camera(next_idx)

    def draw_interface(self, frame, faces):
        """Draw UI elements on the frame"""
        h, w = frame.shape[:2]

        # Draw faces
        for f in faces:
            bbox = [int(val) for val in f.bbox]
            raw_id = f.identity or "unknown"

            # Clean up face id (remove _idx used for unique registration)
            if raw_id != "unknown":
                person_name = raw_id.rsplit('_', 1)[0]
            else:
                person_name = "unknown"

            sim = f.similarity or 0.0

            color = (0, 255, 0) if person_name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = f"{person_name} ({sim:.2f})"
            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Info panel background
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 100), (255, 0, 0), 2)

        # Camera info
        cv2.putText(
            frame,
            f"Camera: {self.current_camera['name']}", # pyright: ignore[reportOptionalSubscript]
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # Registered faces
        cv2.putText(
            frame,
            f"Database: {self.registered_faces} faces",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Faces detected
        cv2.putText(
            frame,
            f"Detected: {len(faces)}",
            (w - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Controls help (bottom)
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - 40), (w, h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            "S: Switch cam | Q: Quit | H: Help",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    def show_help(self):
        """Display help information"""
        help_text = """
╔══════════════════════════════════════════════════════════╗
║          FACE IDENTIFICATION TOOL - HELP                 ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  KEYBOARD CONTROLS:                                      ║
║  ───────────────                                         ║
║  S        : Switch between available cameras             ║
║  Q        : Quit application                             ║
║  H        : Show this help                               ║
║                                                          ║
║  USAGE:                                                  ║
║  ─────                                                   ║
║  1. Ensure you have captured faces using face_capture.py ║
║  2. The system loads faces from data/face_dataset        ║
║  3. Position people in front of the camera               ║
║  4. The system will identify recognized faces            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """
        print(help_text)

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def run(self):
        """Main application loop"""
        print("\n" + "=" * 60)
        print("  FACE IDENTIFICATION TOOL")
        print("=" * 60)
        print("\nPress 'H' at any time to show help")
        print("\nStarting camera...")

        # Start first available camera
        self.start_camera(0)

        # Show initial help
        time.sleep(1)
        self.show_help()

        print("\n✓ Ready! Camera window opened.\n")

        try:
            while True:
                # Capture frame (typically RGB from custom camera handler)
                frame = self.current_camera["camera"].capture_frame() # pyright: ignore[reportOptionalSubscript]

                # FaceInsightPipeline expects BGR images
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Recognize faces
                faces = self.recognizer.recognize(frame_bgr, thresh=0.4)

                # Draw interface
                display_frame = self.draw_interface(frame_bgr, faces)

                # Update FPS
                self.update_fps()

                # Display frame
                cv2.imshow("Face Identify Tool", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nQuitting...")
                    break

                elif key == ord("s") or key == ord("S"):
                    if len(self.cameras) > 1:
                        self.switch_camera()
                    else:
                        print("⚠ Only one camera available")

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
                except:
                    pass

            print("\n✓ Session complete!")
            print("=" * 60)

def main():
    """Main entry point"""
    try:
        tool = FaceIdentifyTool()
        tool.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure:")
        print("  • Cameras are properly connected")
        print("  • Camera overlays are configured in /boot/firmware/config.txt")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
