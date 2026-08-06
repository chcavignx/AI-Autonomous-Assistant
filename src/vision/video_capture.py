#!/usr/bin/env python3
"""Video capture using OpenCV and modular detectors."""

from __future__ import annotations

import logging
import pathlib
import sys
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import cv2

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.utils import config
from src.utils.camera import ThreadedCamera
from src.vision.face_in_frame import FaceInFrame
from src.vision.object_insight_frame import ObjectInsightFrame

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

    from src.utils.config import Config

config.setup_python_path()
vcfg: Config = config.load_config()

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class VideoCapture:
    """Simplified Video Capture class orchestrating modular frame processing."""

    def __init__(self, cfg: Config | None = None) -> None:
        """Initialize the VideoCapture instance.

        Args:
            cfg: Configuration object.

        """
        self.cfg: Config = cfg or vcfg
        self.camera: ThreadedCamera | None = None

        self.running: bool = False
        self.latest_frame: np.ndarray | None = None
        self.latest_results: Any = None
        self.lock: threading.Lock = threading.Lock()

        # Control flags and metrics
        self.enable_object_detection: bool = self.cfg.vision.enable_object_detection
        self.enable_face_detection: bool = self.cfg.vision.enable_face_detection
        self.current_person_name: str = "unknown"
        self.last_face_count: int = 0
        self.capture_count: int = 0
        self.fps: float = 0.0

        self._initialize_camera()
        self._initialize_processors()

    @property
    def model(self) -> Any:
        """Backward-compatible access to the underlying YOLO model."""
        return self.object_processor.detector.model

    @property
    def enable_detection(self) -> bool:
        """Backward-compatible mapping of enable_detection to enable_object_detection."""
        return self.enable_object_detection

    @enable_detection.setter
    def enable_detection(self, value: bool) -> None:
        self.enable_object_detection = value

    def _initialize_camera(self) -> None:
        """Initialize the ThreadedCamera instance."""
        self.camera = ThreadedCamera(self.cfg)

    def _initialize_processors(self) -> None:
        """Initialize modular frame processors."""
        try:
            imx500_instance = getattr(self.camera, "imx500", None)
            picam2_instance = getattr(self.camera, "picam2", None)

            # Initialize Object Processor
            if self.cfg.vision.object_model_type == "yolo_imx500":
                try:
                    from src.vision.yolo_imx500 import Imx500Detector

                    obj_detector = Imx500Detector(imx500=imx500_instance, picam2=picam2_instance)
                    self.object_processor = ObjectInsightFrame(self.cfg, detector=obj_detector)
                except Exception as e:
                    logger.warning("Failed to initialize IMX500 object processor, falling back to default: %s", e)
                    self.object_processor = ObjectInsightFrame(self.cfg)
            else:
                self.object_processor = ObjectInsightFrame(self.cfg)

            # Initialize Face Processor
            detector_type = self.cfg.vision.face_detector_type
            if detector_type == "imx500":
                try:
                    from src.vision.face_insight_pipeline import FaceInsightPipeline

                    face_detector = FaceInsightPipeline(
                        self.cfg,
                        arcface_model_name=str(self.cfg.vision.post_processing_model_full_path),
                        imx500=imx500_instance,
                    )
                    self.face_processor = FaceInFrame(
                        self.cfg, detector_type=detector_type, face_recognizer=face_detector
                    )
                    self.face_processor.detector = face_detector.detector
                except Exception as e:
                    logger.warning("Failed to initialize IMX500 face processor, falling back to default: %s", e)
                    self.face_processor = FaceInFrame(self.cfg, detector_type="insightface")
            else:
                self.face_processor = FaceInFrame(self.cfg, detector_type=detector_type)

            logger.info("Processors loaded successfully")
        except Exception as e:
            logger.exception("Error loading processors: %s", e)

    def start(self) -> None:
        """Start video capture."""
        if not self.running:
            self.running = True
            if self.camera is not None:
                self.camera.start()
                logger.info("Video capture started")
            else:
                logger.error("Camera not initialized")

    def stop(self) -> None:
        """Stop video capture."""
        self.running = False
        if self.camera is not None:
            self.camera.stop()
            logger.info("Video capture stopped")
        else:
            logger.error("Camera not initialized")

    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera."""
        if self.camera is None:
            return None
        frame, metadata = self.camera.read()
        if frame is not None:
            with self.lock:
                self.latest_frame = frame
                self.latest_metadata = metadata
            logger.debug("Frame captured")
        else:
            logger.error("Failed to capture frame")
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using active frame processors."""
        annotated_frame = frame.copy()

        # 1. Run YOLO Object Detection if enabled
        if self.enable_object_detection:
            try:
                metadata = getattr(self, "latest_metadata", None)
                annotated_frame, _detections, results = self.object_processor.process_frame(
                    annotated_frame,
                    draw=True,
                    metadata=metadata,
                )
                if results is not None:
                    self.latest_results = results
            except Exception as e:
                logger.exception("Error in object detection frame processing: %s", e)

        # 2. Run Face Detection & Recognition if enabled
        if self.enable_face_detection:
            try:
                thresh = self.cfg.vision.face_recognition_threshold
                metadata = getattr(self, "latest_metadata", None)
                if hasattr(self.face_processor, "process_frame_metadata"):
                    annotated_frame, faces = self.face_processor.process_frame_metadata(
                        annotated_frame,
                        metadata=metadata,
                        thresh=thresh,
                        draw=True,
                    )
                else:
                    annotated_frame, faces = self.face_processor.process_frame(
                        annotated_frame,
                        thresh=thresh,
                        draw=True,
                    )
                self.last_face_count = len(faces)
            except Exception as e:
                logger.exception("Error in face recognition frame processing: %s", e)
        else:
            self.last_face_count = 0

        # Calculate and overlay speed/metrics
        self._add_performance_overlay(annotated_frame)
        logger.debug("Frame processed")
        return annotated_frame

    def _add_performance_overlay(self, frame: np.ndarray) -> None:
        """Add premium status and speed information using Ultralytics metrics."""
        if self.latest_results is None:
            return

        # Speed metrics (ms)
        speed: dict[str, float] = self.latest_results.speed
        preprocess: float = speed.get("preprocess", 0)
        inference: float = speed.get("inference", 0)
        postprocess: float = speed.get("postprocess", 0)
        total_ms: float = preprocess + inference + postprocess
        fps_calc: float = 1000 / total_ms if total_ms > 0 else 0
        self.fps = fps_calc

        # Create a sleek overlay
        overlay: np.ndarray = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw metrics
        font: int = cv2.FONT_HERSHEY_DUPLEX
        color: tuple[int, int, int] = (255, 255, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25), font, 0.6, color, 1)
        cv2.putText(frame, f"Inf: {inference:.1f}ms", (10, 50), font, 0.5, color, 1)
        cv2.putText(frame, f"Objs: {len(self.latest_results.boxes)}", (10, 75), font, 0.5, color, 1)

    def benchmark(self, iterations: int = 100) -> None:
        """Run a performance benchmark of the current model."""
        if self.latest_frame is None:
            logger.warning("No model or frame available for benchmarking")
            return

        st = time.time()
        for _ in range(iterations):
            self.object_processor.process_frame(self.latest_frame, draw=False)
        et = time.time()

        elapsed_time: float = (et - st) / iterations
        logger.info("Benchmark completed in %.4f seconds", elapsed_time)

    def capture_photo(self) -> tuple[bool, str]:
        """Capture and save detection results from the latest frame."""
        with self.lock:
            if self.latest_frame is None or self.latest_results is None:
                return False, "No frame or results available"
            results = self.latest_results

        output_dir = self.cfg.paths.data_path / "captures"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        # Save full annotated frame
        annotated_path = output_dir / f"{timestamp}_annotated.jpg"
        results.save(filename=str(annotated_path))  # save to disk
        self.capture_count += 1
        logger.info("Saved annotated frame to %s", annotated_path)
        return True, f"Captured result with {len(results.boxes)} object(s)"

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Generate frames for Flask video streaming.

        Yields:
            Boundary-separated JPEG image bytes.

        """
        while self.running:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                processed_frame: np.ndarray = self.process_frame(frame)
                _ret: bool
                buffer: np.ndarray
                _ret, buffer = cv2.imencode(".jpg", processed_frame)
                frame_bytes: bytes = buffer.tobytes()
            except Exception:
                time.sleep(0.1)
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
