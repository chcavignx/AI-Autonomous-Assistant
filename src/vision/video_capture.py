#!/usr/bin/env python3
"""Video capture using OpenCV and modular detectors."""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.utils.camera import ThreadedCamera
from src.utils.config import Config, load_config
from src.vision.face_insight_frame import FaceInsightFrame
from src.vision.object_insight_frame import ObjectInsightFrame

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__.split(".")[1])


def _is_valid_frame(frame: Any) -> bool:
    """Check if frame is a non-empty array or a valid test object (e.g. MagicMock)."""
    if frame is None:
        return False
    size = getattr(frame, "size", None)
    if isinstance(size, (int, float, np.integer)):
        return bool(size > 0)
    return True


def compute_iou(box1: list[int] | tuple[int, int, int, int], box2: list[int] | tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


# Backward compatibility alias
_compute_iou = compute_iou


class _FaceTrack:
    """Represents a tracked face and its recognition state across frames."""

    def __init__(self, track_id: int, bbox: list[int], score: float = 1.0) -> None:
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.name: str = "unknown"
        self.similarity: float = 0.0
        self.last_seen: float = time.time()
        self.last_recognized: float = 0.0
        self.pending_recognition: bool = False


class VideoCapture:
    """Simplified Video Capture class orchestrating modular frame processing."""

    def __init__(self, cfg: Config | None = None) -> None:
        """Initialize the VideoCapture instance.

        Args:
            cfg: Configuration object.

        """
        self.cfg: Config = cfg or load_config()
        self.camera: ThreadedCamera | None = None

        self.running: bool = False
        self.latest_frame: np.ndarray | None = None
        self.latest_results: Any = None
        self.lock: threading.Lock = threading.Lock()

        # Control flags and metrics
        self.enable_object_detection: bool = self.cfg.vision.enable_object_detection
        self.enable_face_detection: bool = self.cfg.vision.enable_face_detection
        self.enable_face_recognition: bool = self.cfg.vision.enable_face_recognition
        self.async_face_recognition: bool = self.cfg.vision.async_face_recognition
        self.face_recognition_interval: float = self.cfg.vision.face_recognition_interval

        self.current_person_name: str = "unknown"
        self.last_face_count: int = 0
        self.capture_count: int = 0
        self.fps: float = 0.0

        # Asynchronous recognition tracking state
        self._tracks: dict[int, _FaceTrack] = {}
        self._next_track_id: int = 1
        self._recognition_lock = threading.Lock()
        self._recognition_executor = None
        if self.async_face_recognition:
            from concurrent.futures import ThreadPoolExecutor

            self._recognition_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face_recog")

        self._initialize_camera()
        self._initialize_processors()

    @property
    def model(self) -> Any:
        """Backward-compatible access to the underlying YOLO model."""
        if hasattr(self, "object_processor") and hasattr(self.object_processor, "detector"):
            return getattr(self.object_processor.detector, "model", None)
        return None

    @property
    def active_tracks(self) -> list[dict[str, Any]]:
        """Create a snapshot list of currently active face tracks."""
        with self._recognition_lock:
            return [
                {
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "name": trk.name,
                    "similarity": trk.similarity,
                    "score": trk.score,
                }
                for trk in self._tracks.values()
            ]

    @property
    def has_async_recognition(self) -> bool:
        """Check whether async face recognition is enabled."""
        return self.async_face_recognition and self._recognition_executor is not None

    def _initialize_camera(self) -> None:
        """Initialize the ThreadedCamera instance."""
        if self.camera is not None:
            self.camera.stop()
        self.camera = ThreadedCamera(self.cfg)

    def _init_object_processor(self, imx500_instance: Any, picam2_instance: Any) -> ObjectInsightFrame:
        """Initialize object processor based on configuration."""
        if self.cfg.vision.object_model_type == "yolo_imx500":
            try:
                from src.vision.yolo_imx500 import Imx500Detector

                obj_detector = Imx500Detector(imx500=imx500_instance, picam2=picam2_instance)
                return ObjectInsightFrame(self.cfg, detector=obj_detector)
            except Exception as e:
                logger.warning("Failed to initialize IMX500 object processor, falling back to default: %s", e)
                return ObjectInsightFrame(self.cfg)
        return ObjectInsightFrame(self.cfg)

    def _init_face_processor(self, imx500_instance: Any) -> FaceInsightFrame:
        """Initialize face processor based on configuration."""
        detector_type = self.cfg.vision.face_detector_type
        if detector_type == "imx500":
            try:
                from src.vision.face_insight_pipeline import FaceInsightPipeline

                logger.debug("Initializing IMX500 face processor")
                face_detector = FaceInsightPipeline(
                    self.cfg,
                    arcface_model_name=str(self.cfg.vision.post_processing_model_full_path),
                    imx500=imx500_instance,
                )
                processor = FaceInsightFrame(self.cfg, detector_type=detector_type, face_recognizer=face_detector)
                processor.detector = face_detector.detector
                return processor
            except Exception as e:
                logger.warning("Failed to initialize IMX500 face processor, falling back to default: %s", e)
                return FaceInsightFrame(self.cfg, detector_type="insightface")
        return FaceInsightFrame(self.cfg, detector_type=detector_type)

    def _initialize_processors(self) -> None:
        """Initialize modular frame processors."""
        try:
            imx500_instance = getattr(self.camera, "imx500", None)
            picam2_instance = getattr(self.camera, "picam2", None)
            self.object_processor = self._init_object_processor(imx500_instance, picam2_instance)
            self.face_processor = self._init_face_processor(imx500_instance)
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
        if self._recognition_executor is not None:
            self._recognition_executor.shutdown(wait=False)
            self._recognition_executor = None
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
        if _is_valid_frame(frame):
            with self.lock:
                self.latest_frame = frame
                self.latest_metadata = metadata
            logger.debug("Frame captured")
            return frame
        logger.debug("Failed to capture frame")
        return None

    def _update_track_from_faces(self, track_id: int, faces: list[Any]) -> None:
        """Update track name and similarity from recognized face instances."""
        with self._recognition_lock:
            track = self._tracks.get(track_id)
            if track is not None and faces:
                best_face = None
                best_iou = 0.0
                for f in faces:
                    iou = compute_iou(track.bbox, [int(v) for v in f.bbox])
                    if iou > best_iou:
                        best_iou = iou
                        best_face = f
                if best_face is None:
                    best_face = faces[0]

                track.name = best_face.identity if best_face.identity is not None else "unknown"
                track.similarity = best_face.similarity if best_face.similarity is not None else 0.0
                track.last_recognized = time.time()
                self.current_person_name = track.name

    def _async_recognize_task(
        self,
        frame_copy: np.ndarray,
        track_id: int,
        thresh: float,
        metadata: dict | None = None,
    ) -> None:
        """Background worker task to extract ArcFace embeddings and match identity."""
        try:
            recognizer = getattr(self.face_processor, "face_recognizer", None)
            if recognizer is not None:
                faces = recognizer.recognize(frame_copy, thresh=thresh, metadata=metadata)
                self._update_track_from_faces(track_id, faces)
        except Exception as e:
            logger.debug("Async face recognition error for track %d: %s", track_id, e)
        finally:
            with self._recognition_lock:
                track = self._tracks.get(track_id)
                if track is not None:
                    track.pending_recognition = False

    def _process_faces_async(
        self,
        annotated_frame: np.ndarray,
        orig_frame: np.ndarray,
        thresh: float,
        metadata: dict | None,
    ) -> np.ndarray:
        """Perform fast detection and asynchronous tracking-based recognition."""
        active_detector = getattr(self.face_processor, "detector", None) or getattr(
            getattr(self.face_processor, "face_recognizer", None), "detector", None
        )
        raw_detections = []
        if active_detector is not None and hasattr(active_detector, "detect"):
            raw_detections = active_detector.detect(annotated_frame, metadata=metadata)

        now = time.time()
        matched_track_ids: set[int] = set()
        current_faces: list[dict] = []

        for det in raw_detections:
            det_box = det["box"]
            det_score = det.get("score", 1.0)
            best_track_id, best_iou = None, 0.3

            with self._recognition_lock:
                for tid, track in self._tracks.items():
                    if tid not in matched_track_ids:
                        iou = compute_iou(det_box, track.bbox)
                        if iou > best_iou:
                            best_iou, best_track_id = iou, tid

                if best_track_id is not None:
                    track = self._tracks[best_track_id]
                    track.bbox = det_box
                    track.score = det_score
                    track.last_seen = now
                    matched_track_ids.add(best_track_id)
                else:
                    best_track_id = self._next_track_id
                    self._next_track_id += 1
                    track = _FaceTrack(track_id=best_track_id, bbox=det_box, score=det_score)
                    self._tracks[best_track_id] = track
                    matched_track_ids.add(best_track_id)

                needs_recog = (
                    self.enable_face_recognition
                    and not track.pending_recognition
                    and (track.last_recognized == 0.0 or (now - track.last_recognized > self.face_recognition_interval))
                )
                if needs_recog and self._recognition_executor is not None:
                    track.pending_recognition = True
                    frame_copy = orig_frame.copy()
                    self._recognition_executor.submit(
                        self._async_recognize_task,
                        frame_copy,
                        best_track_id,
                        thresh,
                        metadata,
                    )

                name = track.name
                best_sim = track.similarity

            current_faces.append(
                {
                    "box": det_box,
                    "name": name,
                    "similarity": best_sim,
                    "score": det_score,
                }
            )

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (det_box[0], det_box[1]), (det_box[2], det_box[3]), color, 2)
            label = f"{name} ({best_sim:.2f})" if name != "unknown" else "face"
            cv2.putText(
                annotated_frame,
                label,
                (det_box[0], det_box[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                color,
                1,
            )

        with self._recognition_lock:
            stale_ids = [tid for tid, trk in self._tracks.items() if now - trk.last_seen > 2.0]
            for tid in stale_ids:
                del self._tracks[tid]

        self.last_face_count = len(current_faces)
        if current_faces:
            self.current_person_name = current_faces[0]["name"]

        return annotated_frame

    def _process_faces_sync(
        self,
        annotated_frame: np.ndarray,
        thresh: float,
        metadata: dict | None,
    ) -> np.ndarray:
        """Perform synchronous face detection and recognition."""
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
        if faces:
            self.current_person_name = faces[0].get("name", "unknown")
        return annotated_frame

    def _process_face_pipeline(self, annotated_frame: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Run face detection & recognition pipeline safely."""
        try:
            thresh = self.cfg.vision.face_recognition_threshold
            metadata = getattr(self, "latest_metadata", None)
            if self.async_face_recognition and self._recognition_executor is not None:
                return self._process_faces_async(annotated_frame, frame, thresh, metadata)
            return self._process_faces_sync(annotated_frame, thresh, metadata)
        except Exception as e:
            logger.exception("Error in face recognition frame processing: %s", e)
            return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using active frame processors."""
        if not _is_valid_frame(frame):
            return frame
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
            annotated_frame = self._process_face_pipeline(annotated_frame, frame)
        else:
            self.last_face_count = 0

        # Calculate and overlay speed/metrics
        self._add_performance_overlay(annotated_frame)
        logger.debug("Frame processed")
        return annotated_frame

    def _add_performance_overlay(self, frame: np.ndarray) -> None:
        """Add premium status and speed information using Ultralytics metrics."""
        if self.latest_results is None or not _is_valid_frame(frame):
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
        boxes = getattr(self.latest_results, "boxes", None)
        if hasattr(self.latest_results, "detections") and isinstance(self.latest_results.detections, (list, tuple)):
            obj_count = len(self.latest_results.detections)
        elif boxes is not None and hasattr(boxes, "__len__"):
            obj_count = len(boxes)
        else:
            obj_count = 0
        cv2.putText(frame, f"Objs: {obj_count}", (10, 75), font, 0.5, color, 1)

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
        boxes = getattr(results, "boxes", None)
        if hasattr(results, "detections") and isinstance(results.detections, (list, tuple)):
            obj_count = len(results.detections)
        elif boxes is not None and hasattr(boxes, "__len__"):
            obj_count = len(boxes)
        else:
            obj_count = 0
        return True, f"Captured result with {obj_count} object(s)"

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
