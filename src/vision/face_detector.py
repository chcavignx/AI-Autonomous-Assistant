"""Face detection modules (CPU/RetinaFace and IMX500)."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis

from src.utils.config import Config, config, load_config
from src.vision.base import BaseDetector, DetectionDict

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


@dataclass
class FaceEmbedding:
    id: str
    embedding: np.ndarray


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 in pixels
    landmark5: np.ndarray | None  # (5,2) or None
    score: float
    identity: str | None = None
    similarity: float | None = None
    embedding: np.ndarray | None = None  # Extracted embedding if available


class InsightFaceDetector(BaseDetector):
    """InsightFace CPU detector (RetinaFace)."""

    def __init__(self, cfg: Config | None = None, det_size: tuple[int, int] = (640, 640)) -> None:
        """Initialize the InsightFace CPU detector."""
        self.cfg = cfg or load_config()
        model_path = self.cfg.vision.face_detector_model_path

        if not pathlib.Path(model_path).exists():
            msg = f"Face detector model not found at {model_path}"
            logger.error(msg)
            raise RuntimeError(msg)

        self.app = FaceAnalysis(
            name=self.cfg.vision.face_model_name,
            root=model_path.parent,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        faces = self.app.get(frame)
        results: list[DetectedFace] = []
        for f in faces:
            landmark = None
            if hasattr(f, "landmark") and isinstance(f.landmark, np.ndarray):
                landmark = f.landmark.astype(float)
            elif hasattr(f, "landmark5") and isinstance(f.landmark5, np.ndarray):
                landmark = f.landmark5.astype(float)

            det_score = float(f.det_score) if hasattr(f, "det_score") else 1.0

            embedding = None
            if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                embedding = f.normed_embedding.astype(np.float32)
            elif hasattr(f, "embedding") and f.embedding is not None:
                embedding = f.embedding.astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

            results.append(
                DetectedFace(
                    bbox=(float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3])),
                    landmark5=landmark,
                    score=det_score,
                    embedding=embedding,
                )
            )
        return results

    def detect(self, frame: np.ndarray) -> list[DetectionDict]:
        faces = self.detect_faces_raw(frame)
        detections: list[DetectionDict] = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": f.score,
                    "class_id": 0,
                    "label": "face",
                }
            )
        return detections


@dataclass
class Imx500Config:
    """Configuration for the IMX500 pipeline for on-sensor detection."""

    frame_width: int = config.vision.camera.imx500_frame_width
    frame_height: int = config.vision.camera.imx500_frame_height


class Imx500Detector(BaseDetector):
    """IMX500 Face detector wrapper using native Picamera2 API."""

    def __init__(self, cfg: Config | None = None, imx500: Path | None = None) -> None:
        """Initialize the IMX500 detector."""
        self.cfg = cfg or load_config()
        model_path = imx500 or self.cfg.vision.face_detector_model_path

        if not pathlib.Path(model_path).exists():
            model_path = "/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"
        self.imx500 = None

        try:
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics

            if self.imx500 is None:
                self.imx500 = IMX500(model_path)
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            self.intrinsics.update_with_defaults()
        except ImportError:
            logger.exception("picamera2 is required for IMX500 detector")
            self.imx500 = None

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        """Infer without metadata is not supported directly."""
        return []

    def detect_faces_metadata(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectedFace]:
        """Fetch latest bounding boxes from metadata."""
        if not metadata or not self.imx500:
            return []

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []

        threshold = self.cfg.vision.face_recognition_threshold
        input_w, input_h = self.imx500.get_input_size()

        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if self.intrinsics.bbox_normalization:
            boxes /= input_h
        if self.intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

        faces = []
        frame_h, frame_w = frame.shape[:2]

        for box, score, _category in zip(boxes, scores, classes, strict=False):
            if score > threshold:
                x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                if self.intrinsics.bbox_order == "yx":
                    y0, x0, y1, x1 = box[0], box[1], box[2], box[3]

                if not self.intrinsics.bbox_normalization and self.intrinsics.postprocess != "nanodet":
                    x0 /= input_w
                    y0 /= input_h
                    x1 /= input_w
                    y1 /= input_h

                # NOTE: IMX500 face models might not provide 5-point landmarks or might provide them in other outputs.
                # Assuming simple bounding boxes for IMX500 face detection as per original rpi-ai-camera implementation
                faces.append(
                    DetectedFace(
                        bbox=(x0 * frame_w, y0 * frame_h, x1 * frame_w, y1 * frame_h),
                        landmark5=None,
                        score=float(score),
                    )
                )

        return faces

    def detect(self, frame: np.ndarray) -> list[DetectionDict]:
        return []

    def stop(self) -> None:
        pass
