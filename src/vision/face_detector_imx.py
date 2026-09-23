"""IMX500 face detector module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectedFace, DetectionDict

logger = logging.getLogger(__name__.split(".")[1])


@dataclass
class Imx500Config:
    """Configuration for the IMX500 pipeline for on-sensor detection."""

    frame_width: int = 640
    frame_height: int = 480


class Imx500Detector(BaseDetector):
    """IMX500 Face detector wrapper using native Picamera2 API."""

    def __init__(self, cfg: Config | None = None, imx500: Any | Path | None = None) -> None:
        """Initialize the IMX500 detector."""
        self.cfg = cfg or load_config()
        self.imx500: Any = None
        self.intrinsics: Any = None

        if imx500 is not None and not isinstance(imx500, (str, Path)):
            logger.debug("Initializing IMX500 detector with imx500 object")
            self.imx500 = imx500
            self.intrinsics = getattr(self.imx500, "network_intrinsics", None)
            return

        raw_path = self.cfg.vision.face_detector_model_path
        model_path = Path(raw_path) if raw_path else Path("/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk")

        if not model_path.exists() or model_path.suffix != ".rpk":
            candidates = [
                Path("/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"),
                Path("/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"),
            ]
            for cand in candidates:
                if cand.exists():
                    model_path = cand
                    break

        try:
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics

            if model_path.exists() and model_path.suffix == ".rpk":
                logger.debug("Initializing IMX500 detector with model path %s", model_path)
                self.imx500 = IMX500(str(model_path))
                self.intrinsics = self.imx500.network_intrinsics
                if not self.intrinsics:
                    self.intrinsics = NetworkIntrinsics()
                    self.intrinsics.task = "object detection"
                self.intrinsics.update_with_defaults()
        except Exception as e:
            logger.info("picamera2 IMX500 face detector initialization skipped/stubbed: %s", e)
            self.imx500 = None

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces returning raw DetectedFace list without metadata."""
        return []

    def detect_faces_metadata(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectedFace]:
        """Detect faces using IMX500 metadata from picamera2."""
        if not metadata or not self.imx500:
            return []

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None or len(np_outputs) < 2:
            return []

        frame_h, frame_w = frame.shape[:2]
        try:
            sz = self.imx500.get_input_size()
            if isinstance(sz, (tuple, list)) and len(sz) >= 2:
                input_w, input_h = int(sz[0]), int(sz[1])
            else:
                input_w, input_h = 640, 480
        except Exception:
            input_w, input_h = 640, 480

        boxes = np_outputs[0]
        scores = np_outputs[1]

        if boxes.ndim > 2:
            boxes = boxes[0]
        if scores.ndim > 1:
            scores = scores[0]

        threshold = getattr(self.cfg.vision, "face_recognition_threshold", 0.5) if hasattr(self.cfg, "vision") else 0.5

        faces: list[DetectedFace] = []
        bbox_order = getattr(self.intrinsics, "bbox_order", "xy")
        bbox_normalized = getattr(self.intrinsics, "bbox_normalization", False)

        for box, score in zip(boxes, scores, strict=False):
            sc = float(score)
            if sc >= threshold:
                box_arr = np.array(box, dtype=np.float32)
                if bbox_normalized:
                    box_arr = box_arr / float(input_h)

                if bbox_order == "yx":
                    y0, x0, y1, x1 = box_arr[0], box_arr[1], box_arr[2], box_arr[3]
                else:
                    x0, y0, x1, y1 = box_arr[0], box_arr[1], box_arr[2], box_arr[3]

                if not bbox_normalized:
                    x0 /= input_w
                    y0 /= input_h
                    x1 /= input_w
                    y1 /= input_h

                det_x1, det_y1, det_x2, det_y2 = x0 * frame_w, y0 * frame_h, x1 * frame_w, y1 * frame_h
                faces.append(
                    DetectedFace(
                        bbox=(float(det_x1), float(det_y1), float(det_x2), float(det_y2)),
                        landmark5=None,
                        score=sc,
                    )
                )
        return faces

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect faces returning standardized detection dictionaries."""
        faces = self.detect_faces_metadata(frame, metadata=metadata)
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

    def stop(self) -> None:
        """Release IMX500 resources."""
        self.imx500 = None


Imx500FaceDetector = Imx500Detector
