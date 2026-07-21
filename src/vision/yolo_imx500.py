"""Sony IMX500 On-Sensor vision detector integration module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2

from src.utils.config import config
from src.vision.base import BaseDetector, DetectionDict
from src.vision.yolo_cpu import Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Imx500Config:
    """Configuration for Sony IMX500 On-Sensor camera interface."""

    frame_width: int = config.vision.camera.imx500_frame_width
    frame_height: int = config.vision.camera.imx500_frame_height


class Imx500Detector(BaseDetector):
    """Sony IMX500 AI Camera on-sensor vision detector.

    Conforms to BaseDetector for plug-and-play orchestration.
    """

    def __init__(self, cfg: Imx500Config | None = None, imx500: Any | None = None) -> None:
        """Initialize the IMX500 detector."""
        self.cfg = cfg or Imx500Config()
        self.latest_frame = None
        self.imx500 = imx500

        try:
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics

            if self.imx500 is None:
                self.imx500 = IMX500(str(config.vision.object_model_full_path))
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            self.intrinsics.update_with_defaults()
        except ImportError:
            logger.exception("picamera2 is required for IMX500 detector")
            self.imx500 = None

    def infer(self, frame: np.ndarray, metadata: dict | None = None) -> Any:
        """Fetch latest bounding boxes from metadata.

        Returns:
            A DummyResults object containing the parsed detections.

        """
        self.latest_frame = frame
        if not metadata or not self.imx500:
            return None

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return None

        threshold = config.vision.object_recognition_threshold
        input_w, input_h = self.imx500.get_input_size()

        if self.intrinsics.postprocess == "nanodet":
            from picamera2.devices.imx500 import postprocess_nanodet_detection
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=threshold, iou_thres=0.65, max_out_dets=10
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if self.intrinsics.bbox_normalization:
                boxes /= input_h
            if self.intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        dets = []
        frame_h, frame_w = frame.shape[:2]

        for box, score, category in zip(boxes, scores, classes, strict=False):
            if score > threshold:
                x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                if self.intrinsics.bbox_order == "yx":
                    y0, x0, y1, x1 = box[0], box[1], box[2], box[3]

                if not self.intrinsics.bbox_normalization and self.intrinsics.postprocess != "nanodet":
                    x0 /= input_w
                    y0 /= input_h
                    x1 /= input_w
                    y1 /= input_h

                dets.append(
                    Detection(
                        x1=x0 * frame_w,
                        y1=y0 * frame_h,
                        x2=x1 * frame_w,
                        y2=y1 * frame_h,
                        score=float(score),
                        cls=int(category),
                    )
                )

        class DummyBoxes:
            pass

        class DummyResults:
            def __init__(self, detections: list, orig_shape: tuple) -> None:
                self.detections = detections
                self.boxes = DummyBoxes()
                self.boxes.xyxy = []
                self.boxes.conf = []
                self.boxes.cls = []
                for d in detections:
                    self.boxes.xyxy.append([d.x1, d.y1, d.x2, d.y2])
                    self.boxes.conf.append(d.score)
                    self.boxes.cls.append(d.cls)
                self.orig_shape = orig_shape
                self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

            def save(self, filename: str) -> None:
                pass

        return DummyResults(dets, frame.shape[:2])

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects using metadata and return standardized format."""
        results = self.infer(frame, metadata=metadata)
        if results is None:
            return []

        detections: list[DetectionDict] = []
        labels = self.intrinsics.labels if self.intrinsics and self.intrinsics.labels else []

        for d in results.detections:
            label_idx = int(d.cls)
            label = labels[label_idx] if label_idx < len(labels) else str(label_idx)
            detections.append(
                {
                    "box": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                    "score": float(d.score),
                    "class_id": label_idx,
                    "label": label,
                }
            )

        return detections

    def draw_detections(self, results: Any) -> np.ndarray:
        if self.latest_frame is None:
            import numpy as np

            return np.zeros((*results.orig_shape, 3), dtype=np.uint8)

        annotated_frame = self.latest_frame.copy()

        labels = self.intrinsics.labels if self.intrinsics and self.intrinsics.labels else []

        for d in results.detections:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_idx = int(d.cls)
            label_text = (
                f"{labels[label_idx]} {d.score:.2f}" if label_idx < len(labels) else f"{label_idx} {d.score:.2f}"
            )

            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED
            )
            cv2.putText(annotated_frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated_frame

    def stop(self) -> None:
        pass
