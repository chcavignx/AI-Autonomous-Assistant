"""Sony IMX500 On-Sensor vision detector integration module."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.utils.config import config
from src.vision.base import BaseDetector, DetectionDict
from src.vision.yolo_cpu import Detection

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


_imx500_w, _imx500_h = config.vision.get_model_resolution("imx500")


@dataclass
class Imx500Config:
    """Configuration for Sony IMX500 On-Sensor camera interface."""

    frame_width: int = _imx500_w
    frame_height: int = _imx500_h


class DummyBoxes:
    """Mock boxes object for Ultralytics API compatibility."""

    def __init__(self) -> None:
        """Initialize empty bounding box attributes."""
        self.xyxy = []
        self.conf = []
        self.cls = []

    def __len__(self) -> int:
        """Return number of bounding boxes."""
        return len(self.xyxy)

    def __iter__(self) -> Any:
        """Return iterator over bounding boxes."""
        return iter(self.xyxy)


class DummyResults:
    """Mock results object for Ultralytics API compatibility."""

    def __init__(self, detections: list, orig_shape: tuple) -> None:
        """Initialize mock results object wrapping detection instances."""
        self.detections = detections
        self.boxes = DummyBoxes()
        for d in detections:
            self.boxes.xyxy.append([d.x1, d.y1, d.x2, d.y2])
            self.boxes.conf.append(d.score)
            self.boxes.cls.append(d.cls)
        self.orig_shape = orig_shape
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    def save(self, filename: str) -> None:
        """No-op fallback for image saving."""


class Imx500Detector(BaseDetector):
    """Sony IMX500 AI Camera on-sensor vision detector.

    Conforms to BaseDetector for plug-and-play orchestration.
    """

    def __init__(
        self,
        cfg: Imx500Config | None = None,
        imx500: Any | None = None,
        picam2: Any | None = None,
    ) -> None:
        """Initialize the IMX500 detector."""
        self.cfg = cfg or Imx500Config()
        self.latest_frame = None
        self.imx500 = imx500
        self.picam2 = picam2
        self.model = self.imx500

        try:
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics

            if self.imx500 is None:
                self.imx500 = IMX500(str(config.vision.object_model_full_path))
                self.model = self.imx500
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            self.intrinsics.update_with_defaults()

            if (
                self.intrinsics.labels is None or len(self.intrinsics.labels) == 0
            ) and config.vision.object_label_full_path:
                label_path = config.vision.object_label_full_path
                if label_path.exists():
                    self.intrinsics.labels = pathlib.Path(label_path).read_text(encoding="utf-8").splitlines()
            if self.imx500 is not None:
                self.imx500.show_network_fw_progress_bar()
        except Exception as e:
            logger.warning("picamera2 / IMX500 hardware not initialized: %s", e)
            self.imx500 = None
            self.model = None
            self.intrinsics = None

    def get_labels(self) -> list[str]:
        """Return label list, applying ignore_dash_labels if specified in network intrinsics."""
        if not self.intrinsics or not self.intrinsics.labels:
            return []
        labels = self.intrinsics.labels
        if getattr(self.intrinsics, "ignore_dash_labels", False):
            labels = [label for label in labels if label and label != "-"]
        return labels

    def infer(self, frame: np.ndarray, metadata: dict | None = None) -> Any:
        """Fetch latest bounding boxes from metadata.

        Returns:
            A DummyResults object containing the parsed detections.

        """
        self.latest_frame = frame

        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        else:
            frame_h, frame_w = 480, 640

        if not metadata or not self.imx500:
            return DummyResults([], (frame_h, frame_w))

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return DummyResults([], (frame_h, frame_w))

        threshold = config.vision.object_recognition_threshold
        try:
            sz = self.imx500.get_input_size()
            if isinstance(sz, (tuple, list)) and len(sz) >= 2:
                input_w, input_h = int(sz[0]), int(sz[1])
            else:
                input_w, input_h = 640, 640
        except Exception:
            input_w, input_h = 640, 640
        postprocess = self.intrinsics.postprocess if self.intrinsics else ""

        is_postprocessed = False
        logger.debug("Object detection postprocess: %s", postprocess)
        if postprocess == "nanodet":
            from picamera2.devices.imx500 import postprocess_nanodet_detection
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=threshold, iou_thres=0.65, max_out_dets=10
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            is_postprocessed = True
        elif postprocess == "yolov8":
            from picamera2.devices.imx500 import postprocess_yolov8_detection
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_yolov8_detection(
                outputs=np_outputs, conf=threshold, iou_thres=0.65, max_out_dets=10
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            is_postprocessed = True
        elif postprocess == "yolov5":
            from picamera2.devices.imx500 import postprocess_yolov5_detection
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_yolov5_detection(
                outputs=np_outputs,
                model_input_shape=(input_h, input_w),
                conf_thres=threshold,
                iou_thres=0.65,
                max_out_dets=10,
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            is_postprocessed = True
        elif postprocess == "efficientdet_lite0":
            from picamera2.devices.imx500 import postprocess_efficientdet_lite0_detection
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_efficientdet_lite0_detection(
                outputs=np_outputs, conf=threshold, iou_thres=0.65, max_out_dets=10
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            is_postprocessed = True
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        dets = []
        for box, score, category in zip(boxes, scores, classes, strict=False):
            if score > threshold:
                det_x1 = det_y1 = det_x2 = det_y2 = 0.0
                converted = False
                if not is_postprocessed and self.imx500 and hasattr(self.imx500, "convert_inference_coords"):
                    try:
                        x, y, w, h = self.imx500.convert_inference_coords(box, metadata, self.picam2)
                        det_x1, det_y1, det_x2, det_y2 = float(x), float(y), float(x + w), float(y + h)
                        converted = True
                    except Exception:
                        converted = False

                if not converted:
                    if is_postprocessed:
                        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
                        det_x1, det_y1, det_x2, det_y2 = xmin * frame_w, ymin * frame_h, xmax * frame_w, ymax * frame_h
                    else:
                        box_arr = np.array(box, dtype=np.float32)
                        if self.intrinsics and self.intrinsics.bbox_normalization:
                            box_arr = box_arr / float(input_h)

                        if self.intrinsics and self.intrinsics.bbox_order == "yx":
                            y0, x0, y1, x1 = box_arr[0], box_arr[1], box_arr[2], box_arr[3]
                        else:
                            x0, y0, x1, y1 = box_arr[0], box_arr[1], box_arr[2], box_arr[3]

                        if not (self.intrinsics and self.intrinsics.bbox_normalization):
                            x0 /= input_w
                            y0 /= input_h
                            x1 /= input_w
                            y1 /= input_h

                        det_x1, det_y1, det_x2, det_y2 = x0 * frame_w, y0 * frame_h, x1 * frame_w, y1 * frame_h

                dets.append(
                    Detection(
                        x1=det_x1,
                        y1=det_y1,
                        x2=det_x2,
                        y2=det_y2,
                        score=float(score),
                        cls=int(category),
                    )
                )

        return DummyResults(dets, frame.shape[:2])

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects using metadata and return standardized format."""
        results = self.infer(frame, metadata=metadata)
        if results is None:
            logger.debug("object not detected")
            return []

        detections: list[DetectionDict] = []
        labels = self.get_labels()

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
            logger.debug(
                "Object detected - label_id = %d, label = %s, score = %f, box = %s",
                label_idx,
                label,
                d.score,
                [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            )

        return detections

    def draw_detections(self, results: Any) -> np.ndarray:
        if self.latest_frame is None:
            import numpy as np

            return np.zeros((*results.orig_shape, 3), dtype=np.uint8)

        annotated_frame = self.latest_frame.copy()

        labels = self.get_labels()

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
