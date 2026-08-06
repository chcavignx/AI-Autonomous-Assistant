"""Frame-level object detection pipeline using YOLO."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cv2

from src.utils.config import Config, load_config
from src.vision.yolo_cpu import YoloCpuDetector

if TYPE_CHECKING:
    import numpy as np

    from src.vision.base import DetectionDict

logger = logging.getLogger(__name__)


class ObjectInsightFrame:
    """Processes a single frame for object detection using YOLO."""

    def __init__(
        self,
        cfg: Config | None = None,
        detector: Any | None = None,
    ) -> None:
        """Initialize the ObjectInsightFrame processor.

        Args:
            cfg: Configuration object.
            detector: Optional pre-configured detector.

        """
        self.cfg = cfg or load_config()
        if detector is not None:
            self.detector = detector
        elif self.cfg.vision.object_model_type == "yolo_imx500":
            from src.vision.yolo_imx500 import Imx500Detector

            self.detector = Imx500Detector()
        else:
            self.detector = YoloCpuDetector(self.cfg)

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
        metadata: dict | None = None,
    ) -> tuple[np.ndarray, list[DetectionDict], Any]:
        """Detect objects in the frame.

        Args:
            frame: Input BGR image.
            draw: Whether to draw bounding boxes and labels on the frame.
            metadata: Optional metadata for IMX500.

        Returns:
            Tuple of (annotated_frame, list of DetectionDict, raw results object).

        """
        annotated_frame = frame.copy() if draw else frame

        # We need both the raw results (for speed metrics) and standardized detections
        results = self.detector.infer(frame, metadata=metadata)

        # For efficiency, if the detector supports getting detections directly from results or
        # if we just call detect. However, calling detect might trigger inference again.
        # We can implement a quick extraction here or check if the detector cached results.
        # Let's extract standardized detections here or rely on the detector's detect method
        # if it's cheap (IMX500 metadata parsing is cheap).
        # Actually, let's just use self.detector.detect which we will optimize if needed.
        # But wait, yolo_cpu.detect runs infer again! Let's just extract it if it's Results.

        detections: list[DetectionDict] = []
        if results is not None:
            if hasattr(results, "detections") and isinstance(getattr(results, "detections", None), list):
                # DummyResults from IMX500
                if hasattr(self.detector, "get_labels"):
                    labels = self.detector.get_labels()
                else:
                    intrinsics = getattr(self.detector, "intrinsics", None)
                    labels = intrinsics.labels if intrinsics and intrinsics.labels else []
                    if intrinsics and getattr(intrinsics, "ignore_dash_labels", False):
                        labels = [lbl for lbl in labels if lbl and lbl != "-"]
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
            elif hasattr(results, "boxes") and results.boxes is not None:
                # Ultralytics Results
                for box in results.boxes:
                    xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    # Handle dummy model names or missing keys safely
                    model = getattr(self.detector, "model", None)
                    names = getattr(model, "names", None)
                    if (names and isinstance(names, dict) and cls in names) or (
                        names and isinstance(names, (list, tuple)) and 0 <= cls < len(names)
                    ):
                        label = names[cls]
                    else:
                        label = str(cls)
                    detections.append(
                        {
                            "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "score": conf,
                            "class_id": cls,
                            "label": label,
                        }
                    )

        if draw:
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                score = det["score"]
                label = det["label"]

                color = (0, 255, 0)  # Green for detected objects
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} ({score:.2f})"
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    color,
                    1,
                )

        return annotated_frame, detections, results
