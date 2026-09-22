"""Frame-level face detection and recognition pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

from src.utils.config import Config, load_config
from src.vision.face_insight_pipeline import FaceInsightPipeline

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__.split(".")[1])


class FaceInsightFrame:
    """Processes a single frame for face detection and identification."""

    def __init__(
        self,
        cfg: Config | None = None,
        detector_type: str | None = None,
        face_recognizer: FaceInsightPipeline | None = None,
        enable_recognition: bool | None = None,
    ) -> None:
        """Initialize the FaceInsightFrame processor.

        Args:
            cfg: Configuration object.
            detector_type: Optional detector backend type ("cascade", "insightface", "hailo", "imx500").
            face_recognizer: Optional shared face recognizer / pipeline instance.
            enable_recognition: Optional flag to enable/disable face recognition mode.

        """
        self.cfg = cfg or load_config()
        if enable_recognition is not None:
            self.cfg.vision.enable_face_recognition = enable_recognition

        raw_type = (detector_type or self.cfg.vision.face_detector_type).lower()
        if raw_type not in {"cascade", "hailo", "insightface", "imx500"}:
            logger.warning("Unknown detector type '%s'. Falling back to 'cascade'.", detector_type)
            self.detector_type = "cascade"
        else:
            self.detector_type = raw_type

        if self.detector_type == "cascade":
            from src.vision.face_detector import CascadeFaceDetector

            self.detector = CascadeFaceDetector(self.cfg)
        elif self.detector_type == "hailo":
            from src.vision.face_detector_hailo import HailoFaceDetector

            self.detector = HailoFaceDetector(self.cfg)
        elif self.detector_type == "imx500":
            self.detector = None
        else:
            from src.vision.face_detector import InsightFaceDetector

            self.detector = InsightFaceDetector(self.cfg)

        self.face_recognizer = face_recognizer or FaceInsightPipeline(self.cfg, detector_type=self.detector_type)

    def process_frame(
        self,
        frame: np.ndarray,
        thresh: float = 0.4,
        draw: bool = True,
        metadata: dict | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Detect and recognize faces in the frame.

        Args:
            frame: Input BGR image.
            thresh: Confidence threshold for face recognition.
            draw: Whether to draw bounding boxes and labels on the frame.
            metadata: Metadata from the frame, if any.

        Returns:
            Tuple of (annotated_frame, list of recognized face dicts).

        """
        annotated_frame = frame.copy() if draw else frame
        recognized_faces: list[dict] = []

        if self.cfg.vision.enable_face_recognition:
            faces = None
            try:
                faces = self.face_recognizer.recognize(frame, thresh=thresh, metadata=metadata)
            except Exception as e:
                logger.exception("Error in %s face recognition: %s", self.detector_type, e)

            if faces is not None:
                for f in faces:
                    x1, y1, x2, y2 = [int(v) for v in f.bbox]
                    name = f.identity if f.identity is not None else "unknown"
                    best_sim = f.similarity if f.similarity is not None else 0.0

                    recognized_faces.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "name": name,
                            "similarity": best_sim,
                            "score": f.score,
                        }
                    )

                    if draw:
                        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} ({best_sim:.2f})" if name != "unknown" else "face"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            color,
                            1,
                        )
                return annotated_frame, recognized_faces

        active_detector = self.detector
        detections = active_detector.detect(frame, metadata=metadata) if hasattr(active_detector, "detect") else []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            name = "unknown"
            best_sim = 0.0

            recognized_faces.append(
                {
                    "box": [x1, y1, x2, y2],
                    "name": name,
                    "similarity": best_sim,
                    "score": det.get("score", 1.0),
                }
            )

            if draw:
                color = (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = "face"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    color,
                    1,
                )

        return annotated_frame, recognized_faces

    def process_frame_metadata(
        self,
        frame: np.ndarray,
        metadata: dict | None = None,
        thresh: float = 0.4,
        draw: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:
        """Process frame using camera metadata."""
        return self.process_frame(frame, thresh=thresh, draw=draw, metadata=metadata)

    def stop(self) -> None:
        """Release underlying detector and recognizer resources."""
        if hasattr(self.detector, "stop"):
            self.detector.stop()
        if hasattr(self.face_recognizer, "stop"):
            self.face_recognizer.stop()
