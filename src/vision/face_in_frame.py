"""Frame-level face detection and recognition pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2

from src.utils.config import Config, load_config
from src.vision.face_detector_cascade import CascadeFaceDetector
from src.vision.face_insight_pipeline import FaceInsightPipeline

if TYPE_CHECKING:
    import numpy as np

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class FaceInFrame:
    """Processes a single frame for face detection and identification."""

    def __init__(
        self,
        cfg: Config | None = None,
        detector_type: str = "cascade",
        face_recognizer: FaceInsightPipeline | None = None,
    ) -> None:
        """Initialize the FaceInsightFrame processor.

        Args:
            cfg: Configuration object.
            detector_type: Detector backend type ("cascade", "insightface", "imx500").
            face_recognizer: Optional shared face recognizer instance.

        """
        self.cfg = cfg or load_config()
        self.detector_type = detector_type.lower()
        self.face_recognizer = face_recognizer or FaceInsightPipeline(self.cfg)

        # Initialize detector based on type
        if self.detector_type == "cascade":
            self.detector = CascadeFaceDetector(self.cfg)
        elif self.detector_type in {"insightface", "imx500"}:
            self.detector = None
        else:
            logger.warning("Unknown detector type '%s'. Falling back to 'cascade'.", detector_type)
            self.detector_type = "cascade"
            self.detector = CascadeFaceDetector(self.cfg)

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

        if self.detector_type in {"insightface", "imx500"}:
            try:
                faces = self.face_recognizer.recognize(frame, thresh=thresh, metadata=metadata)
                for f in faces:
                    x1, y1, x2, y2 = [int(v) for v in f.bbox]
                    name = f.identity if f.identity is not None else "unknown"
                    best_sim = f.similarity if f.similarity is not None else 0.0

                    recognized_faces.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "name": name,
                            "similarity": best_sim,
                        }
                    )

                    if draw:
                        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} ({best_sim:.2f})"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            color,
                            1,
                        )
            except Exception as e:
                logger.exception("Error in native %s processing: %s", self.detector_type, e)
                return annotated_frame, recognized_faces

        else:
            if self.detector is None:
                return annotated_frame, recognized_faces

            detections = self.detector.detect(frame)
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                name = "unknown"
                best_sim = 0.0

                recognized_faces.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "name": name,
                        "similarity": best_sim,
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
