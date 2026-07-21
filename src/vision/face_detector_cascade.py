"""Haar Cascade face detector module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectionDict

if TYPE_CHECKING:
    import numpy as np

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class CascadeFaceDetector(BaseDetector):
    """Face detector using OpenCV's Haar Cascade classifier."""

    def __init__(self, cfg: Config | None = None, cascade_path: str | None = None) -> None:
        """Initialize the Haar Cascade Face Detector.

        Args:
            cfg: Optional Config instance.
            cascade_path: Optional path to Haar Cascade XML file.

        """
        self.cfg = cfg or load_config()
        if cascade_path is None:
            cascade_path = str(self.cfg.vision.face_detector_model_path)

        self.cascade_path = cascade_path
        if not Path(self.cascade_path).exists():
            logger.warning("Cascade file not found at %s. Attempting to load from cv2 defaults.", self.cascade_path)
            self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load cascade classifier from %s", self.cascade_path)

    def detect(self, frame: np.ndarray) -> list[DetectionDict]:
        """Detect faces in BGR frame.

        Args:
            frame: Input BGR image.

        Returns:
            List of standardized DetectionDict.

        """
        if self.face_cascade.empty():
            return []

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detections: list[DetectionDict] = []
        for x, y, w, h in faces:
            detections.append(
                {
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "score": 1.0,
                    "class_id": 0,
                    "label": "face",
                }
            )

        return detections
