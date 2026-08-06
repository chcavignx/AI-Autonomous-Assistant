"""Unified Face Detection and Recognition Pipeline."""

from __future__ import annotations

import logging
import pathlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectionDict
from src.vision.face_detector import DetectedFace, InsightFaceDetector
from src.vision.face_recognizer import ArcFaceRecognizer

if TYPE_CHECKING:
    import numpy as np

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class FaceInsightPipeline(BaseDetector):
    """Unified Face Detection and Recognition wrapper pipeline.

    Conforms to BaseDetector for plug-and-play orchestration.
    Dynamically initializes the appropriate detector (InsightFace or IMX500)
    and pairs it with ArcFace for embeddings.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        det_size: tuple[int, int] = (640, 640),
        imx500: Path | None = None,
        arcface_model_name: str | None = None,
    ) -> None:
        """Initialize the FaceInsightPipeline instance."""
        self.cfg = cfg or load_config()
        self.detector_type = self.cfg.vision.face_detector_type.lower()

        # Initialize detector based on config
        if self.detector_type == "imx500":
            from src.vision.face_detector import Imx500Detector

            if imx500 is None:
                imx500 = self.cfg.vision.face_detector_model_path
            self.detector = Imx500Detector(self.cfg, imx500=imx500)
        elif self.detector_type == "insightface":
            self.detector = InsightFaceDetector(self.cfg, det_size)
        else:
            logger.warning(
                "FaceInsightPipeline instantiated with detector_type='%s', which is not directly supported by this pipeline wrapper natively. Defaulting to InsightFace.",
                self.detector_type,
            )
            self.detector = InsightFaceDetector(self.cfg, det_size)
        logger.info("Face detector type: %s", self.detector_type)
        self.post_processing_enabled = self.cfg.vision.post_processing_enabled

        # We need a separate ArcFace model only if post_processing is enabled
        # or if the detector doesn't supply embeddings (e.g., IMX500).
        load_model = self.post_processing_enabled or self.detector_type != "insightface"
        logger.info("Load post-processing model: %s", load_model)
        # Initialize recognizer
        if arcface_model_name is None:
            arcface_model_name = str(self.cfg.vision.post_processing_model_full_path)
        self.recognizer = ArcFaceRecognizer(model_name=arcface_model_name, load_model=load_model)
        logger.info("Load face recognizer successfully.")

    @property
    def known_faces(self) -> dict[str, np.ndarray]:
        """Expose known_faces registry for backward compatibility."""
        return self.recognizer.known_faces

    @known_faces.setter
    def known_faces(self, value: dict[str, np.ndarray]) -> None:
        self.recognizer.known_faces = value

    def register_face(self, face_id: str, img_bgr: np.ndarray) -> None:
        """Register a face from a BGR image."""
        if self.detector_type == "insightface":
            faces = self.detector.detect_faces_raw(img_bgr)
            if not faces:
                msg = "No face detected for registration."
                logger.error(msg)
                raise RuntimeError(msg)

            # Sort to find the largest face
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

            embedding = getattr(faces[0], "embedding", None)
            if self.post_processing_enabled:
                embedding = None  # Force re-extraction

            self.recognizer.register_face(face_id, img_bgr, landmark5=faces[0].landmark5, embedding=embedding)
        else:
            # For IMX500 or others, rely on recognizer's fallback CPU detection logic
            self.recognizer.register_face(face_id, img_bgr)

    def recognize(self, frame_bgr: np.ndarray, thresh: float = 0.4, metadata: dict | None = None) -> list[DetectedFace]:
        """Detect and recognize faces in a BGR frame."""
        if hasattr(self.detector, "detect_faces_metadata") and metadata is not None:
            faces = self.detector.detect_faces_metadata(frame_bgr, metadata)
        else:
            faces = self.detector.detect_faces_raw(frame_bgr)

        results: list[DetectedFace] = []
        for f in faces:
            # Extract embedding using landmarks if available
            emb = getattr(f, "embedding", None)
            if self.post_processing_enabled or emb is None:
                emb = self.recognizer.extract_embedding(frame_bgr, bbox=f.bbox, landmark5=f.landmark5)

            best_id, best_sim = self.recognizer.match_face(emb, thresh)

            if best_id is not None:
                results.append(
                    DetectedFace(
                        bbox=f.bbox,
                        landmark5=f.landmark5,
                        score=f.score,
                        identity=best_id,
                        similarity=best_sim,
                    )
                )
            else:
                results.append(
                    DetectedFace(
                        bbox=f.bbox,
                        landmark5=f.landmark5,
                        score=f.score,
                        identity=None,
                        similarity=None,
                    )
                )

        logger.info("Recognized %d faces", len(results))
        return results

    def detect(self, frame: np.ndarray) -> list[DetectionDict]:
        """Detect faces in BGR frame using native configured detector.

        Args:
            frame: Input BGR image.

        Returns:
            List of standardized DetectionDict.

        """
        return self.detector.detect(frame)

    def stop(self) -> None:
        if hasattr(self.detector, "stop"):
            self.detector.stop()
