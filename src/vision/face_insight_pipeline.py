"""Unified Face Detection and Recognition Pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectedFace, DetectionDict
from src.vision.face_detector import CASCADE_FALLBACK_PATH, CascadeFaceDetector
from src.vision.face_recognizer import ArcFaceRecognizer

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__.split(".")[1])


class FaceInsightPipeline(BaseDetector):
    """Unified Face Detection and Recognition wrapper pipeline.

    Conforms to BaseDetector for plug-and-play orchestration.
    Dynamically initializes the configured detector (cpu (Cascade and Insightface), Hailo, or IMX500)
    and pairs it with ArcFace for identity embeddings.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        det_size: tuple[int, int] = (640, 640),
        imx500: Path | None = None,
        arcface_model_name: str | None = None,
        detector_type: str | None = None,
    ) -> None:
        """Initialize the FaceInsightPipeline instance."""
        self.cfg = cfg or load_config()
        self.detector_type = (detector_type or self.cfg.vision.face_detector_type).lower()
        self.detector: Any = None

        # Initialize detector based on configuration
        if self.detector_type == "cascade":
            self.detector = CascadeFaceDetector(self.cfg)

        elif self.detector_type == "hailo":
            try:
                from src.vision.face_detector_hailo import HailoFaceDetector

                self.detector = HailoFaceDetector(self.cfg)
            except Exception as e:
                logger.warning("Failed to initialize HailoFaceDetector (%s); falling back to CascadeFaceDetector", e)
                self.detector = CascadeFaceDetector(self.cfg, cascade_path=CASCADE_FALLBACK_PATH)

        elif self.detector_type == "imx500":
            try:
                from src.vision.face_detector_imx import Imx500Detector

                self.detector = Imx500Detector(self.cfg, imx500=imx500)
            except Exception as e:
                logger.warning("Failed to initialize Imx500Detector (%s); falling back to CascadeFaceDetector", e)
                self.detector = CascadeFaceDetector(self.cfg, cascade_path=CASCADE_FALLBACK_PATH)

        elif self.detector_type == "insightface":
            try:
                from src.vision.face_detector import InsightFaceDetector

                self.detector = InsightFaceDetector(self.cfg, det_size)
            except Exception as e:
                logger.warning("Failed to initialize InsightFaceDetector (%s); falling back to CascadeFaceDetector", e)
                self.detector = CascadeFaceDetector(self.cfg, cascade_path=CASCADE_FALLBACK_PATH)

        else:
            logger.warning("Unknown detector_type '%s'; falling back to CascadeFaceDetector", self.detector_type)
            self.detector = CascadeFaceDetector(self.cfg, cascade_path=CASCADE_FALLBACK_PATH)

        logger.info("Face detector initialized with backend: %s", self.detector_type)

        # Initialize recognizer
        load_model = self.cfg.vision.post_processing_enabled or self.detector_type != "insightface"
        if arcface_model_name is None:
            arcface_model_name = str(self.cfg.vision.post_processing_model_full_path)
        self.recognizer = ArcFaceRecognizer(cfg=self.cfg, model_name=arcface_model_name, load_model=load_model)
        logger.info("Face recognizer loaded successfully.")

    @property
    def known_faces(self) -> dict[str, np.ndarray]:
        """Expose known_faces registry for backward compatibility."""
        return self.recognizer.known_faces

    @known_faces.setter
    def known_faces(self, value: dict[str, np.ndarray]) -> None:
        self.recognizer.known_faces = value

    def register_face(self, face_id: str, img_bgr: np.ndarray) -> None:
        """Register a face from a BGR image."""
        faces: list[DetectedFace] = []
        if hasattr(self.detector, "detect_faces_raw"):
            faces = self.detector.detect_faces_raw(img_bgr)
        elif hasattr(self.detector, "detect"):
            detections = self.detector.detect(img_bgr)
            faces = [
                DetectedFace(
                    bbox=(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3])),
                    landmark5=None,
                    score=float(d.get("score", 1.0)),
                )
                for d in detections
            ]

        if faces:
            # Sort to find largest face
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            landmark5 = faces[0].landmark5
            bbox = faces[0].bbox
            embedding = getattr(faces[0], "embedding", None)
            if self.post_processing_enabled:
                embedding = None  # Force re-extraction with ArcFace

            self.recognizer.register_face(
                face_id,
                img_bgr,
                bbox=bbox,
                landmark5=landmark5,
                embedding=embedding,
            )
        elif self.detector_type == "insightface":
            msg = "No face detected for registration."
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            # Fallback direct registration
            self.recognizer.register_face(face_id, img_bgr)

    def recognize(
        self,
        frame_bgr: np.ndarray,
        thresh: float = 0.4,
        metadata: dict | None = None,
    ) -> list[DetectedFace]:
        """Detect and recognize faces in a BGR frame."""
        if hasattr(self.detector, "detect_faces_metadata") and metadata is not None:
            faces = self.detector.detect_faces_metadata(frame_bgr, metadata)
        elif hasattr(self.detector, "detect_faces_raw"):
            faces = self.detector.detect_faces_raw(frame_bgr)
        elif hasattr(self.detector, "detect"):
            detections = self.detector.detect(frame_bgr, metadata=metadata)
            faces = [
                DetectedFace(
                    bbox=(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3])),
                    landmark5=None,
                    score=float(d.get("score", 1.0)),
                )
                for d in detections
            ]
        else:
            faces = []

        results: list[DetectedFace] = []
        for f in faces:
            # Extract embedding using landmarks if available or bbox crop
            emb = getattr(f, "embedding", None)
            if (self.cfg.vision.post_processing_enabled or emb is None) and self.recognizer.arcface is not None:
                try:
                    emb = self.recognizer.extract_embedding(frame_bgr, bbox=f.bbox, landmark5=f.landmark5)
                except Exception as e:
                    logger.debug("Failed to extract embedding with ArcFace: %s", e)
                    emb = None

            best_id, best_sim = None, None
            if emb is not None:
                best_id, best_sim = self.recognizer.match_face(emb, thresh)

            results.append(
                DetectedFace(
                    bbox=f.bbox,
                    landmark5=f.landmark5,
                    score=f.score,
                    identity=best_id,
                    similarity=best_sim,
                    embedding=emb,
                )
            )

        logger.debug("Recognized %d faces", len(results))
        return results

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect faces in BGR frame using configured detector.

        Args:
            frame: Input BGR image.
            metadata: Optional camera metadata.

        Returns:
            List of standardized DetectionDict.

        """
        if hasattr(self.detector, "detect"):
            return self.detector.detect(frame, metadata=metadata)
        return []

    def stop(self) -> None:
        """Release underlying detector resources."""
        if hasattr(self.detector, "stop"):
            self.detector.stop()
