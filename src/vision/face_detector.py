"""Face detection modules (CPU: RetinaFace and Haar Cascade)."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectedFace, DetectionDict

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__.split(".")[1])

with contextlib.suppress(ImportError):
    pass

CASCADE_FALLBACK_PATH = "haarcascade_frontalface_default.xml"


class InsightFaceDetector(BaseDetector):
    """InsightFace CPU detector (RetinaFace)."""

    def __init__(self, cfg: Config | None = None, det_size: tuple[int, int] = (640, 640)) -> None:
        """Initialize the InsightFace CPU detector."""
        self.cfg = cfg or load_config()
        model_name = self.cfg.vision.face_model_name or "buffalo_l"
        if model_name.endswith((".rpk", ".xml", ".hef")):
            model_name = "buffalo_l"

        root_dir = self.cfg.paths.models_vision_path / "insightface"
        if not (root_dir / model_name).exists():
            alt_dir = self.cfg.paths.models_vision_path / self.cfg.vision.face_detector_type
            if (alt_dir / model_name).exists():
                root_dir = alt_dir
            elif (self.cfg.paths.models_vision_path / "insightface" / "buffalo_l").exists():
                root_dir = self.cfg.paths.models_vision_path / "insightface"
                model_name = "buffalo_l"

        if not (root_dir / model_name).exists():
            msg = f"InsightFace model '{model_name}' not found in {root_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if FaceAnalysis is None:
            msg = "insightface is required for InsightFaceDetector. Please install insightface."
            logger.error(msg)
            raise ImportError(msg)

        self.app = FaceAnalysis(
            name=model_name,
            root=str(root_dir),
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in BGR frame returning raw DetectedFace instances."""
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

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect faces in BGR frame returning standardized DetectionDict list."""
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


class CascadeFaceDetector(BaseDetector):
    """Face detector using OpenCV's Haar Cascade classifier."""

    @staticmethod
    def _resolve_default_cascade_path() -> str:
        """Resolve system default OpenCV Haar Cascade path."""
        haarcascades = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if haarcascades:
            default_path = Path(haarcascades) / "haarcascade_frontalface_default.xml"
            if default_path.exists():
                return str(default_path)
        return CASCADE_FALLBACK_PATH

    def __init__(self, cfg: Config | None = None, cascade_path: str | Path | None = None) -> None:
        """Initialize the Haar Cascade Face Detector.

        Args:
            cfg: Optional Config instance.
            cascade_path: Optional path to Haar Cascade XML file.

        """
        self.cfg = cfg or load_config()
        raw_path = cascade_path if cascade_path is not None else self.cfg.vision.face_detector_model_path
        candidate_path = Path(raw_path) if raw_path else None

        default_cascade = self._resolve_default_cascade_path()

        if candidate_path is not None:
            if candidate_path.suffix.lower() != ".xml":
                logger.warning(
                    "CascadeFaceDetector requires an XML model file, but received '%s'. Using cv2 default: %s",
                    candidate_path,
                    default_cascade,
                )
                self.cascade_path = default_cascade
            elif not candidate_path.exists():
                logger.warning(
                    "Cascade file not found at %s. Using cv2 default: %s",
                    candidate_path,
                    default_cascade,
                )
                self.cascade_path = default_cascade
            else:
                self.cascade_path = str(candidate_path)
        else:
            self.cascade_path = default_cascade

        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        except Exception as e:
            logger.warning("Failed to initialize CascadeClassifier from %s: %s", self.cascade_path, e)
            self.face_cascade = cv2.CascadeClassifier()

        if self.face_cascade.empty():
            logger.error("Failed to load cascade classifier from %s", self.cascade_path)

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect faces in BGR frame.

        Args:
            frame: Input BGR image.
            metadata: Optional metadata (ignored in cascade detector).

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

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in BGR frame returning DetectedFace list."""
        detections = self.detect(frame)
        return [
            DetectedFace(
                bbox=(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3])),
                landmark5=None,
                score=float(d.get("score", 1.0)),
            )
            for d in detections
        ]
