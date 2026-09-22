"""Face recognition and embedding module using ArcFace."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.utils.config import Config, load_config

try:
    from insightface.model_zoo import get_model
    from insightface.utils import face_align
except ImportError:
    get_model = None  # type: ignore[assignment, misc]
    face_align = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__.split(".")[1])


class ArcFaceRecognizer:
    """ArcFace face recognizer on CPU."""

    def __init__(self, cfg: Config | None = None, model_name: str | None = None, load_model: bool = True) -> None:
        """Initialize the ArcFace recognizer."""
        self.cfg = cfg or load_config()
        self.arcface: Any = None
        if load_model:
            if model_name is None:
                model_name = str(self.cfg.vision.post_processing_model_full_path)
            if get_model is not None:
                try:
                    self.arcface = get_model(model_name, providers=["CPUExecutionProvider"])
                    if self.arcface is not None:
                        self.arcface.prepare(ctx_id=0)
                except Exception as e:
                    logger.warning("Failed to load ArcFace model from %s: %s", model_name, e)
                    self.arcface = None
            else:
                logger.warning("insightface package not installed; ArcFaceRecognizer model could not be loaded")
                self.arcface = None

        self.known_faces: dict[str, np.ndarray] = {}

    def extract_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] | None = None,
        landmark5: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract L2-normalized embedding for a face.

        If landmark5 is provided, aligns the face first.
        Otherwise, crops via bbox (if provided) and resizes.
        """
        aligned = None
        if landmark5 is not None and face_align is not None:
            try:
                aligned = face_align.norm_crop(frame_bgr, landmark5, image_size=112)
            except Exception:
                aligned = None

        if aligned is None:
            if bbox is not None:
                # Fallback: simple bounding box crop
                h, w = frame_bgr.shape[:2]
                x1, y1, x2, y2 = bbox
                x1i = max(int(x1), 0)
                y1i = max(int(y1), 0)
                x2i = min(int(x2), w - 1)
                y2i = min(int(y2), h - 1)
                crop = frame_bgr[y1i:y2i, x1i:x2i]
                aligned = cv2.resize(crop, (112, 112)) if crop.size > 0 else cv2.resize(frame_bgr, (112, 112))
            else:
                # Fallback: assume the frame_bgr is already a crop
                aligned = cv2.resize(frame_bgr, (112, 112))

        if self.arcface is None:
            msg = "ArcFace model is not loaded; cannot extract face embeddings"
            raise RuntimeError(msg)

        if hasattr(self.arcface, "get_feat"):
            emb = self.arcface.get_feat(np.asarray(aligned))
        elif hasattr(self.arcface, "get_emb"):
            emb = self.arcface.get_emb(np.asarray(aligned))
        elif hasattr(self.arcface, "get"):
            emb = self.arcface.get(np.asarray(aligned))
        else:
            msg = "ArcFace model does not have a recognized embedding extraction method (get_feat/get_emb/get)"
            raise AttributeError(msg)

        emb = np.asarray(emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def register_face(
        self,
        face_id: str,
        img_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] | None = None,
        landmark5: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Register a known face."""
        emb = embedding if embedding is not None else self.extract_embedding(img_bgr, bbox=bbox, landmark5=landmark5)
        self.known_faces[face_id] = emb
        logger.info("Face registered in ArcFaceRecognizer: %s", face_id)

    def match_face(self, emb: np.ndarray, thresh: float = 0.4) -> tuple[str | None, float | None]:
        """Match an embedding against registered faces using cosine similarity."""
        best_id, best_sim = None, 0.0
        for fid, known_emb in self.known_faces.items():
            sim = float(np.dot(emb, known_emb))
            if sim > best_sim:
                best_sim, best_id = sim, fid

        if best_id is not None and best_sim >= thresh:
            return best_id, best_sim
        return None, None
