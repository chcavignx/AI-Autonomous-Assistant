"""Face recognition and embedding module using ArcFace."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align

from src.utils.config import Config, load_config

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class ArcFaceRecognizer:
    """ArcFace face recognizer on CPU."""

    def __init__(self, cfg: Config | None = None, model_name: str | None = None, load_model: bool = True) -> None:
        """Initialize the ArcFace recognizer."""
        self.cfg = cfg or load_config()
        self.arcface = None
        if load_model:
            if model_name is None:
                model_name = str(self.cfg.vision.post_processing_model_full_path)
            self.arcface = get_model(model_name)
            if self.arcface is not None:
                self.arcface.prepare(ctx_id=0)
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
        if landmark5 is not None:
            # High-quality alignment using landmark5
            aligned = face_align.norm_crop(frame_bgr, landmark5, image_size=112)
        elif bbox is not None:
            # Fallback: simple bounding box crop
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            x1i = max(int(x1), 0)
            y1i = max(int(y1), 0)
            x2i = min(int(x2), w - 1)
            y2i = min(int(y2), h - 1)
            crop = frame_bgr[y1i:y2i, x1i:x2i]
            aligned = cv2.resize(crop, (112, 112))
        else:
            # Fallback: assume the frame_bgr is already a crop
            aligned = cv2.resize(frame_bgr, (112, 112))

        emb = self.arcface.get_emb(np.asarray(aligned))
        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb)
        return emb

    def register_face(
        self,
        face_id: str,
        img_bgr: np.ndarray,
        landmark5: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Register a known face."""
        if embedding is not None:
            emb = embedding
        else:
            if self.arcface is None:
                msg = "ArcFaceRecognizer was initialized without loading a model, cannot extract embedding."
                raise RuntimeError(msg)
            emb = self.extract_embedding(img_bgr, landmark5=landmark5)
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
