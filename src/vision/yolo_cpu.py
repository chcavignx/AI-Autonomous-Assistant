"""YOLO CPU detector module."""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from ultralytics import YOLO

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectionDict

if TYPE_CHECKING:
    import numpy as np
    from ultralytics.engine.results import Results

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class YoloCpuDetector(BaseDetector):
    """YOLO Detector using Ultralytics API.

    Optimized for CPU usage (RPi 5, etc.).
    Supports multiple formats (PT, ONNX, NCNN).
    """

    def __init__(
        self,
        cfg: Config | None = None,
        model_path: str | Path | None = None,
        conf_thres: float | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            cfg: Optional Config instance.
            model_path: Path to the model file. If None, uses the one from config.
            conf_thres: Confidence threshold for detections.
            verbose: Whether to print Ultralytics logging.

        """
        self.verbose = verbose
        if cfg is None:
            cfg = load_config()
        self.conf_thres = conf_thres if conf_thres is not None else cfg.vision.object_recognition_threshold
        self.model_path = str(model_path or cfg.vision.object_model_full_path)
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path: str) -> Any:
        """Load standard YOLO or LibreYOLO model based on file path/name."""
        if "libreyolo" in self.model_path.lower():
            try:
                from libreyolo import LibreYOLO

                return LibreYOLO(self.model_path)
            except ImportError:
                logger.warning("libreyolo not installed, falling back to ultralytics YOLO")
        return YOLO(self.model_path)

    def infer(self, frame_bgr: np.ndarray, metadata: dict | None = None) -> Results:
        """Run inference on a single frame.

        Args:
            frame_bgr: The input frame in BGR format.
            metadata: Optional metadata (ignored in CPU detector, used for API consistency).

        Returns:
            An Ultralytics Results object.

        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            verbose=self.verbose,
            device="cpu",
        )
        return results[0]

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects in a BGR frame and return standardized results.

        Args:
            frame: Input frame in BGR format.
            metadata: Optional metadata.

        Returns:
            List of standardized detection dictionaries.

        """
        results = self.infer(frame, metadata=metadata)
        detections: list[DetectionDict] = []
        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls] if hasattr(self.model, "names") and cls in self.model.names else str(cls)
                detections.append(
                    {
                        "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "score": conf,
                        "class_id": cls,
                        "label": label,
                    }
                )
        return detections

    def change_model(self, model_path: str) -> None:
        """Change the underlying model.

        Args:
            model_path: New path to the model file.

        """
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def benchmark(self, frame_bgr: np.ndarray, iterations: int = 100) -> dict[str, float]:
        """Run a benchmark on the current model.

        Args:
            frame_bgr: Sample frame for benchmarking.
            iterations: Number of iterations to run.

        Returns:
            Dictionary with benchmark results (avg_ms, fps).

        """
        # Warmup
        self.infer(frame_bgr)

        start_time = time.time()
        for _ in range(iterations):
            self.infer(frame_bgr)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        fps = 1 / (total_time / iterations)

        return {"avg_ms": avg_time_ms, "fps": fps}

    @staticmethod
    def draw_detections(results: Results) -> np.ndarray:
        """Draw detections on the frame using Ultralytics plot().

        Args:
            results: The Results object from infer().

        Returns:
            Annotated frame.

        """
        return results.plot()


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


class Yolo26NcnnDetector(BaseDetector):
    """Détecteur YOLO au format NCNN via Ultralytics.

    - Modèle nano (n) recommandé pour le CPU Pi 5.
    - YOLO est NMS-free : pas de NMS côté Python.
    - On appelle directement le modèle NCNN exporté.
    """

    def __init__(self, cfg: Config | None = None, conf_thres: float | None = None) -> None:
        """Initialize the YOLO NCNN detector."""
        if cfg is None:
            cfg = load_config()
        self.model_path = str(cfg.vision.object_model_full_path)
        self.model = YOLO(self.model_path)
        self.conf_thres = conf_thres if conf_thres is not None else cfg.vision.object_recognition_threshold

    def infer(self, frame_bgr: np.ndarray, metadata: dict | None = None) -> list[Detection]:
        """Run raw inference on a single frame.

        Args:
            frame_bgr: The input frame in BGR format.
            metadata: Optional metadata (ignored in CPU detector).

        Returns:
            List of Detection objects.

        """
        # Utiliser l'API low-level de Ultralytics pour passer un np.ndarray BGR
        # On désactive l'affichage, la sauvegarde, et on demande une seule image.
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            verbose=False,
            imgsz=max(frame_bgr.shape[0], frame_bgr.shape[1]),
            max_det=300,
        )

        if not results:
            return []

        r = results[0]
        boxes = r.boxes  # Boxes object
        dets: list[Detection] = []

        # r.orig_shape = (h, w), r.boxes.xyxy en pixels sur cette image
        # Format: xyxy, conf, cls
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss, strict=False):
            dets.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(score),
                    cls=int(cls_id),
                )
            )

        return dets

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects in standard DetectionDict format."""
        dets = self.infer(frame, metadata=metadata)
        detections: list[DetectionDict] = []
        for d in dets:
            label = (
                self.model.names[d.cls] if hasattr(self.model, "names") and d.cls in self.model.names else str(d.cls)
            )
            detections.append(
                {
                    "box": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                    "score": d.score,
                    "class_id": d.cls,
                    "label": label,
                }
            )
        return detections

    @staticmethod
    def draw_detections(frame_bgr: np.ndarray, dets: list[Detection], names: dict | list | None = None) -> np.ndarray:
        """Annotate the frame with bounding boxes and labels."""
        out = frame_bgr.copy()
        for d in dets:
            cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            label = f"{d.cls}:{d.score:.2f}"
            cv2.putText(
                out, label, (int(d.x1), int(d.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
        return out
