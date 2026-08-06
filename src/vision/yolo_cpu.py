"""YOLO CPU detector module."""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectionDict

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class LibreYoloOnnxPredictor:
    """ONNX Runtime fallback for LibreYOLO / YOLOX models when libreyolo package is not installed."""

    def __init__(self, model_path: str, names: dict[int, str] | None = None) -> None:
        """Initialize the YOLOX ONNX Runtime predictor.

        Args:
            model_path: Path to the ONNX model file.
            names: Optional dictionary mapping class IDs to class names.

        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.names = names or {i: f"class_{i}" for i in range(80)}

    def predict(
        self,
        source: np.ndarray,
        conf: float = 0.25,
        verbose: bool = False,
        imgsz: tuple[int, int] | int = (416, 416),
        rect: bool = False,
        device: str = "cpu",
    ) -> list[Results]:
        """Run YOLOX ONNX inference on a BGR frame and return Ultralytics Results."""
        h_orig, w_orig = source.shape[:2]
        if isinstance(imgsz, (list, tuple)):
            h_in, w_in = imgsz
        else:
            h_in = w_in = imgsz

        # Letterbox resize
        r = min(h_in / h_orig, w_in / w_orig)
        rw, rh = round(w_orig * r), round(h_orig * r)
        dw, dh = (w_in - rw) / 2, (h_in - rh) / 2

        resized = cv2.resize(source, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
        top = round(dh - 0.1)
        left = round(dw - 0.1)
        padded[top : top + rh, left : left + rw] = resized

        # BGR to RGB, HWC to CHW
        t0 = time.time()
        blob = padded.transpose(2, 0, 1)[::-1]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        t1 = time.time()

        outputs = self.session.run(None, {self.input_name: blob})
        t2 = time.time()

        preds: np.ndarray = cast("np.ndarray", outputs[0])
        if preds.ndim == 3:
            preds = preds[0]

        # Decode YOLOX anchors
        strides = [8, 16, 32]
        grids = []
        expanded_strides = []
        for stride in strides:
            hsize = h_in // stride
            wsize = w_in // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), stride))

        grid_cat = np.concatenate(grids, axis=0)
        strides_cat = np.concatenate(expanded_strides, axis=0)

        cxcy = (preds[:, :2] + grid_cat) * strides_cat
        wh = np.exp(preds[:, 2:4]) * strides_cat

        obj_conf = preds[:, 4:5]
        cls_scores = preds[:, 5:]
        scores = obj_conf * cls_scores

        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        mask = max_scores >= conf
        if not np.any(mask):
            res = Results(orig_img=source, path=self.model_path, names=self.names, boxes=np.zeros((0, 6)))
            res.speed = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "postprocess": 0.0}
            res.save = lambda filename=None, **kwargs: cv2.imwrite(filename, res.plot()) if filename else None
            return [res]

        b_cxcy = cxcy[mask]
        b_wh = wh[mask]
        b_scores = max_scores[mask]
        b_cls = class_ids[mask]

        # Map back to original image resolution
        x1 = (b_cxcy[:, 0] - b_wh[:, 0] / 2 - dw) / r
        y1 = (b_cxcy[:, 1] - b_wh[:, 1] / 2 - dh) / r
        x2 = (b_cxcy[:, 0] + b_wh[:, 0] / 2 - dw) / r
        y2 = (b_cxcy[:, 1] + b_wh[:, 1] / 2 - dh) / r

        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, b_scores.tolist(), float(conf), 0.45)

        if len(indices) > 0:
            idx = indices.flatten()
            box_data = np.stack([x1[idx], y1[idx], x2[idx], y2[idx], b_scores[idx], b_cls[idx]], axis=1)
        else:
            box_data = np.zeros((0, 6))

        t3 = time.time()
        res = Results(orig_img=source, path=self.model_path, names=self.names, boxes=box_data)
        res.speed = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000}
        res.save = lambda filename=None, **kwargs: cv2.imwrite(filename, res.plot()) if filename else None
        return [res]


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
        self.cfg = cfg
        self.conf_thres = conf_thres if conf_thres is not None else self.cfg.vision.object_recognition_threshold
        self.model_path = str(model_path or self.cfg.vision.object_model_full_path)
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path: str) -> Any:
        """Load standard YOLO or LibreYOLO model based on file path/name."""
        if "libreyolo" in self.model_path.lower():
            try:
                from libreyolo import LibreYOLO

                return LibreYOLO(self.model_path)
            except ImportError:
                logger.info("libreyolo package not installed, using ONNX Runtime YOLOX predictor")
                return LibreYoloOnnxPredictor(self.model_path)
        return YOLO(self.model_path)

    def infer(self, frame_bgr: np.ndarray, metadata: dict | None = None) -> Results:
        """Run inference on a single frame.

        Args:
            frame_bgr: The input frame in BGR format.
            metadata: Optional metadata (ignored in CPU detector, used for API consistency).

        Returns:
            An Ultralytics Results object.

        """
        res_w, res_h = self.cfg.vision.get_model_resolution(self.model_path)
        is_libre = type(self.model).__module__.startswith("libreyolo") or isinstance(self.model, LibreYoloOnnxPredictor)

        predict_kwargs = {
            "source": frame_bgr,
            "conf": self.conf_thres,
            "imgsz": (res_h, res_w),
            "device": "cpu",
        }
        if not is_libre:
            predict_kwargs["verbose"] = self.verbose
            predict_kwargs["rect"] = False

        results = self.model.predict(**predict_kwargs)
        res = results[0]
        if not hasattr(res, "speed") or not res.speed:
            res.speed = {"preprocess": 1.0, "inference": 1.0, "postprocess": 1.0}
        if not hasattr(res, "save") or not callable(getattr(res, "save", None)):

            def _save_fallback(filename: str | None = None, **kwargs: Any) -> None:
                if not filename:
                    return
                img = None
                if hasattr(res, "plot") and callable(getattr(res, "plot", None)):
                    try:
                        img = res.plot()
                    except Exception:
                        img = None
                if img is None:
                    for attr in ("orig_img", "img", "frame"):
                        val = getattr(res, attr, None)
                        if isinstance(val, np.ndarray):
                            img = val
                            break
                if img is None:
                    img = frame_bgr
                if img is not None:
                    cv2.imwrite(filename, img)

            res.save = _save_fallback

        if (
            hasattr(res, "boxes")
            and res.boxes is not None
            and len(res.boxes) > 0
            and hasattr(res, "names")
            and isinstance(res.names, dict)
        ):
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in res.names:
                    res.names[cls_id] = str(cls_id)
        return res

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
    """YOLO detector in NCNN format via Ultralytics.

    - Nano (n) model recommended for Raspberry Pi 5 CPU.
    - YOLO is NMS-free: no NMS on the Python side.
    - The exported NCNN model is called directly.
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
        # Use Ultralytics low-level API to pass a BGR np.ndarray
        # Display and saving are disabled, and only one image is requested.
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
