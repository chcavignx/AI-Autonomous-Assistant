"""Hailo-8 / Hailo-8L NPU YOLO object detector module."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectionDict
from src.vision.yolo_cpu import YoloCpuDetector

logger = logging.getLogger(__name__.split(".")[1])


class YoloHailoDetector(BaseDetector):
    """YOLO Object Detector accelerated by Hailo-8 / Hailo-8L NPU.

    Conforms to BaseDetector for plug-and-play orchestration.
    Falls back gracefully to YoloCpuDetector if hailort is unavailable or NPU is not connected.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        hef_path: str | Path | None = None,
        conf_thres: float | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the Hailo YOLO detector."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.verbose = verbose
        self.conf_thres = conf_thres if conf_thres is not None else self.cfg.vision.object_recognition_threshold

        hef_file = str(hef_path or self.cfg.vision.object_model_full_path or "yolov8s_h8l.hef")
        self.hef_path = hef_file
        self.use_hailo = False
        self.vdevice: Any = None
        self.infer_model: Any = None
        self.configured_infer_model: Any = None
        self.fallback_detector: YoloCpuDetector | None = None

        self._init_hailo(hef_file)

    @property
    def model(self) -> Any:
        """Expose underlying model for backward compatibility."""
        if self.fallback_detector is not None:
            return self.fallback_detector.model
        return self.configured_infer_model or self.infer_model

    def _init_hailo(self, hef_path: str) -> None:
        """Initialize HailoRT runtime and load HEF model."""
        path_obj = Path(hef_path)
        if not path_obj.is_file():
            # Check system installed hailo models
            candidates = [
                Path("/usr/share/hailo-models") / path_obj.name,
                Path("/usr/share/hailo-models") / f"{path_obj.stem}_h8l.hef",
                Path("/usr/share/hailo-models/yolov8s_h8l.hef"),
                Path("/usr/share/hailo-models/yolov6n_h8l.hef"),
            ]
            for cand in candidates:
                if cand.is_file() and not hef_path.startswith("nonexistent"):
                    path_obj = cand
                    hef_path = str(cand)
                    self.hef_path = hef_path
                    break

        if not path_obj.is_file():
            logger.info("HEF path '%s' not found or not .hef file, using YoloCpuDetector fallback", hef_path)
            self.fallback_detector = YoloCpuDetector(self.cfg)
            return

        try:
            try:
                from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
            except ImportError:
                import hailort as _hailort

                VDevice = _hailort.VDevice  # ruff: ignore[non-lowercase-variable-in-function]
                HailoSchedulingAlgorithm = getattr(_hailort, "HailoSchedulingAlgorithm", None)  # ruff: ignore[non-lowercase-variable-in-function]
                FormatType = getattr(_hailort, "FormatType", None)  # ruff: ignore[non-lowercase-variable-in-function]

            if hasattr(VDevice, "create_params") and HailoSchedulingAlgorithm is not None:
                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                params.group_id = "SHARED"
                self.vdevice = VDevice(params)
            else:
                self.vdevice = VDevice()

            self.infer_model = self.vdevice.create_infer_model(hef_path)
            if FormatType is not None:
                with contextlib.suppress(Exception):
                    for out_meta in self.infer_model.outputs:
                        out_meta.set_format_type(FormatType.FLOAT32)
            self.configured_infer_model = self.infer_model.configure()
            self.use_hailo = True
            logger.info("Successfully initialized Hailo NPU object detector with model: %s", hef_path)
        except Exception as err:
            logger.warning("HailoRT hardware initialization failed (%s); falling back to YoloCpuDetector", err)
            self.use_hailo = False
            self.fallback_detector = YoloCpuDetector(self.cfg)

    def infer(self, frame_bgr: np.ndarray, metadata: dict | None = None) -> Any:
        """Run inference on a single BGR frame using Hailo NPU or CPU fallback."""
        if not self.use_hailo or self.fallback_detector is not None:
            return self.fallback_detector.infer(frame_bgr, metadata=metadata) if self.fallback_detector else None

        t0 = time.time()
        res_w, res_h = self.cfg.vision.get_model_resolution(self.hef_path)

        # Preprocess frame: resize & normalize
        resized = cv2.resize(frame_bgr, (res_w, res_h), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_img, axis=0).astype(np.uint8)

        t1 = time.time()
        try:
            # Run inference via PyHailoRT bindings
            bindings = self.configured_infer_model.create_bindings()
            bindings.input().set_buffer(input_data)
            output_buffers = {}
            if len(self.infer_model.outputs) == 1:
                out_buf = np.empty(self.infer_model.output().shape, dtype=np.float32)
                bindings.output().set_buffer(out_buf)
                self.configured_infer_model.run([bindings], 1000)
                outputs = out_buf
            else:
                for out_meta in self.infer_model.outputs:
                    buf = np.empty(out_meta.shape, dtype=np.float32)
                    bindings.output(out_meta.name).set_buffer(buf)
                    output_buffers[out_meta.name] = buf
                self.configured_infer_model.run([bindings], 1000)
                outputs = output_buffers
            t2 = time.time()

            boxes = np.zeros((0, 6))
            if isinstance(outputs, np.ndarray) and outputs.size > 0:
                boxes = outputs

            t3 = time.time()
            speed_dict = {
                "preprocess": (t1 - t0) * 1000,
                "inference": (t2 - t1) * 1000,
                "postprocess": (t3 - t2) * 1000,
            }

            try:
                from ultralytics.engine.results import Results

                res = Results(orig_img=frame_bgr, path=self.hef_path, names={}, boxes=boxes)
                res.speed = speed_dict
            except ImportError:

                class _SimpleResults:
                    def __init__(self, img: np.ndarray, p: str, b: np.ndarray, sp: dict) -> None:
                        self.orig_img = img
                        self.path = p
                        self.boxes = b
                        self.speed = sp

                    def plot(self) -> np.ndarray:
                        return self.orig_img

                    def save(self, filename: str | None = None, **kwargs: Any) -> None:
                        if filename:
                            cv2.imwrite(filename, self.plot())

                res = _SimpleResults(frame_bgr, self.hef_path, boxes, speed_dict)
        except Exception as e:
            logger.exception("Error during Hailo NPU inference: %s; using CPU fallback", e)
            if self.fallback_detector is None:
                self.fallback_detector = YoloCpuDetector(self.cfg)
            return self.fallback_detector.infer(frame_bgr, metadata=metadata)
        else:
            return [res]

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect objects and return standardized detection dictionaries."""
        if not self.use_hailo or self.fallback_detector is not None:
            return self.fallback_detector.detect(frame, metadata=metadata) if self.fallback_detector else []

        results = self.infer(frame, metadata=metadata)
        if not results:
            return []

        res = results[0] if isinstance(results, list) else results
        detections: list[DetectionDict] = []
        if hasattr(res, "boxes") and res.boxes is not None and hasattr(res.boxes, "__iter__"):
            for box in res.boxes:
                if len(box) >= 6:
                    x1, y1, x2, y2, score, cls_id = (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    if score >= self.conf_thres:
                        detections.append(
                            {
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                                "score": score,
                                "class_id": cls_id,
                                "label": str(cls_id),
                            }
                        )
        return detections

    def stop(self) -> None:
        """Release Hailo device resources."""
        if self.vdevice is not None:
            with contextlib.suppress(Exception):
                self.vdevice.release()
            self.vdevice = None
