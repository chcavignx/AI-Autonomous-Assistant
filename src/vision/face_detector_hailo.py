"""Hailo-8 / Hailo-8L NPU face detector module."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config import Config, load_config
from src.vision.base import BaseDetector, DetectedFace, DetectionDict
from src.vision.face_detector import CascadeFaceDetector

logger = logging.getLogger(__name__.split(".")[1])


class HailoFaceDetector(BaseDetector):
    """Face detector accelerated by Hailo-8 / Hailo-8L NPU.

    Uses SCRFD or YOLO-face HEF models.
    Falls back to CascadeFaceDetector if Hailo hardware or hailort is unavailable.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        model_path: str | Path | None = None,
        conf_thres: float | None = None,
    ) -> None:
        """Initialize the Hailo face detector."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.conf_thres = conf_thres if conf_thres is not None else self.cfg.vision.face_recognition_threshold

        resolved_model = model_path or self.cfg.vision.face_detector_model_path
        model_file = str(resolved_model or "scrfd_2.5g_h8l.hef")
        self.model_path = model_file
        self.use_hailo = False
        self.vdevice: Any = None
        self.infer_model: Any = None
        self.configured_infer_model: Any = None
        self.fallback_detector: CascadeFaceDetector | None = None

        self._init_hailo(model_file)

    def _init_hailo(self, model_path: str) -> None:
        """Initialize HailoRT runtime for face detection."""
        path_obj = Path(model_path)
        if not path_obj.is_file():
            # Check fallback model paths
            candidates = [
                Path("/usr/share/hailo-models") / path_obj.name,
                Path("/usr/share/hailo-models") / f"{path_obj.stem}_h8l.hef",
            ]
            for cand in candidates:
                if cand.is_file():
                    path_obj = cand
                    model_path = str(cand)
                    self.model_path = model_path
                    break

        if not Path(self.model_path).is_file():
            logger.info("Face HEF path '%s' not found, using CascadeFaceDetector fallback", model_path)
            self.fallback_detector = CascadeFaceDetector(self.cfg)
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

            self.infer_model = self.vdevice.create_infer_model(model_path)
            if FormatType is not None:
                with contextlib.suppress(Exception):
                    for out_meta in self.infer_model.outputs:
                        out_meta.set_format_type(FormatType.FLOAT32)
            self.configured_infer_model = self.infer_model.configure()
            self.use_hailo = True
            logger.info("Successfully initialized Hailo NPU face detector with model: %s", model_path)
        except Exception as err:
            logger.warning("HailoRT face detector init failed (%s); falling back to CascadeFaceDetector", err)
            self.use_hailo = False
            self.fallback_detector = CascadeFaceDetector(self.cfg)

    def infer(self, frame_bgr: np.ndarray, metadata: dict | None = None) -> Any:
        """Run face detection inference on frame BGR."""
        if not self.use_hailo or self.fallback_detector is not None:
            if self.fallback_detector is None:
                return []
            if hasattr(self.fallback_detector, "detect_faces_raw"):
                return self.fallback_detector.detect_faces_raw(frame_bgr)
            return self.fallback_detector.detect(frame_bgr, metadata=metadata)

        try:
            # Resize frame for SCRFD NPU input (640x640)
            res_w, res_h = self.cfg.vision.get_model_resolution(self.model_path)
            resized = cv2.resize(frame_bgr, (res_w, res_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)

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
        except Exception as e:
            logger.exception("Error during Hailo face detection: %s; using CPU fallback", e)
            if self.fallback_detector is None:
                self.fallback_detector = CascadeFaceDetector(self.cfg)
            return self.fallback_detector.detect(frame_bgr, metadata=metadata)
        else:
            return outputs

    def detect(self, frame: np.ndarray, metadata: dict | None = None) -> list[DetectionDict]:
        """Detect faces and return standardized detection dictionaries."""
        if not self.use_hailo or self.fallback_detector is not None:
            return self.fallback_detector.detect(frame, metadata=metadata) if self.fallback_detector else []

        outputs = self.infer(frame, metadata=metadata)
        detections: list[DetectionDict] = []
        if isinstance(outputs, np.ndarray) and outputs.size > 0:
            for box in outputs:
                if len(box) >= 5:
                    x1, y1, x2, y2, score = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
                    if score >= self.conf_thres:
                        detections.append(
                            {
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                                "score": score,
                                "class_id": 0,
                                "label": "face",
                            }
                        )
        return detections

    def detect_faces_raw(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces and return list of DetectedFace dataclass instances."""
        if not self.use_hailo or self.fallback_detector is not None:
            if self.fallback_detector is None:
                return []
            if hasattr(self.fallback_detector, "detect_faces_raw"):
                return self.fallback_detector.detect_faces_raw(frame)
            detections = self.fallback_detector.detect(frame)
            return [
                DetectedFace(
                    bbox=(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3])),
                    landmark5=None,
                    score=float(d.get("score", 1.0)),
                )
                for d in detections
            ]

        outputs = self.infer(frame)
        faces: list[DetectedFace] = []
        if isinstance(outputs, np.ndarray) and outputs.size > 0:
            for box in outputs:
                if len(box) >= 5:
                    x1, y1, x2, y2, score = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
                    if score >= self.conf_thres:
                        faces.append(
                            DetectedFace(
                                bbox=(x1, y1, x2, y2),
                                landmark5=None,
                                score=score,
                            )
                        )
        return faces

    def stop(self) -> None:
        """Release Hailo device resources."""
        if self.vdevice is not None:
            with contextlib.suppress(Exception):
                self.vdevice.release()
            self.vdevice = None
