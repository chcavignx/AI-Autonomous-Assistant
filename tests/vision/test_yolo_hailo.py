"""Unit tests for YoloHailoDetector and HailoFaceDetector with CPU fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.vision.face_detector_hailo import HailoFaceDetector
from src.vision.yolo_hailo import YoloHailoDetector

pytestmark = pytest.mark.basic


@patch("src.vision.yolo_cpu.YoloCpuDetector._load_model")
def test_yolo_hailo_detector_fallback_when_hef_missing(mock_load_model: MagicMock) -> None:
    """Test YoloHailoDetector falls back to YoloCpuDetector when HEF file does not exist."""
    mock_load_model.return_value = MagicMock()
    detector = YoloHailoDetector(hef_path="nonexistent_model.hef")
    assert detector.use_hailo is False
    assert detector.fallback_detector is not None

    # Verify detect runs via CPU fallback without crashing
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert isinstance(dets, list)


@patch("cv2.CascadeClassifier")
def test_hailo_face_detector_fallback_when_hef_missing(mock_cascade: MagicMock) -> None:
    """Test HailoFaceDetector falls back to CascadeFaceDetector when HEF file does not exist."""
    mock_cascade.return_value = MagicMock(empty=lambda: False, detectMultiScale=lambda *a, **k: [])
    detector = HailoFaceDetector(model_path="nonexistent_face_model.hef")
    assert detector.use_hailo is False
    assert detector.fallback_detector is not None

    # Verify detect runs via Cascade fallback without crashing
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_yolo_hailo_detector_mocked_hailort() -> None:
    """Test YoloHailoDetector initialization when hailo_platform is mocked."""
    mock_platform = MagicMock()
    mock_vdevice = MagicMock()
    mock_infer_model = MagicMock()
    mock_configured_model = MagicMock()

    mock_platform.VDevice.return_value = mock_vdevice
    mock_vdevice.create_infer_model.return_value = mock_infer_model
    mock_infer_model.configure.return_value = mock_configured_model

    with patch.dict("sys.modules", {"hailo_platform": mock_platform}), patch("pathlib.Path.is_file", return_value=True):
        detector = YoloHailoDetector(hef_path="dummy_yolo.hef")
        assert detector.use_hailo is True
        assert detector.fallback_detector is None


@patch("src.vision.yolo_cpu.YoloCpuDetector._load_model")
def test_yolo_hailo_detector_model_property_fallback(mock_load_model: MagicMock) -> None:
    """Test model property delegates to fallback detector when not running on Hailo."""
    mock_load_model.return_value = MagicMock()
    detector = YoloHailoDetector(hef_path="nonexistent_model.hef")
    assert detector.model is not None


@patch("src.vision.yolo_cpu.YoloCpuDetector._load_model")
def test_yolo_hailo_detector_stop(mock_load_model: MagicMock) -> None:
    """Test stop() method executes cleanly."""
    mock_load_model.return_value = MagicMock()
    detector = YoloHailoDetector(hef_path="nonexistent_model.hef")
    detector.stop()


@patch("cv2.CascadeClassifier")
def test_hailo_face_detector_stop(mock_cascade: MagicMock) -> None:
    """Test stop() method on HailoFaceDetector executes cleanly."""
    mock_cascade.return_value = MagicMock(empty=lambda: False, detectMultiScale=lambda *a, **k: [])
    detector = HailoFaceDetector(model_path="nonexistent_face_model.hef")
    detector.stop()
