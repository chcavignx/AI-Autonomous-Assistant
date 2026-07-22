"""Unit tests for the Sony IMX500 On-Sensor vision detector."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np  #
import pytest
from src.vision.yolo_imx500 import Imx500Config, Imx500Detector  #

sys.modules["picamera2"] = MagicMock()
sys.modules["picamera2.devices"] = MagicMock()
sys.modules["picamera2.devices.imx500"] = MagicMock()
sys.modules["picamera2.devices.imx500.postprocess"] = MagicMock()


def test_imx500_detector_lifecycle() -> None:
    """Test full metadata consumption."""
    mock_imx500 = MagicMock()

    # Setup mock network intrinsics for detection outputs
    mock_intrinsics = MagicMock()
    mock_imx500.network_intrinsics = mock_intrinsics
    mock_imx500.get_input_size.return_value = (640, 640)

    # Dummy tensors for detection: boxes, scores, classes
    boxes = np.array([[[288, 216, 352, 264]]], dtype=np.float32)
    scores = np.array([[0.8]], dtype=np.float32)
    classes = np.array([[0]], dtype=np.float32)
    mock_imx500.get_outputs.return_value = [boxes, scores, classes]

    detector = Imx500Detector(Imx500Config(), imx500=mock_imx500)

    # Mock metadata payload
    metadata = {"Dummy": "Payload"}

    res = detector.infer(np.zeros((480, 640, 3)), metadata=metadata)
    dets = res.detections
    assert len(dets) == 1
    assert dets[0].cls == 0
    assert dets[0].score == pytest.approx(0.8)
    assert dets[0].x1 == pytest.approx(288)
    assert dets[0].y1 == pytest.approx(162)
    assert dets[0].x2 == pytest.approx(352)
    assert dets[0].y2 == pytest.approx(198)

    detector.stop()
