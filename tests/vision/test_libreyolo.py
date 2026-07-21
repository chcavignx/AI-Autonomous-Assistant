"""Unit tests for YoloCpuDetector with LibreYOLO model support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.vision.yolo_cpu import YoloCpuDetector

pytestmark = pytest.mark.basic


def test_yolo_cpu_detector_libreyolo_load() -> None:
    """Test that YoloCpuDetector loads LibreYOLO when model_path contains 'libreyolo'."""
    mock_libreyolo_cls = MagicMock()
    mock_model = MagicMock()
    mock_libreyolo_cls.return_value = mock_model
    mock_model.names = {0: "person", 1: "bicycle"}

    # Mock the results returned by LibreYOLO
    mock_box = MagicMock()
    mock_box.xyxy = [np.array([10, 20, 100, 200])]
    mock_box.conf = [0.95]
    mock_box.cls = [0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.predict.return_value = [mock_result]

    # Patch sys.modules to mock import of libreyolo
    with patch.dict("sys.modules", {"libreyolo": MagicMock(LibreYOLO=mock_libreyolo_cls)}):
        detector = YoloCpuDetector(model_path="data/models/yolo/LibreYOLOXn.onnx")

        # Verify custom loader initialized LibreYOLO
        assert detector.model_path == "data/models/yolo/LibreYOLOXn.onnx"
        mock_libreyolo_cls.assert_called_once_with("data/models/yolo/LibreYOLOXn.onnx")

        # Run detect
        dets = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0]["box"] == [10, 20, 100, 200]
        assert dets[0]["score"] == 0.95
        assert dets[0]["class_id"] == 0
        assert dets[0]["label"] == "person"
