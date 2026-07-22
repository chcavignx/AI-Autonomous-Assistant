"""Unit tests for the Yolo26NcnnDetector class."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from src.vision.yolo_cpu import Detection, Yolo26NcnnDetector


class TestYolo26NcnnDetector(unittest.TestCase):
    """Test suite for Yolo26NcnnDetector."""

    @patch("src.vision.yolo_cpu.YOLO")
    def test_init(self, mock_yolo: MagicMock) -> None:
        """Test initialisation of the detector."""
        detector = Yolo26NcnnDetector(conf_thres=0.3)
        assert detector.conf_thres == 0.3
        mock_yolo.assert_called_once()

    @patch("src.vision.yolo_cpu.YOLO")
    def test_infer_no_results(self, mock_yolo: MagicMock) -> None:
        """Test inference returning no results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        mock_yolo.return_value = mock_model

        detector = Yolo26NcnnDetector()
        res = detector.infer(np.zeros((100, 100, 3), dtype=np.uint8))
        assert res == []

    @patch("src.vision.yolo_cpu.YOLO")
    def test_infer_with_results(self, mock_yolo: MagicMock) -> None:
        """Test inference returning boxes and detections."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock results[0].boxes
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 30, 40]], dtype=np.float64)
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9], dtype=np.float64)
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([2], dtype=np.float64)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        detector = Yolo26NcnnDetector(conf_thres=0.25)
        res = detector.infer(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(res) == 1
        assert res[0] == Detection(x1=10.0, y1=20.0, x2=30.0, y2=40.0, score=0.9, cls=2)

    @patch("src.vision.yolo_cpu.YOLO")
    def test_detect(self, mock_yolo: MagicMock) -> None:
        """Test standardized detect() method output format."""
        mock_model = MagicMock()
        mock_model.names = {2: "car"}
        mock_yolo.return_value = mock_model

        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10.1, 20.2, 30.3, 40.4]], dtype=np.float64)
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9], dtype=np.float64)
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([2], dtype=np.float64)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        detector = Yolo26NcnnDetector()
        res = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(res) == 1
        assert res[0]["box"] == [10, 20, 30, 40]
        assert res[0]["score"] == 0.9
        assert res[0]["class_id"] == 2
        assert res[0]["label"] == "car"

    def test_draw_detections(self) -> None:
        """Test drawing detections modifying input image pixels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [Detection(x1=10.0, y1=20.0, x2=30.0, y2=40.0, score=0.9, cls=2)]
        out = Yolo26NcnnDetector.draw_detections(img, dets)
        assert out.shape == img.shape
        assert not np.array_equal(out, img)  # pixels modified
