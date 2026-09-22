"""Unit tests for Imx500Detector in src/vision/face_detector_imx.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.vision.face_detector_imx import Imx500Detector

pytestmark = pytest.mark.basic


def test_imx500_face_detector_detect_raw_empty() -> None:
    """Test detect_faces_raw on IMX500 returns empty list without metadata."""
    detector = Imx500Detector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect_faces_raw(frame) == []


def test_imx500_face_detector_metadata_parsing() -> None:
    """Test detect_faces_metadata extracts normalized face coordinates."""
    mock_imx500 = MagicMock()
    mock_imx500.get_input_size.return_value = (640, 480)

    # Boxes format: [y0, x0, y1, x1] or [x0, y0, x1, y1] normalized or unnormalized
    # Output mock: [boxes, scores, classes]
    boxes = np.array([[[100, 100, 300, 300]]], dtype=np.float32)
    scores = np.array([[0.9]], dtype=np.float32)
    classes = np.array([[0]], dtype=np.float32)
    mock_imx500.get_outputs.return_value = [boxes, scores, classes]

    detector = Imx500Detector()
    detector.imx500 = mock_imx500
    mock_intrinsics = MagicMock()
    mock_intrinsics.bbox_order = "yx"
    mock_intrinsics.bbox_normalization = False
    mock_intrinsics.postprocess = ""
    detector.intrinsics = mock_intrinsics

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces_metadata(frame, metadata={"key": "val"})

    assert len(faces) == 1
    assert faces[0].score == pytest.approx(0.9)

    dets = detector.detect(frame, metadata={"key": "val"})
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    assert dets[0]["label"] == "face"


def test_imx500_face_detector_stop() -> None:
    """Test stop() clears imx500 handle."""
    detector = Imx500Detector()
    detector.stop()
    assert detector.imx500 is None
