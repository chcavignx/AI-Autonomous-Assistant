"""Unit tests for CascadeFaceDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.basic


@patch("cv2.CascadeClassifier")
def test_haar_detect_returns_detections(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns correct DetectionDict list when faces are found."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    # detectMultiScale returns list of (x, y, w, h) tuples
    mock_clf.detectMultiScale.return_value = [(10, 20, 50, 60)]
    mock_classifier_cls.return_value = mock_clf

    from src.vision.face_detector_cascade import CascadeFaceDetector

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)

    assert len(dets) == 1
    assert dets[0]["box"] == [10, 20, 60, 80]  # x,y → x+w, y+h
    assert dets[0]["score"] == 1.0
    assert dets[0]["class_id"] == 0
    assert dets[0]["label"] == "face"


@patch("cv2.CascadeClassifier")
def test_haar_detect_empty_returns_empty(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns empty list when no faces are found."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_classifier_cls.return_value = mock_clf

    from src.vision.face_detector_cascade import CascadeFaceDetector

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)
    assert dets == []


@patch("cv2.CascadeClassifier")
def test_haar_detect_empty_classifier(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns empty list when classifier failed to load."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = True
    mock_classifier_cls.return_value = mock_clf

    from src.vision.face_detector_cascade import CascadeFaceDetector

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)
    assert dets == []
