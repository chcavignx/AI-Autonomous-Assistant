"""Unit tests for InsightFaceDetector and CascadeFaceDetector in src/vision/face_detector.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils.config import Config
from src.vision.face_detector import CascadeFaceDetector, InsightFaceDetector

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.basic


@patch("src.vision.face_detector.Path.exists", return_value=False)
def test_insightface_detector_model_not_found(mock_exists: MagicMock) -> None:
    """Test InsightFaceDetector raises FileNotFoundError when model path is missing."""
    cfg = Config()
    with pytest.raises(FileNotFoundError):
        InsightFaceDetector(cfg)


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
def test_insightface_detector_detect_faces_raw(mock_fa_cls: MagicMock, mock_exists: MagicMock) -> None:
    """Test detect_faces_raw returns list of DetectedFace objects."""
    mock_app = MagicMock()
    mock_fa_cls.return_value = mock_app

    mock_face = MagicMock()
    mock_face.bbox = np.array([10.0, 20.0, 100.0, 120.0], dtype=np.float32)
    mock_face.det_score = 0.95
    mock_face.landmark = np.array([[15.0, 25.0]], dtype=np.float32)
    mock_face.normed_embedding = np.ones((512,), dtype=np.float32) / np.sqrt(512)
    mock_app.get.return_value = [mock_face]

    cfg = Config()
    detector = InsightFaceDetector(cfg)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    faces = detector.detect_faces_raw(frame)
    assert len(faces) == 1
    assert faces[0].score == 0.95
    assert faces[0].bbox == (10.0, 20.0, 100.0, 120.0)
    assert faces[0].embedding is not None


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
def test_insightface_detector_detect_standardized(mock_fa_cls: MagicMock, mock_exists: MagicMock) -> None:
    """Test detect() returns standardized DetectionDict list."""
    mock_app = MagicMock()
    mock_fa_cls.return_value = mock_app

    mock_face = MagicMock()
    mock_face.bbox = np.array([10.0, 20.0, 100.0, 120.0], dtype=np.float32)
    mock_face.det_score = 0.95
    mock_app.get.return_value = [mock_face]

    cfg = Config()
    detector = InsightFaceDetector(cfg)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)
    assert len(dets) == 1
    assert dets[0]["box"] == [10, 20, 100, 120]
    assert dets[0]["score"] == 0.95
    assert dets[0]["class_id"] == 0
    assert dets[0]["label"] == "face"


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_returns_detections(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns correct DetectionDict list when faces are found."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    # detectMultiScale returns list of (x, y, w, h) tuples
    mock_clf.detectMultiScale.return_value = [(10, 20, 50, 60)]
    mock_classifier_cls.return_value = mock_clf

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)

    assert len(dets) == 1
    assert dets[0]["box"] == [10, 20, 60, 80]  # x, y -> x+w, y+h
    assert dets[0]["score"] == 1.0
    assert dets[0]["class_id"] == 0
    assert dets[0]["label"] == "face"


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_empty_returns_empty(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns empty list when no faces are found."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_classifier_cls.return_value = mock_clf

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)
    assert dets == []


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_empty_classifier(mock_classifier_cls: MagicMock) -> None:
    """Test detect() returns empty list when classifier failed to load."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = True
    mock_classifier_cls.return_value = mock_clf

    detector = CascadeFaceDetector()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    dets = detector.detect(frame)
    assert dets == []


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_with_config(mock_classifier_cls: MagicMock) -> None:
    """Test initializing CascadeFaceDetector with a custom Config."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_classifier_cls.return_value = mock_clf

    cfg = Config()
    detector = CascadeFaceDetector(cfg)
    assert detector is not None


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_nonexistent_custom_path_fallback(mock_classifier_cls: MagicMock) -> None:
    """Test that a non-existent custom XML path safely triggers fallback to cv2 default without crashing."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_classifier_cls.return_value = mock_clf

    detector = CascadeFaceDetector(cascade_path="/nonexistent/custom_face.xml")
    assert detector is not None
    assert "custom_face.xml" not in detector.cascade_path


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_existing_custom_path(mock_classifier_cls: MagicMock, tmp_path: Path) -> None:
    """Test initializing CascadeFaceDetector with an existing custom XML path."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_classifier_cls.return_value = mock_clf

    custom_xml = tmp_path / "custom_face.xml"
    custom_xml.write_text("<opencv_storage></opencv_storage>")

    detector = CascadeFaceDetector(cascade_path=str(custom_xml))
    assert detector.cascade_path == str(custom_xml)


def test_cascade_face_detector_stop() -> None:
    """Test stop() method succeeds without error."""
    detector = CascadeFaceDetector()
    detector.stop()


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_non_xml_path_fallback(mock_classifier_cls: MagicMock) -> None:
    """Test that passing a non-XML model path triggers fallback and logs warning."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_classifier_cls.return_value = mock_clf

    detector = CascadeFaceDetector(cascade_path="/some/path/model.hef")
    assert "model.hef" not in detector.cascade_path


@patch("cv2.CascadeClassifier")
def test_cascade_face_detector_classifier_exception(mock_classifier_cls: MagicMock) -> None:
    """Test exception during CascadeClassifier instantiation falls back gracefully."""
    mock_classifier_cls.side_effect = [RuntimeError("OpenCV error"), MagicMock()]

    detector = CascadeFaceDetector()
    assert detector is not None


def test_cascade_face_detector_resolve_default_path() -> None:
    """Test _resolve_default_cascade_path returns a valid path string."""
    default_path = CascadeFaceDetector._resolve_default_cascade_path()
    assert isinstance(default_path, str)
    assert len(default_path) > 0
