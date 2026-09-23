"""Unit tests for FaceInsightFrame pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.vision.face_insight_frame import FaceInsightFrame

pytestmark = pytest.mark.basic


def _make_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Cascade Backend Tests
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_face_insight_frame_cascade_backend_no_faces(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Cascade detector with no detections returns empty list."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_clf_cls.return_value = mock_clf
    mock_fa.return_value = MagicMock()

    proc = FaceInsightFrame(detector_type="cascade")
    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert faces == []


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_face_insight_frame_cascade_backend_with_faces(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Cascade detector returns face boxes when cascade fires."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = [(5, 5, 30, 30)]
    mock_clf_cls.return_value = mock_clf

    # Face recognizer: app.get returns a face
    mock_app = MagicMock()
    mock_fa.return_value = mock_app
    face_mock = MagicMock()
    face_mock.bbox = np.array([5, 5, 35, 35], dtype=np.float32)
    face_mock.embedding = np.ones(512)
    mock_app.get.return_value = [face_mock]

    proc = FaceInsightFrame(detector_type="cascade")
    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert len(faces) == 1
    assert "box" in faces[0]
    assert "name" in faces[0]


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_face_insight_frame_draw_boxes(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Test process_frame with draw=True returns an annotated frame."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = [(5, 5, 30, 30)]
    mock_clf_cls.return_value = mock_clf

    proc = FaceInsightFrame(detector_type="cascade")
    frame = _make_frame()
    annotated, faces = proc.process_frame(frame, draw=True)
    assert len(faces) == 1
    assert not np.array_equal(annotated, frame)


# ---------------------------------------------------------------------------
# InsightFace Backend Tests
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_face_insight_frame_insightface_backend(
    mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Insightface backend directly queries face_recognizer.recognize."""
    mock_app = MagicMock()
    mock_fa.return_value = mock_app

    from src.vision.face_detector import DetectedFace
    from src.vision.face_insight_pipeline import FaceInsightPipeline

    face_mock = DetectedFace(
        bbox=np.array([10, 10, 60, 60], dtype=np.float32), landmark5=None, score=0.99, identity=None, similarity=None
    )

    proc = FaceInsightFrame(detector_type="insightface")
    proc.face_recognizer = MagicMock(spec=FaceInsightPipeline)
    proc.face_recognizer.recognize.return_value = [face_mock]

    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert len(faces) == 1
    assert faces[0]["name"] == "unknown"


# ---------------------------------------------------------------------------
# Unknown Backend Fallback Tests
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_face_insight_frame_unknown_backend_fallback(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Unknown detector_type falls back to cascade."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_clf_cls.return_value = mock_clf
    mock_fa.return_value = MagicMock()

    proc = FaceInsightFrame(detector_type="nonexistent")
    assert proc.detector_type == "cascade"


# ---------------------------------------------------------------------------
# Hailo Backend Tests
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector_hailo.HailoFaceDetector")
@patch("src.vision.face_recognizer.get_model")
@patch("src.vision.face_detector.FaceAnalysis")
def test_face_insight_frame_hailo_backend_lazy_load(
    mock_fa: MagicMock, mock_get_model: MagicMock, mock_hailo_cls: MagicMock
) -> None:
    """Hailo backend lazy-imports HailoFaceDetector without crashing."""
    mock_fa.return_value = MagicMock()
    mock_hailo_detector = MagicMock()
    mock_hailo_detector.detect.return_value = []
    mock_hailo_cls.return_value = mock_hailo_detector

    proc = FaceInsightFrame(detector_type="hailo")

    assert proc.detector_type == "hailo"
    assert proc.detector is not None
    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert isinstance(faces, list)


# ---------------------------------------------------------------------------
# IMX500 Backend Tests
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_face_insight_frame_imx500_backend_initialization(
    mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """imx500 backend initializes correctly and sets detector to None."""
    mock_fa.return_value = MagicMock()

    proc = FaceInsightFrame(detector_type="imx500")
    assert proc.detector_type == "imx500"
    assert proc.detector is None


@patch("src.vision.face_detector.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_face_insight_frame_process_frame_metadata(
    mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Test process_frame_metadata forwards metadata to process_frame."""
    mock_fa.return_value = MagicMock()

    proc = FaceInsightFrame(detector_type="cascade")
    proc.process_frame = MagicMock(return_value=(_make_frame(), []))
    frame = _make_frame()
    meta = {"format": "RGB888"}
    proc.process_frame_metadata(frame, metadata=meta, thresh=0.5, draw=False)
    proc.process_frame.assert_called_once_with(frame, thresh=0.5, draw=False, metadata=meta)


def test_face_insight_frame_alias() -> None:
    """Test backward compatibility alias FaceInsightFrame is FaceInsightFrame."""
    assert FaceInsightFrame is FaceInsightFrame


@patch("src.vision.face_detector_hailo.HailoFaceDetector")
@patch("src.vision.face_recognizer.get_model")
@patch("src.vision.face_detector.FaceAnalysis")
def test_face_insight_frame_hailo_backend_with_identity(
    mock_fa: MagicMock, mock_get_model: MagicMock, mock_hailo_cls: MagicMock
) -> None:
    """Test that FaceInsightFrame with hailo backend identifies recognized faces."""
    mock_fa.return_value = MagicMock()
    mock_hailo_cls.return_value = MagicMock()

    from src.vision.face_detector import DetectedFace
    from src.vision.face_insight_pipeline import FaceInsightPipeline

    face_mock = DetectedFace(
        bbox=(10.0, 10.0, 60.0, 60.0),
        landmark5=None,
        score=0.98,
        identity="Alice",
        similarity=0.85,
    )

    proc = FaceInsightFrame(detector_type="hailo", enable_recognition=True)
    proc.face_recognizer = MagicMock(spec=FaceInsightPipeline)
    proc.face_recognizer.recognize.return_value = [face_mock]

    frame = _make_frame()
    annotated, faces = proc.process_frame(frame, draw=True)

    assert len(faces) == 1
    assert faces[0]["name"] == "Alice"
    assert faces[0]["similarity"] == 0.85
    assert not np.array_equal(annotated, frame)


@patch("src.vision.face_recognizer.get_model")
@patch("src.vision.face_detector.FaceAnalysis")
@patch("cv2.CascadeClassifier")
def test_face_insight_frame_disabled_recognition_mode(
    mock_clf_cls: MagicMock, mock_fa: MagicMock, mock_get_model: MagicMock
) -> None:
    """Test that enable_recognition=False falls back to detection-only mode."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = [(10, 10, 50, 50)]
    mock_clf_cls.return_value = mock_clf

    proc = FaceInsightFrame(detector_type="cascade", enable_recognition=False)
    _, faces = proc.process_frame(_make_frame(), draw=False)

    assert len(faces) == 1
    assert faces[0]["name"] == "unknown"


@patch("src.vision.face_recognizer.get_model")
@patch("src.vision.face_detector.FaceAnalysis")
def test_face_insight_frame_stop_lifecycle(mock_fa: MagicMock, mock_get_model: MagicMock) -> None:
    """Test that stop() cleans up both detector and face_recognizer."""
    mock_fa.return_value = MagicMock()

    proc = FaceInsightFrame(detector_type="cascade")
    proc.detector = MagicMock()
    proc.face_recognizer = MagicMock()

    proc.stop()

    proc.detector.stop.assert_called_once()
    proc.face_recognizer.stop.assert_called_once()
