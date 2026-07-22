"""Unit tests for FaceInFrame pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.basic


def _make_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# cascade backend
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.pathlib.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_cascade_backend_no_faces(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Cascade detector with no detections returns empty list."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_clf_cls.return_value = mock_clf
    mock_fa.return_value = MagicMock()

    from src.vision.face_in_frame import FaceInFrame

    proc = FaceInFrame(detector_type="cascade")
    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert faces == []


@patch("src.vision.face_detector.pathlib.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_cascade_backend_with_faces(
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

    from src.vision.face_in_frame import FaceInFrame

    proc = FaceInFrame(detector_type="cascade")
    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert len(faces) == 1
    assert "box" in faces[0]
    assert "name" in faces[0]


# ---------------------------------------------------------------------------
# insightface backend (native RetinaFace)
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.pathlib.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_insightface_backend(mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock) -> None:
    """Insightface backend directly queries face_recognizer.recognize."""
    mock_app = MagicMock()
    mock_fa.return_value = mock_app

    from src.vision.face_detector import DetectedFace
    from src.vision.face_insight_pipeline import FaceInsightPipeline

    face_mock = DetectedFace(
        bbox=np.array([10, 10, 60, 60], dtype=np.float32), landmark5=None, score=0.99, identity=None, similarity=None
    )

    from src.vision.face_in_frame import FaceInFrame

    proc = FaceInFrame(detector_type="insightface")
    proc.face_recognizer = MagicMock(spec=FaceInsightPipeline)
    proc.face_recognizer.recognize.return_value = [face_mock]

    _, faces = proc.process_frame(_make_frame(), draw=False)
    assert len(faces) == 1
    assert faces[0]["name"] == "unknown"


# ---------------------------------------------------------------------------
# unknown backend falls back to cascade
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.pathlib.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
@patch("cv2.CascadeClassifier")
def test_unknown_backend_fallback(
    mock_clf_cls: MagicMock, mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock
) -> None:
    """Unknown detector_type falls back to cascade."""
    mock_clf = MagicMock()
    mock_clf.empty.return_value = False
    mock_clf.detectMultiScale.return_value = []
    mock_clf_cls.return_value = mock_clf
    mock_fa.return_value = MagicMock()

    from src.vision.face_in_frame import FaceInFrame

    proc = FaceInFrame(detector_type="nonexistent")
    assert proc.detector_type == "cascade"


# ---------------------------------------------------------------------------
# hailo backend — lazy-loaded, mocked
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Hailo backend not supported/implemented in CPU context")
@patch("src.vision.face_detector.FaceAnalysis")
def test_hailo_backend_lazy_load(mock_fa: MagicMock) -> None:
    """Hailo backend lazy-imports YoloHailoDetector without crashing."""
    mock_fa.return_value = MagicMock()
    mock_hailo_detector = MagicMock()
    mock_hailo_detector.detect.return_value = []

    # Inject YoloHailoDetector into the module before import
    import src.vision.yolo_hailo as yh_module

    original = getattr(yh_module, "YoloHailoDetector", None)
    yh_module.YoloHailoDetector = MagicMock(return_value=mock_hailo_detector)

    try:
        import importlib

        import src.vision.face_in_frame as fim

        importlib.reload(fim)
        proc = fim.FaceInFrame(detector_type="hailo")
        assert proc.detector_type == "hailo"
        _, faces = proc.process_frame(_make_frame(), draw=False)
        assert isinstance(faces, list)
    finally:
        if original is not None:
            yh_module.YoloHailoDetector = original


# ---------------------------------------------------------------------------
# imx500 backend
# ---------------------------------------------------------------------------


@patch("src.vision.face_detector.pathlib.Path.exists", return_value=True)
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_imx500_backend_initialization(mock_get_model: MagicMock, mock_fa: MagicMock, mock_exists: MagicMock) -> None:
    """imx500 backend initializes correctly and sets detector to None."""
    mock_fa.return_value = MagicMock()

    from src.vision.face_in_frame import FaceInFrame

    proc = FaceInFrame(detector_type="imx500")
    assert proc.detector_type == "imx500"
    assert proc.detector is None
