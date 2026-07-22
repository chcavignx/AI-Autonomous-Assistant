"""Unit tests for FaceInsightPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils.config import Config
from src.vision.face_detector import DetectedFace
from src.vision.face_insight_pipeline import FaceInsightPipeline

pytestmark = pytest.mark.basic


@patch("src.vision.face_detector.pathlib.Path.exists")
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_pipeline_insightface_detect(
    mock_get_model: MagicMock, mock_face_analysis: MagicMock, mock_exists: MagicMock
) -> None:
    """Test FaceInsightPipeline.detect() with InsightFace returns correct DetectionDict list."""
    mock_exists.return_value = True
    mock_app = MagicMock()
    mock_face_analysis.return_value = mock_app

    mock_face = MagicMock()
    mock_face.bbox = np.array([10, 20, 100, 120], dtype=np.float32)
    mock_face.det_score = 0.92
    mock_app.get.return_value = [mock_face]

    cfg = Config()
    cfg.vision.face_detector_type = "insightface"
    detector = FaceInsightPipeline(cfg)
    dets = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert len(dets) == 1
    assert dets[0]["box"] == [10, 20, 100, 120]
    assert abs(dets[0]["score"] - 0.92) < 1e-6
    assert dets[0]["class_id"] == 0
    assert dets[0]["label"] == "face"


@patch("src.vision.face_detector.pathlib.Path.exists")
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_pipeline_insightface_recognize(
    mock_get_model: MagicMock, mock_face_analysis: MagicMock, mock_exists: MagicMock
) -> None:
    """Test recognize() returns unknown face when no known faces registered."""
    mock_exists.return_value = True
    mock_app = MagicMock()
    mock_face_analysis.return_value = mock_app

    mock_face = MagicMock()
    mock_face.bbox = np.array([10, 20, 100, 120], dtype=np.float32)
    mock_face.embedding = np.ones(512)
    mock_face.det_score = 0.9
    mock_face.landmark = np.zeros((5, 2))
    mock_app.get.return_value = [mock_face]

    cfg = Config()
    cfg.vision.face_detector_type = "insightface"
    detector = FaceInsightPipeline(cfg)
    results = detector.recognize(np.zeros((200, 200, 3), dtype=np.uint8))

    assert len(results) == 1
    assert isinstance(results[0], DetectedFace)
    assert results[0].identity is None
    assert results[0].similarity is None


@patch("src.vision.face_detector.pathlib.Path.exists")
@patch("src.vision.face_detector.FaceAnalysis")
@patch("src.vision.face_recognizer.get_model")
def test_pipeline_insightface_register_no_face(
    mock_get_model: MagicMock, mock_face_analysis: MagicMock, mock_exists: MagicMock
) -> None:
    """Test register_face() raises when no face is detected."""
    mock_exists.return_value = True
    mock_app = MagicMock()
    mock_face_analysis.return_value = mock_app
    mock_app.get.return_value = []

    cfg = Config()
    cfg.vision.face_detector_type = "insightface"
    detector = FaceInsightPipeline(cfg)
    with pytest.raises(RuntimeError, match="No face detected"):
        detector.register_face("user", np.zeros((100, 100, 3), dtype=np.uint8))


@patch("src.vision.face_detector.Imx500Detector")
@patch("src.vision.face_recognizer.get_model")
def test_pipeline_imx500_recognize_from_frame(mock_get_model: MagicMock, mock_detector_cls: MagicMock) -> None:
    """Test recognize() with IMX500 detector uses metadata appropriately."""
    cfg = Config()
    cfg.vision.face_detector_type = "imx500"

    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector

    mock_face = DetectedFace(bbox=(64.0, 48.0, 320.0, 240.0), landmark5=None, score=0.95)
    mock_detector.detect_faces_metadata.return_value = [mock_face]

    # The get_model mock creates an ArcFace model
    mock_arcface = MagicMock()
    # It must return a numpy array of shape (512,) to avoid ValueError broadcast mismatch with dot()
    mock_arcface.get.return_value = np.ones((512,), dtype=np.float32)
    mock_get_model.return_value = mock_arcface

    pipeline = FaceInsightPipeline(cfg, arcface_model_name="dummy.onnx")

    # Dummy frame and metadata
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_metadata = {"format": "RGB888"}

    results = pipeline.recognize(dummy_frame, metadata=dummy_metadata)

    assert len(results) == 1
    assert results[0].identity is None
    assert mock_detector.detect_faces_metadata.called
