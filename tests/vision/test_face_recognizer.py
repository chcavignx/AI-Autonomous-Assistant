"""Unit tests for ArcFaceRecognizer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.vision.face_recognizer import ArcFaceRecognizer

pytestmark = pytest.mark.basic


@patch("src.vision.face_recognizer.get_model")
def test_arcface_recognizer_init_with_model(mock_get_model: MagicMock) -> None:
    """Test ArcFaceRecognizer initialization with mocked get_model."""
    mock_arcface = MagicMock()
    mock_get_model.return_value = mock_arcface

    recognizer = ArcFaceRecognizer(model_name="dummy_arcface.onnx", load_model=True)
    assert recognizer.arcface is not None
    mock_arcface.prepare.assert_called_once_with(ctx_id=0)


def test_arcface_recognizer_init_without_model() -> None:
    """Test ArcFaceRecognizer initialization with load_model=False."""
    recognizer = ArcFaceRecognizer(load_model=False)
    assert recognizer.arcface is None
    assert recognizer.known_faces == {}


@patch("src.vision.face_recognizer.get_model")
def test_arcface_recognizer_extract_embedding_with_landmarks(mock_get_model: MagicMock) -> None:
    """Test extract_embedding with 5-point landmarks."""
    mock_arcface = MagicMock()
    dummy_feat = np.ones((512,), dtype=np.float32)
    mock_arcface.get_feat.return_value = dummy_feat
    mock_get_model.return_value = mock_arcface

    recognizer = ArcFaceRecognizer(load_model=True)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = np.array([[10, 10], [20, 10], [15, 20], [12, 30], [18, 30]], dtype=np.float32)

    emb = recognizer.extract_embedding(frame, landmark5=landmarks)
    assert emb.shape == (512,)
    assert np.isclose(np.linalg.norm(emb), 1.0)


@patch("src.vision.face_recognizer.get_model")
def test_arcface_recognizer_extract_embedding_with_bbox_fallback(mock_get_model: MagicMock) -> None:
    """Test extract_embedding with bounding box fallback."""
    mock_arcface = MagicMock()
    dummy_emb = np.ones((512,), dtype=np.float32)
    mock_arcface.get_feat.return_value = dummy_emb
    mock_get_model.return_value = mock_arcface

    recognizer = ArcFaceRecognizer(load_model=True)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10.0, 10.0, 50.0, 50.0)

    emb = recognizer.extract_embedding(frame, bbox=bbox)
    assert emb.shape == (512,)
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_arcface_recognizer_extract_embedding_unloaded_raises() -> None:
    """Test extract_embedding raises RuntimeError when model is not loaded."""
    recognizer = ArcFaceRecognizer(load_model=False)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="ArcFace model is not loaded"):
        recognizer.extract_embedding(frame)


def test_arcface_recognizer_face_registry_crud() -> None:
    """Test registering and matching faces."""
    recognizer = ArcFaceRecognizer(load_model=False)

    # Register faces with direct embeddings
    emb_alice = np.zeros((512,), dtype=np.float32)
    emb_alice[0] = 1.0
    emb_bob = np.zeros((512,), dtype=np.float32)
    emb_bob[1] = 1.0

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    recognizer.register_face("alice", frame, embedding=emb_alice)
    recognizer.register_face("bob", frame, embedding=emb_bob)

    assert set(recognizer.known_faces.keys()) == {"alice", "bob"}

    # Match Alice
    match_id, sim = recognizer.match_face(emb_alice, thresh=0.5)
    assert match_id == "alice"
    assert np.isclose(sim, 1.0)

    # Match Bob
    match_id, sim = recognizer.match_face(emb_bob, thresh=0.5)
    assert match_id == "bob"
    assert np.isclose(sim, 1.0)

    # Match Unknown
    emb_unknown = np.zeros((512,), dtype=np.float32)
    emb_unknown[2] = 1.0
    match_id, sim = recognizer.match_face(emb_unknown, thresh=0.5)
    assert match_id is None
    assert sim is None
