"""Unit tests for ObjectInsightFrame pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils.config import Config
from src.vision.object_insight_frame import ObjectInsightFrame

pytestmark = pytest.mark.basic


def _make_detector(dets: list) -> MagicMock:
    """Return a mock detector whose infer() returns a mock results object with the given dets."""
    mock = MagicMock()
    mock_results = MagicMock()

    mock_boxes = []
    names = {}
    for d in dets:
        box = MagicMock()
        box.xyxy = [d["box"]]
        box.conf = [d["score"]]
        box.cls = [d["class_id"]]
        mock_boxes.append(box)
        names[d["class_id"]] = d["label"]

    mock_results.boxes = tuple(mock_boxes)
    mock.infer.return_value = mock_results

    mock_model = MagicMock()
    mock_model.names = names
    mock.model = mock_model

    return mock


def test_object_insight_frame_no_detections() -> None:
    """Test process_frame returns empty detections when detector finds nothing."""
    proc = ObjectInsightFrame(detector=_make_detector([]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, dets, _ = proc.process_frame(frame, draw=False)

    assert dets == []
    assert annotated.shape == frame.shape


def test_object_insight_frame_with_detection() -> None:
    """Test process_frame returns one detection correctly."""
    detection = {
        "box": [10, 20, 50, 60],
        "score": 0.85,
        "class_id": 2,
        "label": "car",
    }
    proc = ObjectInsightFrame(detector=_make_detector([detection]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, dets, _ = proc.process_frame(frame, draw=False)

    assert len(dets) == 1
    assert dets[0]["box"] == [10, 20, 50, 60]
    assert abs(dets[0]["score"] - 0.85) < 1e-6
    assert dets[0]["class_id"] == 2
    assert dets[0]["label"] == "car"


def test_object_insight_frame_draw() -> None:
    """Test process_frame with draw=True modifies output frame pixels."""
    detection = {
        "box": [5, 5, 50, 50],
        "score": 0.9,
        "class_id": 0,
        "label": "person",
    }
    proc = ObjectInsightFrame(detector=_make_detector([detection]))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    annotated, _, _ = proc.process_frame(frame, draw=True)

    # Drawing should modify at least some pixels
    assert not np.array_equal(annotated, frame)


@patch("src.vision.yolo_hailo.YoloHailoDetector")
def test_object_insight_frame_hailo_init(mock_hailo: MagicMock) -> None:
    """Test ObjectInsightFrame initializes YoloHailoDetector when configured."""
    cfg = Config()
    cfg.vision.object_model_type = "yolo_hailo"
    proc = ObjectInsightFrame(cfg)
    assert proc.detector is not None


def test_object_insight_frame_stop() -> None:
    """Test stop() method forwards stop signal to detector."""
    mock_det = MagicMock()
    proc = ObjectInsightFrame(detector=mock_det)
    proc.stop()
    mock_det.stop.assert_called_once()
