"""Unit tests for ObjectInsightFrame pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

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
    """process_frame returns empty detections when detector finds nothing."""
    from src.vision.object_insight_frame import ObjectInsightFrame

    proc = ObjectInsightFrame(detector=_make_detector([]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, dets, _ = proc.process_frame(frame, draw=False)

    assert dets == []
    assert annotated.shape == frame.shape


def test_object_insight_frame_with_detection() -> None:
    """process_frame returns one detection correctly."""
    from src.vision.object_insight_frame import ObjectInsightFrame

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
    """process_frame with draw=True modifies output frame pixels."""
    from src.vision.object_insight_frame import ObjectInsightFrame

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
