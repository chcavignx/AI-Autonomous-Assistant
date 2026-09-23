"""Unit tests for WebFaceCapture example utility with FaceInsightFrame refactoring."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from examples.vision.face_capture_web import WebFaceCapture

# Mock picamera2 for testing environment
sys.modules["picamera2"] = MagicMock()
sys.modules["picamera2"].Picamera2 = MagicMock()

pytestmark = pytest.mark.basic


@pytest.fixture
def web_face_capture():
    """Create a WebFaceCapture instance with mocked camera and face processor."""
    with patch("examples.vision.face_capture_web.discover_pi_cameras", return_value=[]):
        tool = WebFaceCapture()
        tool.face_processor = MagicMock()
        return tool


@patch("os.path.isdir", return_value=False)
def test_process_directory_dataset_dir_not_exist(mock_isdir: MagicMock, web_face_capture: WebFaceCapture) -> None:
    res = web_face_capture.process_directory_dataset("/nonexistent_dir", "alice")
    assert not res["success"]
    assert "does not exist" in res["message"]


@patch("os.path.isdir", return_value=True)
def test_process_directory_dataset_empty_name(mock_isdir: MagicMock, web_face_capture: WebFaceCapture) -> None:
    res = web_face_capture.process_directory_dataset("/some_dir", "")
    assert not res["success"]
    assert "cannot be empty" in res["message"]


@patch("os.path.isdir", return_value=True)
@patch("os.walk", return_value=[("/some_dir", [], ["file1.txt", "file2.pdf"])])
def test_process_directory_dataset_no_images(
    mock_walk: MagicMock, mock_isdir: MagicMock, web_face_capture: WebFaceCapture
) -> None:
    res = web_face_capture.process_directory_dataset("/some_dir", "alice")
    assert not res["success"]
    assert "No supported images found" in res["message"]


@patch("os.path.isdir", return_value=True)
@patch("os.walk", return_value=[("/some_dir", [], ["pic1.jpg", "pic2.png"])])
@patch("cv2.imread")
@patch("cv2.imwrite")
@patch("os.makedirs")
def test_process_directory_dataset_success(
    mock_makedirs: MagicMock,
    mock_imwrite: MagicMock,
    mock_imread: MagicMock,
    mock_walk: MagicMock,
    mock_isdir: MagicMock,
    web_face_capture: WebFaceCapture,
) -> None:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    # Mock face processor returning one face
    web_face_capture.face_processor.process_frame.return_value = (
        dummy_img,
        [{"box": [10, 10, 50, 50], "score": 1.0, "class_id": 0, "label": "face"}],
    )

    res = web_face_capture.process_directory_dataset("/some_dir", "alice")

    assert res["success"]
    assert res["processed_images"] == 2
    assert res["faces_extracted"] == 2
    assert mock_imwrite.call_count == 4  # 2 images * (1 full + 1 crop) = 4 writes


@patch("os.path.isdir", return_value=False)
def test_process_directory_identify_dir_not_exist(mock_isdir: MagicMock, web_face_capture: WebFaceCapture) -> None:
    res = web_face_capture.process_directory_identify("/nonexistent_dir")
    assert not res["success"]
    assert "does not exist" in res["message"]


@patch("os.path.isdir", return_value=True)
@patch("os.walk", return_value=[("/some_dir", [], ["John_Doe_1.jpg", "random_pic.png"])])
@patch("cv2.imread")
def test_scan_directory_for_catalog_success(
    mock_imread: MagicMock, mock_walk: MagicMock, mock_isdir: MagicMock, web_face_capture: WebFaceCapture
) -> None:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    # Detected face in the first image, none in the second
    web_face_capture.face_processor.process_frame.side_effect = [
        (
            dummy_img,
            [{"box": [10, 10, 50, 50], "score": 1.0, "class_id": 0, "label": "face"}],
        ),  # One face in John_Doe_1.jpg
        (dummy_img, []),  # No face in random_pic.png
    ]

    res = web_face_capture.scan_directory_for_catalog("/some_dir")
    assert res["success"]
    assert len(res["candidates"]) == 1
    assert res["candidates"][0]["filename"] == "John_Doe_1.jpg"
    assert res["candidates"][0]["suggestion"] == "John Doe"


@patch("os.path.exists", return_value=True)
@patch("cv2.imread")
@patch("cv2.imwrite")
@patch("os.makedirs")
def test_save_cataloged_face_success(
    mock_makedirs: MagicMock,
    mock_imwrite: MagicMock,
    mock_imread: MagicMock,
    mock_exists: MagicMock,
    web_face_capture: WebFaceCapture,
) -> None:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = dummy_img

    web_face_capture.face_processor.process_frame.return_value = (
        dummy_img,
        [{"box": [10, 10, 50, 50], "score": 1.0, "class_id": 0, "label": "face"}],
    )

    res = web_face_capture.save_cataloged_face("/some_dir/John_Doe_1.jpg", "John Doe")
    assert res["success"]
    assert "Successfully cataloged" in res["message"]
    assert mock_imwrite.call_count == 2  # 1 full + 1 crop = 2 writes
