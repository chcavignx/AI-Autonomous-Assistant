"""Integration and unit tests for examples/vision scripts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from src.utils.config import Config
from src.vision.face_detector import DetectedFace

pytestmark = pytest.mark.basic


@pytest.fixture
def mock_cam():
    cam = MagicMock()
    cam.index = 0
    cam.camera_info = {"Model": "IMX500"}
    cam.get_info_str.return_value = "Mock Camera 0"
    cam.capture_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return cam


def test_face_capture_tool_init_and_detect(mock_cam, tmp_path):
    """Test FaceCaptureTool initializes and runs face detection."""
    from examples.vision.face_capture import FaceCaptureTool

    cfg = Config()
    cfg.vision.face_dataset_path = str(tmp_path)

    with (
        patch("examples.vision.face_capture.load_config", return_value=cfg),
        patch("examples.vision.face_capture.discover_pi_cameras", return_value=[mock_cam]),
        patch("src.vision.face_detector.cv2.CascadeClassifier"),
    ):
        tool = FaceCaptureTool()
        assert tool.output_dir == str(tmp_path)
        assert len(tool.cameras) == 1

        # Test detect_faces
        mock_faces = [{"box": [10, 20, 100, 120], "score": 0.9, "class_id": 0, "label": "face"}]
        with patch.object(tool.face_processor, "process_frame", return_value=(np.zeros((100, 100, 3)), mock_faces)):
            faces = tool.detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))
            assert len(faces) == 1
            assert faces[0] == (10, 20, 90, 100)

        tool.stop()


def test_face_capture_headless_detect_and_capture(mock_cam, tmp_path):
    """Test HeadlessFaceCapture detects and saves photos to output_dir."""
    from examples.vision.face_capture_headless import HeadlessFaceCapture

    cfg = Config()
    cfg.vision.face_dataset_path = str(tmp_path)

    with (
        patch("examples.vision.face_capture_headless.load_config", return_value=cfg),
        patch("examples.vision.face_capture_headless.discover_pi_cameras", return_value=[mock_cam]),
        patch("src.vision.face_detector.cv2.CascadeClassifier"),
    ):
        tool = HeadlessFaceCapture()
        tool.start_camera(0)
        tool.current_person_name = "test_user"

        mock_dets = [{"box": [10, 20, 100, 120], "score": 0.95, "class_id": 0, "label": "face"}]
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_dets
        tool.face_detector = mock_detector

        success, count = tool.detect_and_capture()
        assert success is True
        assert count == 1

        person_dir = tmp_path / "test_user"

        assert person_dir.exists()
        saved_files = list(person_dir.glob("*.jpg"))
        assert len(saved_files) >= 1

        tool.stop()


def test_face_identify_tool_dataset_and_draw(mock_cam, tmp_path):
    """Test FaceIdentifyTool dataset loading, interface drawing, and lifecycle."""
    from examples.vision.face_identify import FaceIdentifyTool

    # Create dummy dataset directory structure
    person_dir = tmp_path / "alice"
    person_dir.mkdir(parents=True)
    dummy_img_path = person_dir / "001_full.jpg"
    cv2.imwrite(str(dummy_img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    cfg = Config()
    cfg.vision.face_dataset_path = str(tmp_path)

    with (
        patch("examples.vision.face_identify.load_config", return_value=cfg),
        patch("examples.vision.face_identify.discover_pi_cameras", return_value=[mock_cam]),
        patch("src.vision.face_insight_pipeline.FaceInsightPipeline.register_face") as mock_reg,
        patch("src.vision.face_detector.cv2.CascadeClassifier"),
    ):
        tool = FaceIdentifyTool()
        assert tool.dataset_dir == str(tmp_path)
        assert mock_reg.called

        # Test draw_interface with DetectedFace
        face = DetectedFace(
            bbox=np.array([10, 20, 100, 120]),
            identity="alice_0",
            similarity=0.88,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tool.draw_interface(frame, [face])

        tool.stop()


def test_face_identify_web_load_dataset_and_stop(mock_cam, tmp_path):
    """Test WebFaceIdentify dataset loading and cleanup."""
    from examples.vision.face_identify_web import WebFaceIdentify

    cfg = Config()
    cfg.vision.face_dataset_path = str(tmp_path)

    with (
        patch("examples.vision.face_identify_web.config", cfg),
        patch("examples.vision.face_identify_web.discover_pi_cameras", return_value=[mock_cam]),
        patch("src.vision.face_insight_pipeline.FaceInsightPipeline"),
    ):
        tool = WebFaceIdentify()
        assert tool.dataset_dir == str(tmp_path)
        tool.stop()
        assert tool.running is False


def test_object_identify_web_threshold_and_stop(mock_cam):
    """Test WebObjectIdentify threshold setting and stop lifecycle."""
    from examples.vision.object_identify_web import WebObjectIdentify

    cfg = Config()

    with (
        patch("examples.vision.object_identify_web.config", cfg),
        patch("examples.vision.object_identify_web.discover_pi_cameras", return_value=[mock_cam]),
        patch("examples.vision.object_identify_web.VideoCapture") as mock_vc_cls,
    ):
        mock_vc = MagicMock()
        mock_vc_cls.return_value = mock_vc

        tool = WebObjectIdentify()
        tool.set_threshold(0.75)
        assert abs(tool.conf_threshold - 0.75) < 1e-4

        tool.stop()
        assert tool.running is False
        mock_vc.stop.assert_called()


def test_simultaneous_vision_e2e_automated():
    """Test simultaneous vision CLI end-to-end automated test suite."""
    from examples.vision.simultaneous_face_object_e2e import run_automated_test

    assert run_automated_test() is True


def test_simultaneous_vision_web_automated():
    """Test simultaneous vision Web Flask endpoints and streaming automated test."""
    from examples.vision.simultaneous_vision_web import run_automated_web_test

    assert run_automated_web_test() is True
