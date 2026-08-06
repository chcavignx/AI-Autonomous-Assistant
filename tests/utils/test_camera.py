"""Unit tests for camera utilities in src/utils/camera.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils import camera
from src.utils.config import Config

pytestmark = pytest.mark.basic


def test_threaded_camera_init_and_read() -> None:
    """Test ThreadedCamera lifecycle and frame reading."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.ones((10, 10, 3), dtype=np.uint8))

    cfg = Config()
    cfg.vision.camera.camera_index = 0
    cfg.vision.camera.frame_width = 10
    cfg.vision.camera.frame_height = 10

    with patch("cv2.VideoCapture", return_value=mock_cap):
        threaded_cam = camera.ThreadedCamera(cfg)

        # Stop background loop immediately to prevent uncontrolled thread executions
        threaded_cam.started = False
        if threaded_cam.thread:
            threaded_cam.thread.join(timeout=1.0)

        assert threaded_cam.grabbed is True
        assert threaded_cam.frame is not None

        # Check read returns copy
        frame, _metadata = threaded_cam.read()
        assert frame is not None
        assert np.array_equal(frame, threaded_cam.frame)

        # Check stop releasing camera
        threaded_cam.stop()
        mock_cap.release.assert_called_once()


def test_threaded_camera_model_resolution() -> None:
    """Test ThreadedCamera configures frame width and height from model resolution table."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.ones((10, 10, 3), dtype=np.uint8))

    cfg = Config()
    cfg.vision.object_model_name = "LibreYOLOXn.onnx"

    with patch("cv2.VideoCapture", return_value=mock_cap):
        threaded_cam = camera.ThreadedCamera(cfg)
        threaded_cam.started = False
        if threaded_cam.thread:
            threaded_cam.thread.join(timeout=1.0)

        assert threaded_cam.frame_width == 416
        assert threaded_cam.frame_height == 416


def test_threaded_camera_read_none() -> None:
    """Test ThreadedCamera read returning None when not grabbed."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    cfg = Config()

    with patch("cv2.VideoCapture", return_value=mock_cap):
        threaded_cam = camera.ThreadedCamera(cfg)
        threaded_cam.started = False
        if threaded_cam.thread:
            threaded_cam.thread.join(timeout=1.0)

        threaded_cam.grabbed = False
        assert threaded_cam.read() == (None, None)


def test_pi_camera_wrapper() -> None:
    """Test PiCamera class wrappers around Picamera2."""
    mock_picam = MagicMock()
    mock_picam.camera_info = "Mock Pi Camera 2"
    mock_picam.create_video_configuration.return_value = {"main": {"size": (640, 480)}}
    mock_picam.capture_array.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock sys.modules for picamera2 to allow importing it in PiCamera
    with patch.dict("sys.modules", {"picamera2": MagicMock(Picamera2=MagicMock(return_value=mock_picam))}):
        pi_cam = camera.PiCamera(index=0)
        assert pi_cam.index == 0
        assert pi_cam.camera_info == "Mock Pi Camera 2"
        assert pi_cam.get_info_str() == "Mock Pi Camera 2"

        pi_cam.start(320, 240, video_format="YUV420")
        mock_picam.create_video_configuration.assert_called_once_with(main={"size": (320, 240), "format": "YUV420"})
        mock_picam.configure.assert_called_once()
        mock_picam.start.assert_called_once()

        frame = pi_cam.capture_frame()
        assert frame.shape == (480, 640, 3)
        mock_picam.capture_array.assert_called_once()

        pi_cam.stop()
        mock_picam.stop.assert_called_once()


def test_discover_pi_cameras_no_picamera2() -> None:
    """Test discover_pi_cameras when picamera2 package is missing."""
    with patch.dict("sys.modules", {"picamera2": None}):
        cams = camera.discover_pi_cameras()
        assert cams == []


def test_discover_pi_cameras_success() -> None:
    """Test discover_pi_cameras successfully discovers devices."""
    mock_picam = MagicMock()
    mock_picam.camera_info = "Mock Pi Camera"
    with (
        patch.dict("sys.modules", {"picamera2": MagicMock(Picamera2=MagicMock(return_value=mock_picam))}),
        patch("importlib.util.find_spec", return_value=True),
    ):
        cams = camera.discover_pi_cameras()
        assert len(cams) == 2
        assert all(isinstance(c, camera.PiCamera) for c in cams)


def test_is_imx500_camera_true() -> None:
    """Test is_imx500_camera when model name contains imx500."""
    mock_picam = MagicMock()
    mock_picam.camera_properties = {"Model": "IMX500_Sensor"}

    # Mock context manager: enter returns mock_picam, exit does nothing
    mock_picam_cls = MagicMock()
    mock_picam_cls.return_value = mock_picam
    mock_picam.__enter__.return_value = mock_picam

    with patch.dict("sys.modules", {"picamera2": MagicMock(Picamera2=mock_picam_cls)}):
        assert camera.is_imx500_camera() is True


def test_is_imx500_camera_false() -> None:
    """Test is_imx500_camera when model name does not contain imx500."""
    mock_picam = MagicMock()
    mock_picam.camera_properties = {"Model": "IMX219"}

    mock_picam_cls = MagicMock()
    mock_picam_cls.return_value = mock_picam
    mock_picam.__enter__.return_value = mock_picam

    with patch.dict("sys.modules", {"picamera2": MagicMock(Picamera2=mock_picam_cls)}):
        assert camera.is_imx500_camera() is False


def test_is_imx500_camera_exception() -> None:
    """Test is_imx500_camera returns False on import or camera error."""
    with patch.dict("sys.modules", {"picamera2": None}):
        assert camera.is_imx500_camera() is False
