"""Threaded camera reader for low-latency frame capture."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

import cv2

if TYPE_CHECKING:
    import numpy as np

    from utils.config import Config

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class ThreadedCamera:
    """Threaded camera reader using OpenCV or Picamera2 for IMX500."""

    started: bool
    read_lock: threading.Lock
    thread: threading.Thread | None  # pyright: ignore[reportRedeclaration]
    frame: np.ndarray[Any, Any] | Any

    def __init__(self, cfg: Config, model_name: str | None = None) -> None:
        """Initialize the threaded camera."""
        self.cfg = cfg
        self.camera_index = cfg.vision.camera.camera_index
        width, height = cfg.vision.get_model_resolution(model_name)
        self.frame_width = width
        self.frame_height = height

        self.use_picamera2 = False
        self.picam2 = None
        self.imx500 = None
        self.metadata = None

        # Check if IMX500 vision is enabled
        if cfg.vision.object_model_type == "yolo_imx500" or cfg.vision.face_detector_type == "imx500":
            try:
                from picamera2 import Picamera2
                from picamera2.devices import IMX500
                from picamera2.devices.imx500 import NetworkIntrinsics

                model_path = str(cfg.vision.object_model_full_path)
                self.imx500 = IMX500(model_path)

                intrinsics = self.imx500.network_intrinsics
                if not intrinsics:
                    intrinsics = NetworkIntrinsics()
                    intrinsics.task = "object detection"

                intrinsics.update_with_defaults()

                self.picam2 = Picamera2(self.imx500.camera_num)
                picam2_config = self.picam2.create_preview_configuration(
                    main={"size": (self.frame_width, self.frame_height), "format": cfg.vision.camera.format},
                    controls={"FrameRate": intrinsics.inference_rate},
                    buffer_count=6,
                )
                self.picam2.configure(picam2_config)
                self.picam2.start()
                self.use_picamera2 = True

                # Pre-allocate frames
                req = self.picam2.capture_request()
                self.grabbed = True
                self.frame = req.make_array("main")
                self.metadata = req.get_metadata()
                req.release()

            except Exception as e:
                logger.exception("Failed to initialize Picamera2 IMX500: %s", e)
                self.use_picamera2 = False

        if not self.use_picamera2:
            self.cap = cv2.VideoCapture(self.camera_index)
            _ = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            _ = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.grabbed, self.frame = self.cap.read()
            self.metadata = None

        self.started = False
        self.read_lock = threading.Lock()
        self.thread: threading.Thread | None = None

        _ = self.start()

    def start(self) -> ThreadedCamera:
        """Start the background frame reading thread."""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self) -> None:
        """Continuously grab frames from the camera in a background thread."""
        while self.started:
            if self.use_picamera2:
                try:
                    req = self.picam2.capture_request()  # type: ignore[attr-defined]
                    frame = req.make_array("main")  # type: ignore[attr-defined]
                    metadata = req.get_metadata()
                    with self.read_lock:
                        self.grabbed = True
                        self.frame = frame
                        self.metadata = metadata
                    req.release()
                except Exception:
                    time.sleep(0.01)
            else:
                grabbed, frame = self.cap.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.metadata = None
                time.sleep(0.01)

    def read(self) -> tuple[np.ndarray[Any, Any] | None, dict[str, Any] | None]:
        """Read the latest frame and metadata.

        Returns:
            Tuple containing the latest camera frame array and metadata dict.

        """
        with self.read_lock:
            if not self.grabbed:
                return None, None
            return (self.frame.copy(), self.metadata)

    def stop(self) -> None:
        """Stop frame reading and release camera resources."""
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.use_picamera2 and self.picam2:
            self.picam2.stop()
        elif hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


class PiCamera:
    """Wrapper around Picamera2 hardware interface to separate it from application code."""

    index: int
    _cam: Any
    camera_info: dict[str, Any]
    started: bool

    def __init__(self, index: int) -> None:
        """Initialize the PiCamera wrapper.

        Args:
            index: Camera device index.

        """
        from picamera2 import Picamera2

        self.index = index
        self._cam = Picamera2(index)
        if hasattr(self._cam, "camera_info"):
            self.camera_info = self._cam.camera_info
        else:
            try:
                global_info = self._cam.global_camera_info()
                self.camera_info = next((info for info in global_info if info.get("Num") == self._cam.camera_idx), {})
                if not self.camera_info and global_info:
                    self.camera_info = global_info[0]
            except Exception:
                self.camera_info = {}

    def get_info_str(self) -> str:
        """Return the string representation of camera_info."""
        return str(self.camera_info)

    def start(self, width: int | None = None, height: int | None = None, video_format: str = "XRGB8888") -> None:
        """Configure the video configuration and start the stream."""
        if width is None or height is None:
            from src.utils.config import config

            w, h = config.vision.get_model_resolution()
            width = width if width is not None else w
            height = height if height is not None else h
        config = self._cam.create_video_configuration(main={"size": (width, height), "format": video_format})
        self._cam.configure(config)
        self._cam.start()  # type: ignore[attr-defined]
        self.started = True

    def capture_frame(self) -> np.ndarray[Any, Any]:
        """Capture a single frame as a numpy array."""
        return self._cam.capture_array()  # type: ignore[attr-defined]

    def stop(self) -> None:
        """Stop camera stream."""
        with contextlib.suppress(Exception):
            self._cam.stop()


def discover_pi_cameras() -> list[PiCamera]:
    """Scan and return a list of available Picamera2 devices."""
    cameras = []
    try:
        import importlib.util

        if importlib.util.find_spec("picamera2") is None:
            return cameras
    except ImportError:
        return cameras

    for idx in [0, 1]:
        with contextlib.suppress(Exception):
            cameras.append(PiCamera(idx))
    return cameras


def is_imx500_camera() -> bool:
    """Check if an IMX500 camera is connected."""
    try:
        from picamera2 import Picamera2

        with Picamera2() as picam2:
            camera_properties = picam2.camera_properties
            model = camera_properties.get("Model", "")
            return "imx500" in str(model).lower()
    except Exception:
        return False
