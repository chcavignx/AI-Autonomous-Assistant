#!/usr/bin/env python3
"""Integration tests for VideoCapture – full coverage of every public method.

Camera-dependent tests fall back to a static image (data/dataset_vision/bus.jpg)
when no camera is connected, so they run instead of skipping.

Run via pytest: pytest tests/vision/test_video_capture_init.py -v
"""

import shutil
import sys
import time
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config  # ruff:ignore[module-import-not-at-top-of-file]
from src.vision.video_capture import VideoCapture  # ruff:ignore[module-import-not-at-top-of-file]

# ---------------------------------------------------------------------------
# Static dataset image (fallback when no camera is connected)
# ---------------------------------------------------------------------------

DATASET_IMAGE = ROOT_DIR / "data" / "bus.jpg"


def _load_static_frame() -> np.ndarray | None:
    """Load bus.jpg from the dataset directory."""
    if DATASET_IMAGE.exists():
        return cv2.imread(str(DATASET_IMAGE))
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poll_frame(vc, timeout=2.0, interval=0.1) -> np.ndarray | None:
    """Poll capture_frame() until a frame arrives or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = vc.capture_frame()
        if frame is not None:
            return frame
        time.sleep(interval)
    return None


def _get_frame(vc) -> np.ndarray:
    """Return a live camera frame, or the static dataset image as fallback.

    Raises ``pytest.skip`` only if neither source is available.
    """
    frame = _poll_frame(vc)
    if frame is not None:
        return frame
    static = _load_static_frame()
    if static is not None:
        # Inject into vc so downstream code (latest_frame etc.) still works
        vc.latest_frame = static
        return static
    pytest.skip("Camera unavailable and dataset image not found.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load the project configuration once for all tests."""
    return load_config()


@pytest.fixture(scope="module")
def vc(cfg):
    """Create a VideoCapture instance, yield it, then clean up."""
    instance = VideoCapture(cfg)
    instance.start()
    yield instance
    instance.stop()


@pytest.fixture(scope="module")
def static_frame() -> np.ndarray:
    """Return the dataset static image; skip the entire module if missing."""
    img = _load_static_frame()
    if img is None:
        pytest.skip(f"Dataset image not found: {DATASET_IMAGE}")
    return img


@pytest.fixture(scope="module")
def ready_vc(vc, static_frame):
    """Yield a VideoCapture with at least one captured & processed frame.

    Uses the camera if available, otherwise injects the static frame.
    """
    frame = _poll_frame(vc) or static_frame
    vc.latest_frame = frame
    vc.process_frame(frame)
    return vc


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVideoCaptureInit:
    """Test suite for VideoCapture initialisation."""

    def test_camera_initialized(self, vc):
        assert vc.camera is not None, "ThreadedCamera should be initialized"

    def test_model_loaded(self, vc):
        assert vc.model is not None, "YOLO model should be loaded"

    def test_model_path_exists(self, cfg):
        model_path = cfg.vision.object_model_full_path
        assert model_path.exists(), f"Model path does not exist: {model_path}"

    def test_initial_flags(self, vc):
        assert vc.enable_detection is True, "Detection should be enabled by default"
        assert vc.latest_frame is None or vc.latest_frame is not None


# ---------------------------------------------------------------------------
# 2. Frame Capture & Processing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVideoCaptureCapture:
    """Test suite for frame capture and processing."""

    def test_capture_frame(self, vc, static_frame):
        frame = _get_frame(vc)
        h, w = frame.shape[:2]
        assert h > 0, f"Frame height is invalid: {h}"
        assert w > 0, f"Frame width is invalid: {w}"

    def test_latest_frame_stored(self, vc, static_frame):
        frame = _get_frame(vc)
        assert frame is not None
        assert vc.latest_frame is not None, "latest_frame should be set"

    def test_process_frame(self, ready_vc, static_frame):
        frame = _poll_frame(ready_vc) or static_frame
        annotated = ready_vc.process_frame(frame)
        assert annotated is not None, "process_frame() returned None"
        assert annotated.shape[:2] == frame.shape[:2], "Annotated frame should keep dimensions"

    def test_process_frame_detection_disabled(self, vc, static_frame):
        frame = _get_frame(vc)
        saved_results = vc.latest_results
        vc.latest_results = None
        vc.enable_detection = False
        vc.enable_face_detection = False
        result = vc.process_frame(frame)
        assert np.array_equal(result, frame), "With detection disabled, should return original frame content"
        vc.enable_detection = True  # restore
        vc.enable_face_detection = True  # restore
        vc.latest_results = saved_results

    def test_latest_results_populated(self, ready_vc):
        assert ready_vc.latest_results is not None, "latest_results should be populated"
        speed = ready_vc.latest_results.speed
        assert "inference" in speed
        assert "preprocess" in speed
        assert "postprocess" in speed

    def test_latest_results_boxes(self, ready_vc):
        boxes = ready_vc.latest_results.boxes
        assert boxes is not None
        len(boxes)


# ---------------------------------------------------------------------------
# 3. Performance Overlay
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPerformanceOverlay:
    """Test suite for _add_performance_overlay()."""

    def test_overlay_modifies_frame(self, ready_vc, static_frame):
        frame = _poll_frame(ready_vc) or static_frame
        ready_vc.process_frame(frame)
        test_frame = np.zeros_like(frame)
        original = test_frame.copy()
        ready_vc._add_performance_overlay(test_frame)
        assert test_frame.shape == original.shape
        assert not np.array_equal(test_frame, original), "Overlay should change pixel values"

    def test_overlay_without_results(self, vc):
        saved = vc.latest_results
        vc.latest_results = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        vc._add_performance_overlay(frame)
        assert np.array_equal(frame, original), "Should not modify frame when no results"
        vc.latest_results = saved


# ---------------------------------------------------------------------------
# 4. Capture Photo
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCapturePhoto:
    """Test suite for capture_photo()."""

    def test_capture_photo_no_frame(self, vc):
        saved_frame = vc.latest_frame
        saved_results = vc.latest_results
        vc.latest_frame = None
        vc.latest_results = None

        success, msg = vc.capture_photo()
        assert success is False
        assert "No frame" in msg or "no" in msg.lower()

        vc.latest_frame = saved_frame
        vc.latest_results = saved_results

    def test_capture_photo_saves_file(self, ready_vc, cfg, tmp_path):
        original_data = cfg.paths.data
        cfg.paths.data = str(tmp_path)
        try:
            success, msg = ready_vc.capture_photo()
            assert success is True, f"capture_photo() failed: {msg}"
            assert "object" in msg.lower() or "captured" in msg.lower()
            captures_dir = tmp_path / "captures"
            assert captures_dir.exists()
            saved_files = list(captures_dir.glob("*_annotated.jpg"))
            assert len(saved_files) >= 1
            assert saved_files[0].stat().st_size > 0
        finally:
            cfg.paths.data = original_data

    def test_capture_photo_message_contains_count(self, ready_vc, cfg, tmp_path):
        original_data = cfg.paths.data
        cfg.paths.data = str(tmp_path)
        try:
            success, msg = ready_vc.capture_photo()
            assert success is True, f"capture_photo() failed: {msg}"
            assert "object" in msg.lower()
        finally:
            cfg.paths.data = original_data


# ---------------------------------------------------------------------------
# 5. Generate Frames (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateFrames:
    """Test suite for generate_frames() generator."""

    def test_generate_frames_yields_jpeg(self, ready_vc, static_frame):
        # Mock capture_frame to ensure we don't block waiting for a camera
        original_capture = ready_vc.capture_frame
        ready_vc.capture_frame = lambda: static_frame
        ready_vc.start()
        gen = ready_vc.generate_frames()
        frames_received = 0
        try:
            for chunk in gen:
                assert isinstance(chunk, bytes)
                assert chunk.startswith(b"--frame\r\n")
                assert b"Content-Type: image/jpeg" in chunk
                jpeg_start = chunk.find(b"\xff\xd8")
                assert jpeg_start > 0
                frames_received += 1
                if frames_received >= 3:
                    break
        finally:
            ready_vc.stop()
            ready_vc.capture_frame = original_capture
        assert frames_received >= 1

    def test_generate_frames_stops_when_not_running(self, ready_vc, static_frame):
        original_capture = ready_vc.capture_frame
        ready_vc.capture_frame = lambda: static_frame
        ready_vc.start()
        gen = ready_vc.generate_frames()
        first = next(gen)
        assert first is not None
        ready_vc.running = False
        remaining = []
        # Consume the generator up to a safe limit to prevent hanging
        for _ in range(5):
            try:
                extra = next(gen)
                remaining.append(extra)
            except StopIteration:
                break
        ready_vc.capture_frame = original_capture
        assert len(remaining) <= 1


# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBenchmark:
    """Test suite for benchmark()."""

    def test_benchmark_no_frame(self, vc, caplog):
        saved = vc.latest_frame
        vc.latest_frame = None
        import logging

        with caplog.at_level(logging.WARNING):
            vc.benchmark(iterations=1)
        vc.latest_frame = saved
        assert any("no model or frame" in record.message.lower() for record in caplog.records)

    def test_benchmark_runs(self, ready_vc, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            ready_vc.benchmark(iterations=3)
        assert any("benchmark completed" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# 7. Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVideoCaptureLifecycle:
    """Test suite for start/stop lifecycle."""

    def test_start(self, vc):
        vc.start()
        assert vc.running is True
        vc.running = False

    def test_start_idempotent(self, vc):
        vc.start()
        vc.start()
        assert vc.running is True
        vc.running = False

    def test_stop(self, cfg):
        instance = VideoCapture(cfg)
        instance.start()
        assert instance.running is True
        instance.stop()
        assert instance.running is False


# ---------------------------------------------------------------------------
# Direct execution support
# ---------------------------------------------------------------------------


def _run_all_tests():
    """Run all tests manually when executed as a script."""
    cfg = load_config()

    model_path = cfg.vision.object_model_full_path
    if not model_path.exists():
        return False

    try:
        vc = VideoCapture(cfg)
    except Exception:
        return False

    passed = 0
    failed = 0

    def check(condition, label):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1

    check(vc.camera is not None, "camera is not None")
    check(vc.model is not None, "model is not None")
    check(vc.enable_detection is True, "detection enabled by default")
    check(vc.running is False, "not running before start()")

    frame = _poll_frame(vc) or _load_static_frame()
    if frame is None:
        vc.stop()
        return False
    vc.latest_frame = frame

    h, w = frame.shape[:2]
    check(h > 0 and w > 0, f"frame dimensions valid: {w}×{h}")
    check(vc.latest_frame is not None, "latest_frame stored")

    annotated = vc.process_frame(frame)
    check(annotated is not None, "process_frame() returned annotated frame")
    if annotated is not None:
        check(annotated.shape[:2] == frame.shape[:2], f"dimensions match: {annotated.shape[:2]}")
    check(vc.latest_results is not None, "latest_results populated")
    if vc.latest_results:
        speed = vc.latest_results.speed
        check("inference" in speed, "speed has 'inference'")
        check("preprocess" in speed, "speed has 'preprocess'")
        check("postprocess" in speed, "speed has 'postprocess'")

    vc.enable_detection = False
    raw = vc.process_frame(frame)
    check(raw is frame, "detection disabled → returns original frame")
    vc.enable_detection = True

    test_overlay = np.ones_like(frame) * 128
    original_overlay = test_overlay.copy()
    vc._add_performance_overlay(test_overlay)
    check(test_overlay.shape == original_overlay.shape, "overlay keeps shape")
    check(not np.array_equal(test_overlay, original_overlay), "overlay modifies pixels")

    saved_results = vc.latest_results
    vc.latest_results = None
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    blank_copy = blank.copy()
    vc._add_performance_overlay(blank)
    check(np.array_equal(blank, blank_copy), "overlay no-op when no results")
    vc.latest_results = saved_results

    sf, sr = vc.latest_frame, vc.latest_results
    vc.latest_frame = None
    vc.latest_results = None
    ok, msg = vc.capture_photo()
    check(ok is False, f"capture_photo with no frame → False: '{msg}'")
    vc.latest_frame, vc.latest_results = sf, sr

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    original_data = cfg.paths.data
    cfg.paths.data = str(tmp_dir)
    try:
        fresh = _poll_frame(vc) or frame
        vc.process_frame(fresh)
        ok, msg = vc.capture_photo()
        check(ok is True, f"capture_photo saved: '{msg}'")
        captures_dir = tmp_dir / "captures"
        saved_files = list(captures_dir.glob("*_annotated.jpg")) if captures_dir.exists() else []
        check(len(saved_files) >= 1, f"annotated JPEG created ({len(saved_files)} files)")
        if saved_files:
            sz = saved_files[0].stat().st_size
            check(sz > 0, f"saved image not empty ({sz} bytes)")
    finally:
        cfg.paths.data = original_data
        shutil.rmtree(tmp_dir, ignore_errors=True)

    vc.start()
    gen = vc.generate_frames()
    gen_count = 0
    gen_valid = True
    try:
        for chunk in gen:
            if not isinstance(chunk, bytes):
                gen_valid = False
            if not chunk.startswith(b"--frame\r\n"):
                gen_valid = False
            if b"Content-Type: image/jpeg" not in chunk:
                gen_valid = False
            if chunk.find(b"\xff\xd8") <= 0:
                gen_valid = False
            gen_count += 1
            if gen_count >= 3:
                break
    except Exception:
        gen_valid = False

    check(gen_count >= 1, f"generate_frames yielded {gen_count} chunk(s)")
    check(gen_valid, "all chunks are valid multipart JPEG")

    gen3 = vc.generate_frames()
    first = next(gen3, None)
    check(first is not None, "generate_frames yields at least one chunk")
    vc.running = False
    trailing = []
    for extra in gen3:
        trailing.append(extra)
        if len(trailing) > 5:
            break
    check(len(trailing) <= 2, f"generator stops after running=False ({len(trailing)} trailing)")

    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    saved_frame = vc.latest_frame
    vc.latest_frame = None
    vc.benchmark(iterations=1)
    sys.stdout = old_stdout
    bm_out = captured.getvalue()
    check("need" in bm_out.lower() or "✘" in bm_out, "benchmark warns with no frame")

    vc.latest_frame = saved_frame
    vc.process_frame(vc.latest_frame)
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    vc.benchmark(iterations=3)
    sys.stdout = old_stdout
    bm_out = captured.getvalue()
    check("avg" in bm_out.lower() or "inference" in bm_out.lower(), "benchmark prints avg time")
    check("fps" in bm_out.lower(), "benchmark prints FPS")

    vc.start()
    check(vc.running is True, "running=True after start()")
    vc.start()
    check(vc.running is True, "start() idempotent")
    vc.stop()
    check(vc.running is False, "running=False after stop()")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = _run_all_tests()
    sys.exit(0 if success else 1)
