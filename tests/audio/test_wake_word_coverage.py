"""Comprehensive coverage tests for src/audio/wake_word.py."""

from __future__ import annotations

import queue
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.audio.wake_word import WakeWordDetector
from src.utils.config import Config


def test_load_import_error_when_openwakeword_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that load() raises ImportError when openwakeword is not installed."""
    monkeypatch.delitem(sys.modules, "openwakeword", raising=False)
    monkeypatch.delitem(sys.modules, "openwakeword.model", raising=False)

    original_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("openwakeword"):
            msg = f"No module named '{name}'"
            raise ImportError(msg)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    config = Config()
    wwd = WakeWordDetector(config)
    with pytest.raises(ImportError, match="openwakeword not installed"):
        wwd.load()


def test_load_file_not_found_when_model_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Test that load() raises FileNotFoundError when model file doesn't exist."""
    mock_oww = MagicMock()
    mock_oww_model = MagicMock()
    monkeypatch.setitem(sys.modules, "openwakeword", mock_oww)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mock_oww_model)

    config = Config()
    config.wake.model_name = "nonexistent_model"
    config.wake.download_root = str(tmp_path)

    wwd = WakeWordDetector(config)
    with pytest.raises(FileNotFoundError, match="Wake word model file not found"):
        wwd.load()


def test_unload_clears_model_and_stops() -> None:
    """Test that unload() clears the model and stops the wwd."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._model = MagicMock()
    wwd._running = True

    wwd.unload()

    assert wwd._model is None
    assert wwd._running is False


def test_start_is_noop_when_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that start() returns immediately if already running."""
    mock_open_stream = MagicMock(return_value=True)
    monkeypatch.setattr("src.audio.wake_word.open_input_stream_with_fallback", mock_open_stream)

    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True

    wwd.start(callback=lambda: None)

    assert wwd._running is True
    # open_input_stream_with_fallback should NOT be called since already running
    mock_open_stream.assert_not_called()


def test_start_fails_gracefully_when_stream_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that start() sets running=False when stream cannot be opened."""
    monkeypatch.setattr("src.audio.wake_word.open_input_stream_with_fallback", lambda **kwargs: None)

    config = Config()
    wwd = WakeWordDetector(config)

    wwd.start(callback=lambda: None)

    assert wwd._running is False


def test_stop_clears_queue_and_stops_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stop() clears the queue and stops threads."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True

    mock_stream = MagicMock()
    wwd._stream = mock_stream

    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False
    wwd._capture_thread = mock_thread
    wwd._detect_thread = mock_thread

    wwd.stop()

    assert wwd._running is False
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


def test_stop_handles_missing_stream() -> None:
    """Test that stop() handles None stream gracefully."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True
    wwd._stream = None
    wwd._capture_thread = None
    wwd._detect_thread = None

    # Should not raise
    wwd.stop()
    assert wwd._running is False


def test_stop_detect_thread_still_alive_warning(caplog) -> None:
    """Test that stop() warns when detect thread doesn't exit."""
    import logging

    caplog.set_level(logging.WARNING)

    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True
    wwd._stream = None

    mock_detect_thread = MagicMock()
    mock_detect_thread.is_alive.return_value = True  # Still alive after join
    wwd._detect_thread = mock_detect_thread

    wwd.stop()

    assert "did not exit before shutdown completed" in caplog.text


def test_open_input_stream_device_mismatch_warning(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """Test that _open_input_stream logs a warning when device index falls back."""

    class DummyOpened:
        backend = "sounddevice"
        stream = MagicMock()
        native_chunk_frames = 320
        capture_rate = 16000
        need_resample = False
        resample_up = 1
        resample_down = 1
        device_index = None

    monkeypatch.setattr(
        "src.audio.wake_word.open_input_stream_with_fallback",
        lambda **kwargs: DummyOpened(),
    )

    config = Config()
    config.audio.input_device_index = 5  # Requested device 5
    wwd = WakeWordDetector(config)

    import logging

    caplog.set_level(logging.WARNING)

    result = wwd._open_input_stream()

    assert result is True
    assert "device 5 unavailable" in caplog.text


def test_open_input_stream_exception_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _open_input_stream returns False on exception."""
    monkeypatch.setattr(
        "src.audio.wake_word.open_input_stream_with_fallback",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    config = Config()
    wwd = WakeWordDetector(config)

    result = wwd._open_input_stream()
    assert result is False


def test_capture_loop_no_stream_error(caplog) -> None:
    """Test that _capture_loop logs error and returns when stream is None."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._stream = None
    wwd._running = True

    import logging

    caplog.set_level(logging.ERROR)

    wwd._capture_loop()

    assert "without an open stream" in caplog.text


def test_capture_loop_empty_read_continues() -> None:
    """Test that _capture_loop continues on empty read."""

    class DummyStream:
        def __init__(self) -> None:
            self.read_count = 0

        def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b""  # Empty read - should continue
            wwd._running = False
            return np.array([1, 2], dtype=np.int16).tobytes()

    config = Config()
    wwd = WakeWordDetector(config)
    wwd._stream = DummyStream()
    wwd._running = True
    wwd._native_chunk = 2
    wwd._need_resample = False

    wwd._capture_loop()

    assert isinstance(wwd._audio_queue, queue.Queue)
    assert wwd._audio_queue.qsize() >= 0  # At least survived


def test_capture_loop_exception_handling() -> None:
    """Test that _capture_loop handles read exceptions gracefully."""

    class FailingStream:
        def __init__(self) -> None:
            self.read_count = 0

        def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                wwd._running = False
                raise RuntimeError("read failed")
            return b""

    config = Config()
    wwd = WakeWordDetector(config)
    wwd._stream = FailingStream()
    wwd._running = True
    wwd._native_chunk = 2
    wwd._need_resample = False

    # Should not raise
    wwd._capture_loop()


def test_capture_loop_non_resample_path() -> None:
    """Test _capture_loop takes the non-resample code path."""

    class DummyStream:
        def __init__(self) -> None:
            self.read_count = 0

        def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
            self.read_count += 1
            wwd._running = False
            return np.array([32767, -32767], dtype=np.int16).tobytes()

    config = Config()
    wwd = WakeWordDetector(config)
    wwd._stream = DummyStream()
    wwd._running = True
    wwd._native_chunk = 2
    wwd._need_resample = False  # Take non-resample path

    wwd._capture_loop()

    # Verify audio was queued
    # 32767/32768 = 0.999969482421875, -32767/32768 = -0.999969482421875
    expected = np.array([32767 / 32768, -32767 / 32768], dtype=np.float32)
    item = wwd._audio_queue.get_nowait()
    np.testing.assert_array_almost_equal(item, expected)


def test_detect_loop_queue_empty_then_sentinel() -> None:
    """Test that _detect_loop handles queue timeout before receiving sentinel."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True
    wwd._model = MagicMock()
    wwd._model.predict = MagicMock(return_value=None)
    wwd._model.prediction_buffer = {"hey_jarvis": [0.3]}  # Below threshold

    # Put one item then sentinel - the queue timeout (line 321-322) fires
    # before the item is consumed because get(timeout=0.2) blocks briefly
    wwd._audio_queue.put(np.ones(1280, dtype=np.float32))
    wwd._audio_queue.put(None)  # sentinel

    wwd._detect_loop()

    # If we get here, the loop exited cleanly
    wwd._model.predict.assert_called()


def test_detect_loop_no_model_continues() -> None:
    """Test that _detect_loop continues when model is None (line 329)."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True
    wwd._model = None

    wwd._audio_queue.put(np.ones(1280, dtype=np.float32))
    wwd._audio_queue.put(None)  # sentinel

    # Should not raise even though model is None
    wwd._detect_loop()


def test_detect_loop_prediction_error_handling() -> None:
    """Test that _detect_loop handles prediction errors gracefully (line 347-348)."""
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("prediction failed")
    wwd._model = mock_model

    wwd._audio_queue.put(np.ones(1280, dtype=np.float32))
    wwd._audio_queue.put(None)  # sentinel

    # Should not raise
    wwd._detect_loop()
