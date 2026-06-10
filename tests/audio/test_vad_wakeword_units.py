from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
from src.audio.vad import VADEngine
from src.audio.wake_word import WakeWordDetector
from src.utils.config import Config

if TYPE_CHECKING:
    import pytest


def test_vad_is_speech_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_config = MagicMock()
    mock_config.vad.threshold = 0.5
    mock_config.vad.sample_rate = 16000
    mock_config.cpu_cores = 4
    mock_config.sample_rate = 16000
    mock_config.stt_language = "en"
    mock_config.stt_model_size = "base"
    mock_config.faster_stt_model_size = "small"
    monkeypatch.setattr("src.audio.vad.load_silero_vad", MagicMock)

    def mock_get_speech_timestamps(audio, model, min_speech_duration_ms):
        return [{"start": 0, "end": 1}]

    monkeypatch.setattr("src.audio.vad.get_speech_timestamps", mock_get_speech_timestamps)

    vad = VADEngine(mock_config)

    audio = np.ones(16000, dtype=np.float32)  # 1 second at 16kHz
    assert vad.is_speech_detected(audio)

    # Short audio
    short_audio = np.ones(1000, dtype=np.float32)
    assert not vad.is_speech_detected(short_audio)


def test_vad_get_speech_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.audio.vad.load_silero_vad", MagicMock)
    config = Config()
    vad = VADEngine(config)

    segments = [{"start": 0.0, "end": 1.0}]

    def mock_get_speech_timestamps(audio, model, min_speech_duration_ms, min_silence_duration_ms, return_seconds):
        return segments

    import src.audio.vad as vad_module

    monkeypatch.setattr(vad_module, "get_speech_timestamps", mock_get_speech_timestamps)

    audio = np.ones(16000, dtype=np.float32)
    result = vad.get_speech_segments(audio)
    assert result == segments


def test_vad_get_speech_segments_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.audio.vad.load_silero_vad", MagicMock)
    config = Config()
    vad = VADEngine(config)

    # Mock error
    monkeypatch.setattr("src.audio.vad.get_speech_timestamps", MagicMock(side_effect=RuntimeError("Mock error")))

    audio = np.ones(16000, dtype=np.float32)
    result = vad.get_speech_segments(audio)
    assert result == []


def test_wake_word_load(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        model_file = tmp_path / "mock_model.onnx"
        model_file.touch()

        mock_config = MagicMock()
        mock_config.wake.model_name = "mock_model"
        mock_config.wake.full_model_path = model_file
        mock_config.wake.inference_framework = "onnx"
        mock_config.wake.download_path = tmp_path
        mock_config.cpu_cores = 4

        loaded = {}

        def mock_Model(wakeword_models, inference_framework, **kwargs):
            loaded["wakeword_models"] = wakeword_models
            loaded["inference_framework"] = inference_framework
            return "mock_model"

        monkeypatch.setattr("openwakeword.model.Model", mock_Model)

        wwd = WakeWordDetector(mock_config)
        wwd.load()
        assert loaded["wakeword_models"] == [str(model_file)]
        assert wwd._model == "mock_model"


def test_wake_word_detect_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    wwd = WakeWordDetector(config)
    wwd._running = True
    wwd._model = MagicMock()
    wwd._model.prediction_buffer = {"hey_jarvis": [0.9]}

    # Mock callback
    callback_called = False

    def mock_callback():
        nonlocal callback_called
        callback_called = True

    wwd._callback = mock_callback

    # Put audio in queue
    wwd._audio_queue.put(np.ones(1280, dtype=np.float32))
    wwd._audio_queue.put(None)  # Sentinel

    # Run one iteration
    wwd._detect_loop()
    assert callback_called


def test_wake_word_open_input_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    wwd = WakeWordDetector(config)

    mock_sd = MagicMock()
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream
    mock_sd.query_devices.return_value = [{"name": "Mic", "max_input_channels": 1, "default_samplerate": 16000.0}]

    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    result = wwd._open_input_stream()
    assert result is True
