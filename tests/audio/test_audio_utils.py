from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, Never, Self

import numpy as np
from src.audio.audio_utils import AudioUtils, install_alsa_error_handler, suppress_pa_stderr

if TYPE_CHECKING:
    from collections.abc import Callable
    from ctypes import _CFunctionType

    from numpy._typing import _32Bit
    from numpy._typing._array_like import NDArray
    from numpy._typing._dtype_like import floating
    from numpy._typing._nested_sequence import ndarray


def test_validate_and_clean_audio_handles_nan_inf_and_extremes() -> None:
    audio: NDArray[floating[_32Bit]] = np.array([0.0, np.nan, np.inf, -np.inf, 1e12, -1e12], dtype=np.float32)
    cleaned: ndarray[Any, Any] = AudioUtils.validate_and_clean_audio(audio_data=audio)
    assert cleaned.dtype == np.float32
    assert np.all(np.isfinite(cleaned))
    assert cleaned.size == audio.size


def test_validate_and_clean_audio_handles_empty_array() -> None:
    audio: NDArray[floating[_32Bit]] = np.array([], dtype=np.float32)
    cleaned: ndarray[Any, Any] = AudioUtils.validate_and_clean_audio(audio_data=audio)
    assert cleaned.dtype == np.float32
    assert cleaned.size == 0


def test_convert_to_int16_scales_and_clips() -> None:
    audio: NDArray[floating[_32Bit]] = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    converted: ndarray[Any, Any] = AudioUtils.convert_to_int16(audio_float=audio)
    assert converted.dtype == np.int16
    assert converted[0] == 0
    # The cleaner rescales to avoid clipping, so values are slightly below max.
    assert converted[1] > 30000
    assert converted[2] < -30000


def test_suppress_pa_stderr_redirects_fd2(monkeypatch) -> None:
    read_fd, write_fd = os.pipe()
    original_open: Callable[..., int] = os.open

    def fake_open(path, flags, *args, **kwargs) -> int:
        if path == os.devnull:
            return write_fd
        return original_open(path, flags, *args, **kwargs)

    from src.audio import audio_utils

    monkeypatch.setattr(audio_utils.os, "open", fake_open)

    with suppress_pa_stderr():
        os.write(2, b"suppressed\n")

    output: bytes = os.read(read_fd, 100)
    assert output == b"suppressed\n"
    os.close(read_fd)


def test_install_alsa_error_handler_returns_callable(monkeypatch) -> None:
    dummy_ctypes: ModuleType = ModuleType(name="ctypes")
    dummy_ctypes.c_char_p = int
    dummy_ctypes.c_int = int

    def CFUNCTYPE(*args, **kwargs) -> Callable[..., Any]:
        return lambda func: func

    class DummyLoader:
        def LoadLibrary(self, name) -> Self:
            assert name == "libasound.so.2"
            return self

        def snd_lib_error_set_handler(self, handler) -> None:
            self.handler: Any = handler

    dummy_ctypes.CFUNCTYPE = CFUNCTYPE
    dummy_ctypes.cdll = DummyLoader()
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    handler: _CFunctionType | None = install_alsa_error_handler()
    assert callable(handler)


def test_install_alsa_error_handler_handles_load_failure(monkeypatch) -> None:
    from src.audio import audio_utils

    class DummyLoader:
        def LoadLibrary(self, name) -> Never:
            raise OSError("Library not found")

    dummy_ctypes: ModuleType = ModuleType(name="ctypes")
    dummy_ctypes.cdll = DummyLoader()
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    handler: _CFunctionType | None = audio_utils.install_alsa_error_handler()
    assert handler is None


def test_play_audio_file_calls_system_player(monkeypatch) -> None:
    called: dict[Any, Any] = {}

    def fake_run(args, capture_output, check) -> None:
        called["args"] = args
        called["capture_output"] = capture_output
        called["check"] = check

    from src.audio import audio_utils

    monkeypatch.setattr(audio_utils.subprocess, "run", fake_run)
    AudioUtils.play_audio_file(file_path="audio.wav")
    assert called["args"] == ["aplay", "audio.wav"]
    assert called["capture_output"] is True
    assert called["check"] is False


def test_play_audio_file_handles_subprocess_error(monkeypatch) -> None:
    import subprocess

    def fake_run(args, capture_output, check) -> Never:
        raise subprocess.SubprocessError("Command failed")

    from src.audio import audio_utils

    monkeypatch.setattr(audio_utils.subprocess, "run", fake_run)
    # Should not raise
    AudioUtils.play_audio_file(file_path="audio.wav")


def test_play_audio_file_to_sd_uses_sounddevice(monkeypatch) -> None:
    from src.audio import audio_utils

    audio_data: NDArray[floating[_32Bit]] = np.ones((4, 1), dtype=np.float32)
    sample_rate = 16000
    monkeypatch.setattr(audio_utils.sf, "read", lambda file_path, dtype: (audio_data, sample_rate))

    class DummySDModule(ModuleType):
        def __init__(self) -> None:
            super().__init__(name="sounddevice")
            self.played = False
            self.waited = False
            self.stopped = False

        def play(self, data, samplerate) -> None:
            assert samplerate == sample_rate
            self.played = True

        def wait(self) -> None:
            self.waited = True

        def stop(self) -> None:
            self.stopped = True

        class PortAudioError(Exception):
            pass

    dummy_sd: DummySDModule = DummySDModule()
    monkeypatch.setitem(sys.modules, "sounddevice", dummy_sd)

    AudioUtils.play_audio_file_to_sd(file_path="audio.wav")
    assert dummy_sd.played
    assert dummy_sd.waited
    assert dummy_sd.stopped


def test_play_audio_file_to_sd_handles_read_error(monkeypatch) -> None:
    import soundfile as sf
    from src.audio import audio_utils

    def fake_read(file_path, dtype) -> Never:
        raise sf.LibsndfileError(code="Read failed")

    monkeypatch.setattr(audio_utils.sf, "read", fake_read)
    # Should not raise
    AudioUtils.play_audio_file_to_sd(file_path="audio.wav")


def test_play_audio_stream_writes_chunks(monkeypatch) -> None:

    created_stream: dict[Any, Any] = {}

    class DummyStream:
        def __init__(self) -> None:
            self.started = False
            self.written: list[Any] = []
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def write(self, payload) -> None:
            self.written.append(payload)

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    class DummySDModule(ModuleType):
        def __init__(self):
            super().__init__(name="sounddevice")
            self.PortAudioError: type[Exception] = Exception

        def OutputStream(self, samplerate, channels, dtype) -> DummyStream:
            created_stream["stream"] = DummyStream()
            return created_stream["stream"]

    dummy_sd_module: DummySDModule = DummySDModule()
    monkeypatch.setitem(sys.modules, "sounddevice", dummy_sd_module)

    class Packet:
        def __init__(self, array) -> None:
            self.audio_int16_array = array

    # Create a single numpy array from the packet data
    audio_data = np.concatenate([np.array([1, 2], dtype=np.int16), np.array([3, 4], dtype=np.int16)])
    AudioUtils.play_audio_stream(audio_stream=audio_data, sample_rate=16000, channels=1, dtype="int16")
    assert created_stream["stream"].started
    assert np.array_equal(a1=created_stream["stream"].written[0], a2=audio_data)
    assert created_stream["stream"].stopped
    assert created_stream["stream"].closed
    assert created_stream["stream"].stopped
    assert created_stream["stream"].closed


def test_validate_and_clean_audio_adds_dither_to_silence() -> None:
    audio: NDArray[floating[_32Bit]] = np.zeros(1000, dtype=np.float32)
    cleaned: ndarray[Any, Any] = AudioUtils.validate_and_clean_audio(audio_data=audio)
    assert cleaned.dtype == np.float32
    assert cleaned.size == audio.size
    assert not np.allclose(cleaned, 0.0)
