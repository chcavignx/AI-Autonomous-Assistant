from __future__ import annotations

import builtins
import io
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Never, Self

import numpy as np
import pytest
from src.audio import audio_utils

if TYPE_CHECKING:
    from collections.abc import Callable


def make_config(
    *,
    backend: str = "sounddevice",
    input_device_index: int | None = None,
    output_device_index: int | None = None,
    input_sample_rate: int = 16000,
    input_chunk_size: int = 1024,
) -> SimpleNamespace:
    return SimpleNamespace(
        audio=SimpleNamespace(
            backend=backend,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            input_sample_rate=input_sample_rate,
            input_chunk_size=input_chunk_size,
        )
    )


def install_sounddevice_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: list[dict[str, Any]] | None = None,
) -> ModuleType:
    module = ModuleType("sounddevice")
    module.play_calls: list[dict[str, Any]] = []
    module.wait_called = False
    module.stop_called = False

    class DummyInputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.callback = kwargs.get("callback")
            self.started = False
            self.stopped = False
            self.closed = False
            self.written: list[Any] = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

        def write(self, data: Any) -> None:
            self.written.append(data)

    class DummyOutputStream(DummyInputStream):
        pass

    def play(data: Any, samplerate: int, device: Any = None) -> None:
        module.play_calls.append({"data": data, "samplerate": samplerate, "device": device})

    def wait() -> None:
        module.wait_called = True

    def stop() -> None:
        module.stop_called = True

    def query_devices() -> list[dict[str, Any]]:
        return devices or [
            {
                "name": "Mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
            {
                "name": "Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]

    module.InputStream = DummyInputStream
    module.OutputStream = DummyOutputStream
    module.play = play
    module.wait = wait
    module.stop = stop
    module.query_devices = query_devices
    module.PortAudioError = Exception
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


class DummyStream:
    def __init__(self, read_value: bytes = b"chunk") -> None:
        """Initialize mock stream."""
        self.read_calls: list[tuple[int, bool]] = []
        self.write_calls: list[bytes] = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.read_value = read_value

    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
        self.read_calls.append((num_frames, exception_on_overflow))
        return self.read_value

    def write(self, data: bytes) -> None:
        self.write_calls.append(data)

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class DummyPyAudio:
    def __init__(self, devices: list[dict[str, Any]] | None = None, read_value: bytes = b"chunk") -> None:
        """Initialize mock PyAudio."""
        self.open_calls: list[dict[str, Any]] = []
        self.terminated = False
        self.stream = DummyStream(read_value)
        self.devices = devices or []

    def open(self, **kwargs: Any) -> DummyStream:
        self.open_calls.append(kwargs)
        return self.stream

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_info_by_index(self, index: int) -> dict[str, Any]:
        return self.devices[index]

    def terminate(self) -> None:
        self.terminated = True


def install_pyaudio_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: list[dict[str, Any]] | None = None,
    read_value: bytes = b"chunk",
) -> ModuleType:
    module = ModuleType("pyaudio")
    module.paInt16 = 8

    class ConfiguredDummyPyAudio(DummyPyAudio):
        def __init__(self) -> None:
            super().__init__(devices=devices, read_value=read_value)

    module.PyAudio = ConfiguredDummyPyAudio
    monkeypatch.setitem(sys.modules, "pyaudio", module)
    return module


def test_get_chunk_frames_rounds_and_never_returns_less_than_one() -> None:
    assert audio_utils.get_chunk_frames(sample_rate=16000, chunk_ms=20) == 320
    assert audio_utils.get_chunk_frames(sample_rate=16000, chunk_ms=1) == 16
    assert audio_utils.get_chunk_frames(sample_rate=16000, chunk_ms=0) == 1


def test_validate_and_clean_audio_replaces_invalid_values_and_normalizes() -> None:
    audio = np.array([0.0, np.nan, np.inf, -np.inf, 1e12, -1e12], dtype=np.float32)

    cleaned = audio_utils.validate_and_clean_audio(audio)

    assert cleaned.dtype == np.float32
    assert np.array_equal(
        cleaned,
        np.array([0.0, 0.0, 0.95, -0.95, 0.95, -0.95], dtype=np.float32),
    )


def test_validate_and_clean_audio_adds_dither_to_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.zeros(4, dtype=np.float32)
    monkeypatch.setattr(
        audio_utils.np.random,
        "normal",
        lambda loc, scale, size: np.full(size, 1e-8, dtype=np.float32),
    )

    cleaned = audio_utils.validate_and_clean_audio(audio)

    assert cleaned.dtype == np.float32
    assert np.array_equal(cleaned, np.full(4, 1e-8, dtype=np.float32))


def test_convert_to_int16_uses_clean_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "validate_and_clean_audio",
        lambda audio_data: np.array([0.0, 0.5, -0.5], dtype=np.float32),
    )

    converted = audio_utils.convert_to_int16(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert converted.dtype == np.int16
    assert np.array_equal(converted, np.array([0, 16383, -16383], dtype=np.int16))


def test_convert_to_float32_normalizes_int16() -> None:
    converted = audio_utils.convert_to_float32(np.array([0, 32767, -32768], dtype=np.int16))

    assert converted.dtype == np.float32
    assert np.allclose(converted, np.array([0.0, 32767 / 32768.0, -1.0], dtype=np.float32))


def test_audio_device_info_capabilities() -> None:
    device = audio_utils.AudioDeviceInfo(
        index=3,
        name="Mic",
        max_input_channels=2,
        max_output_channels=0,
    )

    assert device.can_capture() is True
    assert device.can_playback() is False


def test_resolve_device_index_returns_none_for_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [
            audio_utils.AudioDeviceInfo(index=0, name="out", max_output_channels=2),
            audio_utils.AudioDeviceInfo(index=1, name="in", max_input_channels=1),
        ],
    )

    assert audio_utils.resolve_device_index(device_index=99, is_input=True, backend="sounddevice") is None


def test_resolve_device_index_returns_none_for_wrong_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [
            audio_utils.AudioDeviceInfo(index=0, name="speaker", max_output_channels=2),
        ],
    )

    assert audio_utils.resolve_device_index(device_index=0, is_input=True, backend="sounddevice") is None


def test_resolve_device_index_keeps_valid_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [
            audio_utils.AudioDeviceInfo(index=0, name="mic", max_input_channels=1),
        ],
    )

    assert audio_utils.resolve_device_index(device_index=0, is_input=True, backend="sounddevice") == 0


def test_suppress_pa_stderr_redirects_fd2(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    original_open: Callable[..., int] = os.open

    def fake_open(path: str, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == os.devnull:
            return write_fd
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(audio_utils.os, "open", fake_open)

    try:
        with audio_utils.suppress_pa_stderr():
            os.write(2, b"suppressed\n")

        output = os.read(read_fd, 100)
        assert output == b"suppressed\n"
    finally:
        os.close(read_fd)


def test_install_alsa_error_handler_returns_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_ctypes: ModuleType = ModuleType(name="ctypes")
    dummy_ctypes.c_char_p = int
    dummy_ctypes.c_int = int

    def cfunctype(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        return lambda func: func

    class DummyLoader:
        def LoadLibrary(self, name: str) -> Self:
            assert name == "libasound.so.2"
            return self

        def snd_lib_error_set_handler(self, handler: Any) -> None:
            self.handler = handler

    dummy_ctypes.CFUNCTYPE = cfunctype
    dummy_ctypes.cdll = DummyLoader()
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    handler = audio_utils.install_alsa_error_handler()

    assert callable(handler)


def test_install_alsa_error_handler_handles_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_ctypes: ModuleType = ModuleType(name="ctypes")

    class DummyLoader:
        def LoadLibrary(self, name: str) -> Never:
            raise OSError("Library not found")

    dummy_ctypes.cdll = DummyLoader()
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    assert audio_utils.install_alsa_error_handler() is None


def test_audio_utils_play_audio_file_uses_system_player_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    class DummyPlayer:
        def __init__(self) -> None:
            called["init"] = True

        def play_with_system_player(self, file_path: str) -> bool:
            called["file_path"] = file_path
            return True

        def close(self) -> None:
            called["closed"] = True

    monkeypatch.setattr(audio_utils, "AudioPlayer", DummyPlayer)

    audio_utils.AudioUtils.play_audio_file("audio.wav")

    assert called == {"init": True, "file_path": "audio.wav", "closed": True}


def test_audio_utils_play_audio_file_to_sd_forces_sounddevice_path(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    class DummyPlayer:
        def __init__(self, config: Any = None) -> None:
            called["init"] = True
            self._backend = "pyaudio"

        def play_file(self, file_path: str) -> bool:
            called["file_path"] = file_path
            called["backend"] = self._backend
            return True

        def close(self) -> None:
            called["closed"] = True

    monkeypatch.setattr(audio_utils, "AudioPlayer", DummyPlayer)

    audio_utils.AudioUtils.play_audio_file_to_sd("audio.wav")

    assert called == {"init": True, "file_path": "audio.wav", "backend": "sounddevice", "closed": True}


def test_audio_utils_play_audio_stream_converts_int16_before_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    class DummyPlayer:
        def __init__(self) -> None:
            called["init"] = True

        def play_data(self, audio_data: np.ndarray, sample_rate: int, block: bool = True) -> bool:
            called["audio_data"] = audio_data
            called["sample_rate"] = sample_rate
            called["block"] = block
            return True

        def close(self) -> None:
            called["closed"] = True

    monkeypatch.setattr(audio_utils, "AudioPlayer", DummyPlayer)

    audio_stream = np.array([0, 32767, -32768], dtype=np.int16)
    audio_utils.AudioUtils.play_audio_stream(audio_stream, sample_rate=16000, channels=1, dtype="int16")

    assert called["init"] is True
    assert called["sample_rate"] == 16000
    assert called["block"] is True
    assert called["closed"] is True
    assert called["audio_data"].dtype == np.float32
    assert np.allclose(called["audio_data"], np.array([0.0, 32767 / 32768.0, -1.0], dtype=np.float32))


def test_backend_selection_availability_and_device_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    sd_module = install_sounddevice_module(monkeypatch)
    pa_module = install_pyaudio_module(
        monkeypatch,
        devices=[
            {
                "name": "Mic",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 16000.0,
            },
            {
                "name": "Speaker",
                "maxInputChannels": 0,
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
        ],
    )

    assert audio_utils.get_audio_backend(make_config(backend="sounddevice")) == "sounddevice"
    assert audio_utils.get_audio_backend(make_config(backend="pyaudio")) == "pyaudio"
    assert audio_utils.is_backend_available("sounddevice") is True
    assert audio_utils.is_backend_available("pyaudio") is True
    assert audio_utils.is_backend_available("bogus") is False

    sounddevice_devices = audio_utils.list_audio_devices("sounddevice")
    assert [dev.name for dev in sounddevice_devices] == ["Mic", "Speaker"]
    assert audio_utils.get_default_input_device("sounddevice").name == "Mic"
    assert audio_utils.get_default_output_device("sounddevice").name == "Speaker"

    pyaudio_devices = audio_utils.list_audio_devices("pyaudio")
    assert [dev.name for dev in pyaudio_devices] == ["Mic", "Speaker"]
    assert audio_utils.get_default_input_device("pyaudio").name == "Mic"
    assert audio_utils.get_default_output_device("pyaudio").name == "Speaker"

    assert sd_module is not None
    assert pa_module is not None


def test_backend_selection_auto_and_invalid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sounddevice_module(monkeypatch)
    assert audio_utils.get_audio_backend(make_config(backend="auto")) == "sounddevice"

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    install_pyaudio_module(monkeypatch)
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "sounddevice":
            raise ImportError("sounddevice missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert audio_utils.get_audio_backend(make_config(backend="auto")) == "pyaudio"

    monkeypatch.delitem(sys.modules, "pyaudio", raising=False)

    def missing_all(name: str, *args: Any, **kwargs: Any):
        if name in {"sounddevice", "pyaudio"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_all)
    with pytest.raises(ImportError, match="No audio backend available"):
        audio_utils.get_audio_backend(make_config(backend="auto"))


def test_backend_selection_unknown_backend_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sounddevice_module(monkeypatch)
    # The current implementation defaults to 'auto' when unknown, and get_audio_backend will try SD then PA.
    assert audio_utils.get_audio_backend(make_config(backend="mystery")) == "sounddevice"


def test_device_index_helpers_and_opened_input_stream_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [
            audio_utils.AudioDeviceInfo(index=0, name="speaker", max_output_channels=2),
            audio_utils.AudioDeviceInfo(index=1, name="mic", max_input_channels=1),
        ],
    )

    assert audio_utils.resolve_device_index(None, is_input=True, backend="sounddevice") is None
    assert audio_utils.resolve_device_index(1, is_input=True, backend="sounddevice") == 1
    assert audio_utils._candidate_device_indexes(backend="sounddevice", device_index=1, is_input=True) == [1, None]
    assert audio_utils._candidate_device_indexes(backend="sounddevice", device_index=0, is_input=True) == [None]

    stream = object()
    opened = audio_utils._opened_input_stream(
        stream=stream,  # type: ignore[arg-type]
        backend="sounddevice",
        capture_rate=16000,
        native_chunk_frames=320,
        device_index=1,
        requested_rate=48000,
    )

    assert opened.need_resample is True
    assert opened.resample_up == 3
    assert opened.resample_down == 1


def test_open_input_stream_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySDInputStream:
        created: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            DummySDInputStream.created.append(kwargs)

        def start(self) -> bool:
            return self.kwargs["rate"] == 44100

    class DummyPaInputStream:
        created: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            DummyPaInputStream.created.append(kwargs)

        def start(self) -> bool:
            return self.kwargs["rate"] == 16000

    monkeypatch.setattr(audio_utils, "SoundDeviceInputStream", DummySDInputStream)
    result = audio_utils.open_input_stream_with_fallback(
        rate=48000,
        chunk_ms=20,
        backend="sounddevice",
        candidate_rates=[48000, 44100],
    )
    assert result is not None
    assert result.capture_rate == 44100
    assert result.need_resample is True

    monkeypatch.setattr(audio_utils, "PyAudioInputStream", DummyPaInputStream)
    result = audio_utils.open_input_stream_with_fallback(
        rate=16000,
        chunk_ms=20,
        backend="pyaudio",
        candidate_rates=[8000, 16000],
    )
    assert result is not None
    assert result.capture_rate == 16000
    assert result.need_resample is False

    # Current implementation ignores the backend argument if it's not None/auto in some calls,
    # but open_input_stream_with_fallback does use it. However, it doesn't raise ImportError
    # for "bogus" because it calls get_audio_backend() if backend is None/auto.
    # If explicitly passed "bogus", it currently raises ImportError at the end.
    with pytest.raises(ImportError, match="Unknown audio backend"):
        audio_utils.open_input_stream_with_fallback(rate=16000, chunk_ms=20, backend="bogus")


def test_open_input_stream_with_fallback_auto_backend_and_pyaudio_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_utils, "get_audio_backend", lambda config=None: "sounddevice")
    monkeypatch.setattr(
        audio_utils,
        "SoundDeviceInputStream",
        lambda **kwargs: type("DummySD", (), {"start": lambda self: False})(),
    )

    assert audio_utils.open_input_stream_with_fallback(rate=16000, chunk_ms=20, backend=None) is None

    class ExplodingPyAudioInputStream:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(audio_utils, "PyAudioInputStream", ExplodingPyAudioInputStream)
    assert audio_utils.open_input_stream_with_fallback(rate=16000, chunk_ms=20, backend="pyaudio") is None


def test_list_and_resolve_device_index_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_utils, "get_audio_backend", lambda config=None: "sounddevice")
    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [audio_utils.AudioDeviceInfo(index=0, name="mic", max_input_channels=1)],
    )

    assert [dev.name for dev in audio_utils.list_audio_devices(None)] == ["mic"]
    assert audio_utils.resolve_device_index(0, is_input=True, backend=None) == 0


def test_resample_audio_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    assert np.array_equal(audio_utils.resample_audio(audio, 16000, 16000), audio)

    scipy_signal = ModuleType("scipy.signal")
    scipy_signal.called: dict[str, Any] = {}

    def fake_resample_poly(data: np.ndarray, up: int, down: int) -> np.ndarray:
        scipy_signal.called = {"up": up, "down": down, "shape": data.shape}
        return np.array([1.0, 2.0], dtype=np.float64)

    scipy_signal.resample_poly = fake_resample_poly
    scipy_module = ModuleType("scipy")
    scipy_module.signal = scipy_signal
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.signal", scipy_signal)

    resampled = audio_utils.resample_audio(audio, 16000, 32000)
    assert np.array_equal(resampled, np.array([1.0, 2.0], dtype=np.float32))
    assert scipy_signal.called["up"] == 2
    assert scipy_signal.called["down"] == 1

    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.signal", raising=False)
    fallback = audio_utils.resample_audio(audio, original_rate=4, target_rate=2)
    assert fallback.dtype == np.float32
    assert len(fallback) == 2


def test_sounddevice_streams_cover_start_read_stop_close(monkeypatch: pytest.MonkeyPatch) -> None:
    sd_module = install_sounddevice_module(monkeypatch)

    input_stream = audio_utils.SoundDeviceInputStream(
        rate=16000,
        chunk_frames=320,
        device_index=2,
    )
    assert input_stream.active is False
    assert input_stream.start() is True
    assert input_stream.active is True

    input_stream._audio_queue = audio_utils.queue.Queue(maxsize=1)
    input_stream._stream.callback(np.array([0.5], dtype=np.float32), 1, None, None)
    input_stream._stream.callback(np.array([1], dtype=np.int16), 1, None, None)
    assert input_stream.read(320) == np.array([1], dtype=np.int16).tobytes()
    assert input_stream.read(320) == b""

    input_stream.stop()
    assert input_stream.active is False
    input_stream.close()
    assert input_stream._stream is None

    output_stream = audio_utils.SoundDeviceOutputStream(
        rate=16000,
        chunk_frames=320,
        device_index=3,
    )
    assert output_stream.start() is True
    output_stream.write(np.array([1, -1], dtype=np.int16))
    output_stream.write(np.array([0.0, 1.0], dtype=np.float32).tobytes())
    assert len(output_stream._stream.written) == 2
    assert output_stream._stream.written[0].dtype == np.float32
    assert output_stream._stream.written[1].dtype == np.float32
    output_stream.stop()
    output_stream.close()
    assert output_stream._stream is None
    assert sd_module.wait_called is False


def test_sounddevice_stream_start_failure_and_write_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingInputStream:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("cannot start")

    class FailingOutputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def write(self, data: Any) -> None:
            raise RuntimeError("cannot write")

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    sd_module = install_sounddevice_module(monkeypatch)
    sd_module.InputStream = FailingInputStream
    assert audio_utils.SoundDeviceInputStream(rate=16000, chunk_frames=320).start() is False

    sd_module.OutputStream = FailingOutputStream
    output_stream = audio_utils.SoundDeviceOutputStream(rate=16000, chunk_frames=320)
    output_stream._stream = FailingOutputStream()
    output_stream._active = True
    output_stream.write(np.array([0.0, 1.0], dtype=np.float32))
    output_stream._active = False
    output_stream.write(np.array([0.0, 1.0], dtype=np.float32))


def test_pyaudio_streams_cover_start_read_write_stop_close(monkeypatch: pytest.MonkeyPatch) -> None:
    pa_module = install_pyaudio_module(monkeypatch, read_value=b"\x01\x00\x02\x00")

    input_stream = audio_utils.PyAudioInputStream(rate=16000, chunk_frames=320, device_index=1)
    assert input_stream.read(320) == b""
    assert input_stream.start() is True
    pa_instance = input_stream._pa
    assert input_stream.read(320) == b"\x01\x00\x02\x00"
    input_stream.stop()
    input_stream.close()
    assert pa_instance.terminated is True

    output_stream = audio_utils.PyAudioOutputStream(rate=16000, chunk_frames=320, device_index=1)
    assert output_stream.start() is True
    pa_out_instance = output_stream._pa
    output_stream.write(np.array([1, 2], dtype=np.int16))
    output_stream.write(b"\x03\x00\x04\x00")
    assert output_stream._stream.write_calls[0] == np.array([1, 2], dtype=np.int16).tobytes()
    assert output_stream._stream.write_calls[1] == b"\x03\x00\x04\x00"
    output_stream.stop()
    output_stream.close()
    assert pa_out_instance.terminated is True
    assert pa_module.paInt16 == 8

    convert_stream = audio_utils.PyAudioOutputStream(rate=16000, chunk_frames=320, device_index=1)
    assert convert_stream.start() is True
    convert_stream.write(np.array([1.0, -1.0], dtype=np.float32))
    assert convert_stream._stream.write_calls[-1] == np.array([1, -1], dtype=np.int16).tobytes()


def test_pyaudio_stream_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    pa_module = install_pyaudio_module(monkeypatch)

    input_stream = audio_utils.PyAudioInputStream(rate=16000, chunk_frames=320)
    input_stream._active = False
    assert input_stream.read(320) == b""

    input_stream._active = True
    input_stream._stream = SimpleNamespace(
        read=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        stop_stream=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert input_stream.read(320) == b""
    input_stream.stop()

    output_stream = audio_utils.PyAudioOutputStream(rate=16000, chunk_frames=320)
    output_stream._active = False
    output_stream.write(b"")
    output_stream._active = True
    output_stream._stream = SimpleNamespace(
        write=lambda data: (_ for _ in ()).throw(RuntimeError("boom")),
        stop_stream=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    output_stream.write(np.array([0.0, 1.0], dtype=np.float32))
    output_stream.stop()
    assert pa_module.paInt16 == 8


def test_base_stream_close_and_pyaudio_read_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyInput(audio_utils.AudioInputStream):
        def start(self) -> bool:
            return True

        def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes | None:
            return b""

        def stop(self) -> None:
            super().stop()

        def close(self) -> None:
            super().close()

    class DummyOutput(audio_utils.AudioOutputStream):
        def start(self) -> bool:
            return True

        def write(self, data: bytes | np.ndarray) -> None:
            return None

        def stop(self) -> None:
            super().stop()

        def close(self) -> None:
            super().close()

    class RaisingCloser:
        def close(self) -> None:
            raise RuntimeError("boom")

    input_stream = DummyInput(rate=16000, chunk_frames=320)
    input_stream._audio_queue.put(b"one")
    input_stream._audio_queue.put(b"two")
    input_stream._stream = RaisingCloser()
    input_stream.stop()
    assert input_stream._audio_queue.empty() is True
    input_stream.close()

    output_stream = DummyOutput(rate=16000, chunk_frames=320)
    output_stream._stream = RaisingCloser()
    output_stream.stop()
    output_stream.close()

    install_pyaudio_module(monkeypatch)
    input_stream = audio_utils.PyAudioInputStream(rate=16000, chunk_frames=320)
    assert input_stream.start() is True
    input_stream._stream.read = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    assert input_stream.read(320) == b""


def test_create_stream_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_utils, "get_audio_backend", lambda config=None: "sounddevice")
    assert isinstance(audio_utils.create_input_stream(16000, 320), audio_utils.SoundDeviceInputStream)
    assert isinstance(audio_utils.create_output_stream(16000, 320), audio_utils.SoundDeviceOutputStream)

    monkeypatch.setattr(audio_utils, "get_audio_backend", lambda config=None: "pyaudio")
    assert isinstance(audio_utils.create_input_stream(16000, 320), audio_utils.PyAudioInputStream)
    assert isinstance(audio_utils.create_output_stream(16000, 320), audio_utils.PyAudioOutputStream)


def test_audio_player_system_player_and_file_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice"))
    monkeypatch.setattr(audio_utils.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert player.play_with_system_player("audio.wav") is True

    def failing_run(*args: Any, **kwargs: Any) -> Never:
        raise audio_utils.subprocess.SubprocessError("failed")

    monkeypatch.setattr(audio_utils.subprocess, "run", failing_run)
    assert player.play_with_system_player("audio.wav") is False

    missing = player.play_file("/path/that/does/not/exist.wav")
    assert missing is False

    dummy_sf = ModuleType("soundfile")
    dummy_sf.read = lambda file_path, dtype: (np.array([0.0, 1.0], dtype=np.float32), 16000)
    monkeypatch.setitem(sys.modules, "soundfile", dummy_sf)
    monkeypatch.setattr(player, "play_data", lambda audio_data, sample_rate, block=True: True)
    wav_file = tmp_path / "audio.wav"
    wav_file.write_bytes(b"not a real wav, dummy soundfile ignores content")
    assert player.play_file(wav_file) is True


def test_audio_player_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    install_sounddevice_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice"))

    monkeypatch.setattr(audio_utils.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert player.play_with_system_player("audio.wav") is False

    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"not wav")
    dummy_sf = ModuleType("soundfile")
    dummy_sf.read = lambda file_path, dtype: (_ for _ in ()).throw(ValueError("bad file"))
    monkeypatch.setitem(sys.modules, "soundfile", dummy_sf)
    assert player.play_file(bad_wav) is False

    monkeypatch.setattr(audio_utils, "validate_and_clean_audio", lambda audio_data: audio_data)
    monkeypatch.setattr(
        sys.modules["sounddevice"],
        "play",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert player._play_data_sounddevice(np.array([0.0, 1.0], dtype=np.float32), 16000, block=False) is False

    install_pyaudio_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="pyaudio"))
    monkeypatch.setattr(audio_utils, "convert_to_int16", lambda audio_data: np.array([1, -1], dtype=np.int16))
    monkeypatch.setattr(
        player,
        "_pa",
        SimpleNamespace(
            open=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    )
    assert player._play_data_pyaudio(np.array([0.0, 1.0], dtype=np.float32), 16000, block=True) is False

    bad_player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice"))
    bad_player._stream = SimpleNamespace(close=lambda: None)
    bad_player._pa = SimpleNamespace(terminate=lambda: None)
    bad_player._owns_pa = True
    bad_player.close()


def test_audio_player_play_data_and_wav_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sounddevice_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice", output_device_index=7))
    audio = np.array([0.0, 1.0], dtype=np.float32)
    assert player.play_data(audio, 16000, block=True) is True
    assert audio_utils.__dict__["validate_and_clean_audio"] is not None
    assert sys.modules["sounddevice"].play_calls[0]["device"] == 7
    assert sys.modules["sounddevice"].wait_called is True

    install_pyaudio_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="pyaudio", output_device_index=3))
    assert player.play_data(np.array([0, 1, 2, 3], dtype=np.float32), 16000, block=False) is True
    assert player._pa.open_calls[0]["output_device_index"] == 3
    assert player._pa.stream.write_calls

    def make_wav(sample_width: int, channels: int = 1) -> bytes:
        import wave
        from io import BytesIO

        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(16000)
            if sample_width == 2:
                frames = np.array([1, -1], dtype=np.int16).tobytes()
            elif sample_width == 4:
                frames = np.array([1, -1], dtype=np.int32).tobytes()
            else:
                frames = b"\x00\x01"
            wf.writeframes(frames)
        return buf.getvalue()

    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice"))
    seen: list[tuple[np.ndarray, int, bool]] = []

    def record_play_data(audio_data: np.ndarray, sample_rate: int, block: bool = True) -> bool:
        seen.append((audio_data, sample_rate, block))
        return True

    monkeypatch.setattr(player, "play_data", record_play_data)
    assert player.play_wav_bytes(make_wav(2), block=False) is True
    assert player.play_wav_bytes(make_wav(4), block=True) is True
    assert player.play_wav_bytes(make_wav(1)) is False
    assert seen[0][1] == 16000
    assert seen[0][2] is False
    assert np.allclose(
        seen[0][0],
        np.array([1 / 32768.0, -1 / 32768.0], dtype=np.float32),
    )
    assert seen[1][1] == 16000
    assert seen[1][2] is True
    assert np.allclose(
        seen[1][0],
        np.array([1 / 2147483648.0, -1 / 2147483648.0], dtype=np.float32),
    )

    player._stream = SimpleNamespace(close=lambda: None)
    player._pa = SimpleNamespace(terminate=lambda: None)
    player._owns_pa = True
    player.close()


def test_audio_player_play_wav_bytes_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sounddevice_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice"))

    assert player.play_wav_bytes(b"not a wav") is False

    import wave
    from io import BytesIO

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(16000)
        wf.writeframes(b"\x00")
    assert player.play_wav_bytes(buf.getvalue()) is False


def test_audio_recorder_start_read_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyInputStream:
        def __init__(self) -> None:
            self.active = False
            self.stopped = False
            self.closed = False

        def start(self) -> bool:
            self.active = True
            return True

        def stop(self) -> None:
            self.stopped = True
            self.active = False

        def read(self, num_frames: int) -> bytes:
            return b"\x01\x00\x02\x00"

        def close(self) -> None:
            self.closed = True

    dummy_stream = DummyInputStream()
    monkeypatch.setattr(audio_utils, "create_input_stream", lambda **kwargs: dummy_stream)
    recorder = audio_utils.AudioRecorder(config=make_config(backend="sounddevice"))

    assert recorder.start() is True
    assert recorder.is_recording is True
    assert recorder.read() == b"\x01\x00\x02\x00"
    assert np.array_equal(recorder.read_numpy(), np.array([1, 2], dtype=np.int16))
    assert np.allclose(recorder.read_float(), np.array([1, 2], dtype=np.float32) / 32768.0)
    assert recorder.sample_rate == 16000

    recorder.stop()
    assert recorder.is_recording is False
    recorder.close()
    assert dummy_stream.closed is True

    failing_stream = DummyInputStream()
    failing_stream.start = lambda: False
    monkeypatch.setattr(audio_utils, "create_input_stream", lambda **kwargs: failing_stream)
    recorder = audio_utils.AudioRecorder(config=make_config(backend="sounddevice"))
    assert recorder.start() is False
    assert recorder.read() is None
    assert recorder.read_numpy() is None
    assert recorder.read_float() is None


def test_audio_recorder_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingStream:
        active = True

        def start(self) -> bool:
            raise RuntimeError("boom")

    monkeypatch.setattr(audio_utils, "create_input_stream", lambda **kwargs: ExplodingStream())
    recorder = audio_utils.AudioRecorder()
    assert recorder.start() is False


def test_legacy_wrappers_cover_module_helpers() -> None:
    assert np.array_equal(
        audio_utils.AudioUtils.validate_and_clean_audio(np.array([1.0], dtype=np.float32)),
        audio_utils.validate_and_clean_audio(np.array([1.0], dtype=np.float32)),
    )
    assert np.array_equal(
        audio_utils.AudioUtils.convert_to_int16(np.array([1.0], dtype=np.float32)),
        audio_utils.convert_to_int16(np.array([1.0], dtype=np.float32)),
    )


def test_additional_backend_and_device_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    install_pyaudio_module(monkeypatch)
    original_import = builtins.__import__

    def import_missing_sounddevice(name: str, *args: Any, **kwargs: Any):
        if name == "sounddevice":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing_sounddevice)
    assert audio_utils.get_audio_backend(make_config(backend="sounddevice")) == "pyaudio"
    assert audio_utils.is_backend_available("sounddevice") is False

    failing_sd = install_sounddevice_module(monkeypatch)
    failing_sd.query_devices = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert audio_utils.list_audio_devices("sounddevice") == []

    failing_pa = ModuleType("pyaudio")

    class BadPyAudio:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    failing_pa.PyAudio = BadPyAudio
    monkeypatch.setitem(sys.modules, "pyaudio", failing_pa)
    assert audio_utils.list_audio_devices("pyaudio") == []

    monkeypatch.setattr(
        audio_utils,
        "list_audio_devices",
        lambda backend: [audio_utils.AudioDeviceInfo(index=0, name="speaker", max_output_channels=0)],
    )
    assert audio_utils.get_default_input_device("sounddevice") is None
    assert audio_utils.get_default_output_device("sounddevice") is None
    assert audio_utils.resolve_device_index(0, is_input=False, backend="sounddevice") is None

    cleaned = audio_utils.validate_and_clean_audio(np.array([0.1, -0.2], dtype=np.float32))
    assert np.array_equal(cleaned, np.array([0.1, -0.2], dtype=np.float32))


def test_additional_resample_stream_and_factory_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    original_import = builtins.__import__

    def import_missing_scipy(name: str, *args: Any, **kwargs: Any):
        if name == "scipy.signal":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing_scipy)
    fallback = audio_utils.resample_audio(audio, 4, 2)
    assert fallback.dtype == np.float32
    assert len(fallback) == 2

    class FailingInputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def start(self) -> bool:
            return False

    class FailingOutputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(audio_utils, "SoundDeviceInputStream", FailingInputStream)
    assert audio_utils.open_input_stream_with_fallback(rate=16000, chunk_ms=20, backend="sounddevice") is None

    sd_module = install_sounddevice_module(monkeypatch)
    sd_module.InputStream = FailingInputStream
    assert audio_utils.SoundDeviceInputStream(rate=16000, chunk_frames=320).start() is False

    class ExplodingOutputStream:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("cannot open")

    sd_module.OutputStream = ExplodingOutputStream
    assert audio_utils.SoundDeviceOutputStream(rate=16000, chunk_frames=320).start() is False

    pa_module = install_pyaudio_module(monkeypatch)
    pa_module.PyAudio = type(
        "BadPyAudio",
        (),
        {
            "__init__": lambda self: None,
            "open": lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("open failed")),
            "terminate": lambda self: None,
        },
    )
    input_stream = audio_utils.PyAudioInputStream(rate=16000, chunk_frames=320)
    assert input_stream.start() is False
    assert input_stream.read(320) == b""
    output_stream = audio_utils.PyAudioOutputStream(rate=16000, chunk_frames=320)
    assert output_stream.start() is False
    output_stream.write(b"no-op")

    monkeypatch.setattr(
        audio_utils, "create_input_stream", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    recorder = audio_utils.AudioRecorder()
    assert recorder.start() is False
    assert recorder.sample_rate == 16000


def test_additional_audio_player_and_recorder_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    install_sounddevice_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="sounddevice", output_device_index=2))

    original_import = builtins.__import__

    def import_missing_soundfile(name: str, *args: Any, **kwargs: Any):
        if name == "soundfile":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing_soundfile)
    monkeypatch.setattr(player, "play_with_system_player", lambda file_path: True)
    assert player.play_file(tmp_path / "missing.wav") is True
    monkeypatch.setattr(builtins, "__import__", original_import)

    assert player.play_data(np.array([0.0, 1.0], dtype=np.float32), 16000, block=False) is True
    assert sys.modules["sounddevice"].play_calls[-1]["device"] == 2

    install_pyaudio_module(monkeypatch)
    player = audio_utils.AudioPlayer(config=make_config(backend="pyaudio"))
    assert player.play_data(np.array([0.0, 1.0], dtype=np.float32), 16000, block=True) is True

    wav_bytes = io.BytesIO()
    import wave

    with wave.open(wav_bytes, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.array([1, -1, 2, -2], dtype=np.int16).tobytes())

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        player,
        "play_data",
        lambda audio_data, sample_rate, block=True: (
            seen.update({"shape": audio_data.shape, "sample_rate": sample_rate}) or True
        ),
    )
    assert player.play_wav_bytes(wav_bytes.getvalue()) is True
    assert seen["shape"] == (2, 2)
    assert seen["sample_rate"] == 16000

    class DummyInputStream:
        def __init__(self) -> None:
            self.active = True
            self.closed = False

        def start(self) -> bool:
            return True

        def stop(self) -> None:
            self.active = False

        def read(self, num_frames: int) -> bytes:
            return b"\x01\x00"

        def close(self) -> None:
            self.closed = True

    dummy_stream = DummyInputStream()
    monkeypatch.setattr(audio_utils, "create_input_stream", lambda **kwargs: dummy_stream)
    recorder = audio_utils.AudioRecorder()
    assert recorder.start() is True
    assert recorder.read() == b"\x01\x00"
    recorder.close()
    assert dummy_stream.closed is True

    monkeypatch.setattr(
        audio_utils, "create_input_stream", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    recorder = audio_utils.AudioRecorder()
    assert recorder.start() is False
