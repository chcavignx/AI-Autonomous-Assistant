"""src/audio/audio_utils.py.
=========================

Provides:
- Audio device management (input/output)
- Streaming classes for sounddevice
- Playback utilities
- Audio format conversion and validation
- Unified audio processing utilities using sounddevice as the exclusive audio backend.

All audio device operations should use this module exclusively.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import queue
import subprocess
import sys
import threading
from typing_extensions import override
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import gcd
from typing import Protocol, final, runtime_checkable, cast

import numpy as np
from numpy.typing import NDArray

from src.utils.config import Config

module_name = __name__
lib_name = module_name.split('.')[1]
logger = logging.getLogger(lib_name)

# =============================================================================
# Type Definitions
# =============================================================================

@runtime_checkable
class AudioStreamProtocol(Protocol):
    """Protocol for audio stream objects (read)."""

    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes | NDArray[np.generic]: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioOutputStreamProtocol(Protocol):
    """Protocol for audio output stream objects (write)."""

    def write(self, data: bytes | NDArray[np.generic]) -> None: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class OpenedInputStream:
    """Result of opening an input stream with backend and rate fallback."""

    stream: "AudioInputStream"
    backend: str
    capture_rate: int
    native_chunk_frames: int
    device_index: int | None
    need_resample: bool
    resample_up: int = 1
    resample_down: int = 1


# =============================================================================
# Backend Selection
# =============================================================================

def get_audio_backend() -> str:
    try:
        import sounddevice
        _ = sounddevice
    except ImportError as e:
        raise ImportError("The 'sounddevice' library is required") from e
    return "sounddevice"

def is_backend_available(backend: str) -> bool:
    """Check if a specific audio backend is available.

    Args:
        backend: "sounddevice" (others are unsupported)

    Returns:
        bool: True if backend is installed and usable
    """
    if backend == "sounddevice":
        return True
    return False


def get_chunk_frames(sample_rate: int, chunk_ms: int) -> int:
    """Convert a chunk duration in milliseconds to frames."""
    return max(1, int(round(sample_rate * chunk_ms / 1000)))


def _candidate_device_indexes(
    *,
    device_index: int | None,
    is_input: bool,
) -> list[int | None]:
    """Build an ordered list of device candidates for fallback probing."""
    candidates: list[int | None] = []
    if device_index is not None:
        resolved = resolve_device_index(
            device_index=device_index,
            is_input=is_input
        )
        if resolved is not None:
            candidates.append(resolved)
        else:
            logger.warning(
                "Configured %s device %s is unavailable or incompatible; using default",
                "input" if is_input else "output",
                device_index,
            )
    if None not in candidates:
        candidates.append(None)
    return candidates


def open_input_stream_with_fallback(
    *,
    rate: int,
    chunk_ms: int,
    device_index: int | None = None,
    candidate_rates: list[int] | None = None,
    dtype: str = "float32",
) -> OpenedInputStream | None:
    """Open an input stream and probe devices/rates until one works."""

    if candidate_rates is None:
        candidate_rates = [rate, 44100, 48000, 22050, 16000, 8000]

    device_candidates = _candidate_device_indexes(
        device_index=device_index,
        is_input=True,
    )

    for dev in device_candidates:
        for capture_rate in candidate_rates:
            chunk_frames = get_chunk_frames(capture_rate, chunk_ms)
            stream = SoundDeviceInputStream(
                rate=capture_rate,
                chunk_frames=chunk_frames,
                device_index=dev,
                dtype=dtype,
            )
            if stream.start():
                return _opened_input_stream(
                    stream=stream,
                    backend="sounddevice",
                    capture_rate=capture_rate,
                    native_chunk_frames=chunk_frames,
                    device_index=dev,
                    requested_rate=rate,
                )
    return None


def _opened_input_stream(
    *,
    stream: "AudioInputStream",
    backend: str,
    capture_rate: int,
    native_chunk_frames: int,
    device_index: int | None,
    requested_rate: int,
) -> OpenedInputStream:
    """Build the structured input-stream result and resampling metadata."""
    need_resample = capture_rate != requested_rate
    resample_up = 1
    resample_down = 1
    if need_resample:
        g = gcd(requested_rate, capture_rate)
        resample_up = requested_rate // g
        resample_down = capture_rate // g
    return OpenedInputStream(
        stream=stream,
        backend=backend,
        capture_rate=capture_rate,
        native_chunk_frames=native_chunk_frames,
        device_index=device_index,
        need_resample=need_resample,
        resample_up=resample_up,
        resample_down=resample_down,
    )


# =============================================================================
# Device Management
# =============================================================================

@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""

    index: int | None
    name: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: float = 0.0
    is_input: bool = False
    is_output: bool = False

    def can_capture(self) -> bool:
        """Check if device can capture audio."""
        return self.max_input_channels > 0

    def can_playback(self) -> bool:
        """Check if device can playback audio."""
        return self.max_output_channels > 0

def _get_sounddevice_default_device(_kind: str | None = None) -> AudioDeviceInfo | None:
    """Get the default sounddevice device."""
    try:
        import sounddevice as sd
    except ImportError:
        return None

    try:
        info: dict[str, object] = sd.query_devices(device=None, kind=_kind if _kind is not None else "input")  # pyright: ignore[reportAny, reportUnknownMemberType]
        return AudioDeviceInfo(
            index=sd.default.device[0] if (_kind or "input") == "input" else sd.default.device[1],  # pyright: ignore[reportAny]
            name=str(cast(str, info.get("name", ""))),
            max_input_channels=int(cast(int, info.get("max_input_channels", 0))),
            max_output_channels=int(cast(int, info.get("max_output_channels", 0))),
            default_samplerate=float(cast(float, info.get("default_samplerate", 0.0))),
            is_input=int(cast(int, info.get("max_input_channels", 0))) > 0,
            is_output=int(cast(int, info.get("max_output_channels", 0))) > 0,
        )
    except sd.PortAudioError:
        return None
    except Exception:
        return None


def list_audio_devices(backend: str | None = None) -> list[AudioDeviceInfo]:
    """List all available audio devices for a given backend.

    Args:
        backend: Audio backend to use ("sounddevice", "pyaudio", or None for auto)

    Returns:
        list[AudioDeviceInfo]: List of device information
    """
    if backend is None:
        backend = get_audio_backend()

    devices: list[AudioDeviceInfo] = []

    if backend == "sounddevice":
        try:
            import sounddevice as sd
            # sd_devices is a DeviceList
            sd_devices = cast(list[dict[str, object]], sd.query_devices())  # pyright: ignore[reportUnknownMemberType]

            for i, dev in enumerate(sd_devices):
                devices.append(AudioDeviceInfo(
                    index=i,
                    name=str(dev.get("name", f"Device {i}")),
                    max_input_channels=int(cast(int, dev.get("max_input_channels", 0))),
                    max_output_channels=int(cast(int, dev.get("max_output_channels", 0))),
                    default_samplerate=float(cast(float, dev.get("default_samplerate", 0.0))),
                    is_input=int(cast(int, dev.get("max_input_channels", 0))) > 0,
                    is_output=int(cast(int, dev.get("max_output_channels", 0))) > 0,
                ))
        except Exception as e:
            logger.warning("Failed to list sounddevice devices: %s", e)

    return devices

def get_default_input_device() -> int | None:
    """Get the default input device.

    Returns:
        AudioDeviceInfo or None if no input device available
    """
    dev = _get_sounddevice_default_device("input")
    if dev is not None:
        logger.info("Found input device: %s", dev.name)
        return dev.index
    return None


def get_default_output_device() -> int | None:
    """Get the default output device.

    Returns:
        AudioDeviceInfo or None if no output device available
    """

    dev: AudioDeviceInfo | None= _get_sounddevice_default_device("output")
    if dev is not None:
        logger.info("Found output device: %s", dev.name)
        return dev.index
    return None


def resolve_device_index(
    device_index: int | None,
    is_input: bool
) -> int | None:
    """Resolve a device index, validating it exists and has the right capability.

    Args:
        device_index: Requested device index (None for default)
        is_input: True if need input capability, False for output

    Returns:
        int | None: Validated device index or None for default
    """
    _default = False
    if device_index is None:
        _default = True

    devices = list_audio_devices()

    # Check if index is valid
    if device_index is not None and (device_index < 0 or device_index >= len(devices)):
        logger.warning(
            "Device index %d out of range (0-%d), using default",
            device_index, len(devices) - 1
        )
        _default = True

    if _default:
        if is_input:
            return get_default_input_device()
        else:
            return get_default_output_device()

    dev: AudioDeviceInfo = devices[cast(int, device_index)]

    # Check capability
    if is_input and not dev.can_capture():
        logger.warning(
            "Device %d (%s) has no input channels, using default",
            device_index, dev.name
        )
        return None

    if not is_input and not dev.can_playback():
        logger.warning(
            "Device %d (%s) has no output channels, using default",
            device_index, dev.name
        )
        return None
    logger.info(
        "Resolved device index: %d (%s)",
        device_index, dev.name
    )
    return device_index


# =============================================================================
# Error Handling Utilities
# =============================================================================

def install_alsa_error_handler() -> object:
    """Suppress ALSA's noisy enumeration warnings globally.

    This installs a callback into ALSA that discards all error messages.
    Reference: https://github.com/openai/whisper.cpp/blob/c227e4862554a73c6a271c46f6487664df1d8fc0/whisper.cpp#L3673

    Returns:
        Handler reference (must be kept to prevent garbage collection)
    """
    try:
        import ctypes

        # Define the callback function signature for ALSA error handling
        # typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...);
        alsa_cb_type = ctypes.CFUNCTYPE(
            None,                # Return type: void
            ctypes.c_char_p,     # const char *file
            ctypes.c_int,        # int line
            ctypes.c_char_p,     # const char *function
            ctypes.c_int,        # int err
            ctypes.c_char_p      # const char *fmt (start of variable arguments)
        )

        # Create a callback that does nothing (silences all errors)
        def null_error_handler(*_args: object, **_kwargs: object) -> None:
            pass

        handler = alsa_cb_type(null_error_handler)

        # Load the ALSA library
        alsa = ctypes.cdll.LoadLibrary("libasound.so.2")

        # Register the error handler
        alsa.snd_lib_error_set_handler(handler)

        logger.debug("✅ ALSA error handler installed successfully")
        return handler

    except Exception as e:
        # Don't fail the whole program if this fails
        logger.warning(f"⚠️ ALSA error handler installation failed: {e}")
        return None

# =============================================================================
# Audio Format Utilities
# =============================================================================

def validate_and_clean_audio(
    audio_data: NDArray[np.generic],
    silence_threshold: float = 1e-6,
    noise_floor: float = 1e-8,
) -> NDArray[np.float32]:
    """Enhanced validation and cleaning of audio data.

    Prevents numerical issues in whisper and other audio processing:
    - Removes NaN and Inf values
    - Clips extreme values
    - Normalizes audio to prevent clipping in whisper
    - Adds dithering for near-silent audio

    Args:
        audio_data: Input audio array
        silence_threshold: RMS threshold below which to add dithering
        noise_floor: Amount of noise to add for dithering

    Returns:
        NDArray[np.float32]: Cleaned audio array
    """
    audio_array = np.asarray(audio_data, dtype=np.float32)

    if audio_array.size == 0:
        return np.array([], dtype=np.float32)

    has_invalid = False

    # Remove NaN values
    if np.any(np.isnan(audio_array)):
        audio_array = np.nan_to_num(audio_array, nan=0.0)
        has_invalid = True

    # Remove Inf values
    if np.any(np.isinf(audio_array)):
        audio_array = np.nan_to_num(audio_array, posinf=1.0, neginf=-1.0)
        has_invalid = True

    # Clip extreme values
    extreme_threshold = 32767.0  # Anything above int16 max is likely invalid
    if np.any(np.abs(audio_array) > extreme_threshold):
        audio_array = np.clip(audio_array, -1.0, 1.0)
        has_invalid = True

    # Normalize to prevent whisper clipping
    abs_audio = np.abs(audio_array)
    max_val = float(cast(float, abs_audio.max()))
    if max_val > 0 and max_val > 0.95:
        audio_array *= 0.95 / max_val
        has_invalid = True

    # Add dithering for near-silent audio
    rms_energy = math.sqrt(float(np.mean(audio_array**2)))
    if rms_energy < silence_threshold:
        dither = np.random.normal(0.0, noise_floor, audio_array.shape).astype(np.float32)
        audio_array += dither
        has_invalid = True

    # Final cleanup
    audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.95, neginf=-0.95)
    audio_array = audio_array.astype(np.float32)

    if has_invalid:
        # Verify the cleanup worked
        _ = np.sqrt(np.mean(audio_array**2))  # pyright: ignore[reportAny]

    return audio_array


def convert_to_int16(audio_float: NDArray[np.generic]) -> NDArray[np.int16]:
    """Convert float32 audio to int16 with validation.

    Args:
        audio_float: Float32 audio array (normalized to [-1, 1])

    Returns:
        NDArray[np.int16]: Int16 audio array
    """
    clean_audio = validate_and_clean_audio(audio_float)
    return (clean_audio * 32767).astype(np.int16)


def convert_to_float32(audio_int: NDArray[np.generic]) -> NDArray[np.float32]:
    """Convert int16 audio to float32.

    Args:
        audio_int: Int16 audio array

    Returns:
        NDArray[np.float32]: Float32 audio array (normalized to [-1, 1])
    """
    return np.asarray(audio_int, dtype=np.int16).astype(np.float32) / 32768.0


def resample_audio(
    audio: NDArray[np.float32],
    original_rate: int,
    target_rate: int,
) -> NDArray[np.float32]:
    """Resample audio to a different sample rate.

    Uses scipy.signal.resample_poly for high-quality resampling.

    Args:
        audio: Audio array to resample
        original_rate: Current sample rate
        target_rate: Desired sample rate

    Returns:
        NDArray[np.float32]: Resampled audio
    """
    if original_rate == target_rate:
        return audio

    try:
        from scipy.signal import resample_poly  # pyright: ignore[reportUnknownVariableType]

        g = gcd(original_rate, target_rate)
        up = target_rate // g
        down = original_rate // g

        _audio: NDArray[np.float32] = resample_poly(audio, up, down).astype(np.float32)  # pyright: ignore[reportAny]
        return _audio
    except ImportError:
        logger.warning("scipy not available, using simple resampling")
        # Simple resampling as fallback
        ratio = target_rate / original_rate
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio), new_length, endpoint=False),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)


# =============================================================================
# Base Stream Classes
# =============================================================================

class AudioStream(ABC):
    """Abstract base class for audio streams."""

    rate: int
    chunk_frames: int
    device_index: int | None
    is_input: bool
    _stream: object | None
    _active: bool

    def __init__(
        self,
        rate: int,
        chunk_frames: int,
        device_index: int | None = None,
        is_input: bool = True,
    ):
        """Initialize audio stream.

        Args:
            rate: Sample rate in Hz
            chunk_frames: Number of frames per chunk
            device_index: Device index or None for default
            is_input: True for input stream, False for output
        """
        self.rate = rate
        self.chunk_frames = chunk_frames
        self.device_index = device_index
        self.is_input = is_input
        self._stream = None
        self._active = False

    @abstractmethod
    def start(self) -> bool:
        """Start the stream.

        Returns:
            bool: True if successful
        """
        return False

    @abstractmethod
    def close(self) -> None:
        """Close the stream and release resources."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the stream."""
        pass

    @property
    def active(self) -> bool:
        """Check if stream is active."""
        return self._active


class AudioInputStream(AudioStream, ABC):
    """Abstract base class for audio input streams."""

    _active: bool
    _stream: object | None
    _audio_queue: queue.Queue[bytes | None]
    _stop_event: threading.Event

    def __init__(
        self,
        rate: int,
        chunk_frames: int,
        device_index: int | None = None,
    ):
        super().__init__(rate, chunk_frames, device_index, is_input=True)
        self._audio_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

    @abstractmethod
    @override
    def start(self) -> bool:
        """Start the stream."""
        return False

    @abstractmethod
    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes | None:
        """Read audio data from the stream.

        Args:
            num_frames: Number of frames to read
            exception_on_overflow: Whether to raise exception on overflow

        Returns:
            bytes | None: Audio data or None on timeout
        """
        pass

    @override
    def stop(self) -> None:
        """Stop the stream."""
        self._stop_event.set()
        self._active = False

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                _ = self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @override
    def close(self) -> None:
        """Close the stream and release resources."""
        self.stop()
        if self._stream is not None:
            try:
                if hasattr(self._stream, 'close'):
                    _ = getattr(self._stream, 'close')()  # pyright: ignore[reportAny]
            except Exception as e:
                logger.debug("Error closing stream: %s", e)
            self._stream = None


class AudioOutputStream(AudioStream, ABC):
    """Abstract base class for audio output streams."""

    _active: bool
    _stream: object | None

    def __init__(
        self,
        rate: int,
        chunk_frames: int,
        device_index: int | None = None,
    ):
        super().__init__(rate, chunk_frames, device_index, is_input=False)

    @abstractmethod
    @override
    def start(self) -> bool:
        """Start the stream."""
        return False

    @abstractmethod
    def write(self, data: bytes | NDArray[np.generic]) -> None:
        """Write audio data to the stream.

        Args:
            data: Audio data to write
        """
        pass

    @override
    def stop(self) -> None:
        """Stop the stream."""
        self._active = False

    @override
    def close(self) -> None:
        """Close the stream and release resources."""
        self.stop()
        if self._stream is not None:
            try:
                if hasattr(self._stream, 'close'):
                    _ = getattr(self._stream, 'close')()  # pyright: ignore[reportAny]
            except Exception as e:
                logger.debug("Error closing stream: %s", e)
            self._stream = None


# =============================================================================
# SoundDevice Implementation
# =============================================================================

@final
class SoundDeviceInputStream(AudioInputStream):
    """Audio input stream using sounddevice library."""

    dtype: str
    _callback_exception: Exception | None

    def __init__(
        self,
        rate: int,
        chunk_frames: int,
        device_index: int | None = None,
        dtype: str = "float32",
    ):
        super().__init__(rate, chunk_frames, device_index)
        self.dtype = dtype
        self._callback_exception = None

    @override
    def start(self) -> bool:
        """Start the sounddevice input stream."""
        try:
            import sounddevice as sd

            def _callback(
                indata: NDArray[np.generic],
                frames: int,
                time_info: object,
                status: object
            ) -> None:
                """Callback function for sounddevice input stream."""
                _ = frames
                _ = time_info
                if status:
                    logger.debug("SoundDevice input callback status: %s", status)

                try:
                    # Convert to int16 bytes for consistency
                    data_bytes: bytes
                    if indata.dtype == np.float32:
                        float_data = cast(NDArray[np.float32], indata)
                        int16_data: NDArray[np.int16] = (float_data * 32767).astype(np.int16)
                        data_bytes = int16_data.tobytes()
                    else:
                        data_bytes = indata.tobytes()

                    if not self._stop_event.is_set():
                        try:
                            self._audio_queue.put(data_bytes, block=False)
                        except queue.Full:
                            # Drop oldest frame to prevent blocking
                            with contextlib.suppress(queue.Empty):
                                _ = self._audio_queue.get_nowait()
                            self._audio_queue.put(data_bytes, block=False)
                except Exception as e:
                    self._callback_exception = e
                    logger.debug("SoundDevice callback error: %s", e)

            self._stream = sd.InputStream(
                samplerate=self.rate,
                channels=1,
                blocksize=self.chunk_frames,
                dtype=self.dtype,
                device=self.device_index,
                callback=_callback,
            )
            self._stream.start()
            self._active = True

        except Exception as e:
            logger.debug("Failed to start SoundDevice input stream: %s", e)
            self._active = False

        return self._active

    @override
    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes | None:
        """Read audio data from the queue."""
        _ = num_frames
        _ = exception_on_overflow
        if not self._active:
            return None
        try:
            data = self._audio_queue.get(timeout=0.1)
            # None sentinel placed by stop(); treat empty bytes as None too
            if data is None or len(data) == 0:
                return None
            return data
        except queue.Empty:
            return None

    @override
    def stop(self) -> None:
        """Stop the stream."""
        super().stop()
        if self._stream is not None:
            try:
                _ = getattr(self._stream, 'stop')()  # pyright: ignore[reportAny]
            except Exception as e:
                logger.debug("Error stopping SoundDevice stream: %s", e)

    @override
    def close(self) -> None:
        """Close the stream."""
        self.stop()
        super().close()


@final
class SoundDeviceOutputStream(AudioOutputStream):
    """Audio output stream using sounddevice library."""

    dtype: str

    def __init__(
        self,
        rate: int,
        chunk_frames: int,
        device_index: int | None = None,
        dtype: str = "int16",
    ):
        super().__init__(rate, chunk_frames, device_index)
        self.dtype = dtype


    @override
    def start(self) -> bool:
        """Start the sounddevice output stream."""
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=self.rate,
                channels=1,
                blocksize=self.chunk_frames,
                dtype=self.dtype,
                device=self.device_index,
            )

            self._stream.start()
            self._active = True
            return True

        except Exception as e:
            logger.debug("Failed to start SoundDevice output stream: %s", e)
            self._active = False
            return False

    @override
    def write(self, data: bytes | NDArray[np.generic]) -> None:
        """Write audio data to the stream."""
        if not self._active or self._stream is None:
            return

        try:
            if isinstance(data, bytes):
                # Convert bytes to numpy array based on dtype
                audio_array: NDArray[np.generic]
                if self.dtype == "float32":
                    audio_array = np.frombuffer(data, dtype=np.float32)
                else:
                    audio_array = np.frombuffer(data, dtype=np.int16)

                # Ensure correct shape
                if audio_array.ndim == 0:
                    audio_array = audio_array.reshape(-1)

                # sounddevice expects float32 in [-1, 1]
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0

                _ = getattr(self._stream, 'write')(audio_array)  # pyright: ignore[reportAny]
            else:
                # Already numpy array
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                _ = getattr(self._stream, 'write')(data)  # pyright: ignore[reportAny]
        except Exception as e:
            logger.debug("SoundDevice output write error: %s", e)


# =============================================================================
# Playback Utilities
# =============================================================================

@final
class AudioPlayer:
    """High-level audio playback utility.

    Supports playing WAV files and raw audio data using the configured backend.

    Usage:
        player = AudioPlayer(config)
        player.play_file("audio.wav")
        player.play_data(audio_data, sample_rate)
        player.close()
    """

    _config: Config | None
    _stream: AudioOutputStream | None

    def __init__(self, config: Config | None = None):
        """Initialize audio player.

        Args:
            config: Config object for backend detection
        """
        self._config = config
        self._stream = None

    def get_backend(self) -> str:
        """Get the current backend."""
        return "sounddevice"

    def play_file(self, file_path: str | pathlib.Path) -> bool:
        """Play a WAV file.

        Args:
            file_path: Path to the WAV file

        Returns:
            bool: True if successful
        """
        try:
            import soundfile as sf

            resolved_path = pathlib.Path(file_path)
            if not resolved_path.exists():
                logger.error("Audio file not found: %s", resolved_path)
                return False

            read_result = sf.read(resolved_path, dtype="float32")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            audio_data, sample_rate = read_result  # pyright: ignore[reportUnknownVariableType, reportAny]
            return self.play_data(audio_data, int(sample_rate))  # pyright: ignore[reportUnknownArgumentType, reportAny]

        except ImportError:
            # Fallback to system player
            logger.debug("soundfile not available, using system player")
            return self.play_with_system_player(file_path)
        except Exception as e:
            logger.error("Failed to play file %s: %s", file_path, e)
            return False

    def play_with_system_player(self, file_path: str | pathlib.Path) -> bool:
        """Play audio file using system player."""
        try:
            player = "aplay" if sys.platform != "darwin" else "afplay"
            result = subprocess.run(
                [player, str(file_path)],
                capture_output=True,
                check=False,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired):
            logger.debug("System player failed")
            return False

    def play_data(
        self,
        audio_data: NDArray[np.generic],
        sample_rate: int,
        block: bool = True,
    ) -> bool:
        """Play audio data.

        Args:
            audio_data: Audio data (float32 in [-1, 1])
            sample_rate: Sample rate in Hz
            block: Whether to block until playback finishes

        Returns:
            bool: True if successful
        """
        return self._play_data_sounddevice(audio_data, sample_rate, block)

    def _play_data_sounddevice(
        self,
        audio_data: NDArray[np.generic],
        sample_rate: int,
        block: bool,
    ) -> bool:
        """Play audio data using sounddevice."""
        try:
            import sounddevice as sd

            # Ensure correct dtype and range
            audio = validate_and_clean_audio(audio_data)

            device = self._config.audio.output_device_index if self._config is not None else None
            _ = sd.play(audio, samplerate=sample_rate, device=device)  # pyright: ignore[reportUnknownMemberType]
            if block:
                _ = sd.wait()  # pyright: ignore[reportUnknownVariableType]

            return True

        except Exception as e:
            logger.error("SoundDevice playback error: %s", e)
            return False

    def play_wav_bytes(self, wav_bytes: bytes, block: bool = True) -> bool:
        """Play WAV data from bytes.

        Args:
            wav_bytes: WAV file data as bytes
            block: Whether to block until playback finishes

        Returns:
            bool: True if successful
        """
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                rate = wf.getframerate()

                # Read all frames
                frames = wf.readframes(wf.getnframes())

                # Convert to numpy array
                if sample_width == 2:  # int16
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:  # int32
                    audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    logger.warning("Unsupported sample width: %d", sample_width)
                    return False

                # Reshape for multi-channel
                if channels > 1:
                    audio = audio.reshape(-1, channels)

                return self.play_data(audio, rate, block)

        except Exception as e:
            logger.error("Failed to play WAV bytes: %s", e)
            return False

    def close(self) -> None:
        """Close the player and release resources."""
        if self._stream is not None:
            _ = getattr(self._stream, 'close')()  # pyright: ignore[reportAny]
            self._stream = None

# =============================================================================
# Audio Recorder
# =============================================================================

@final
class AudioRecorder:
    """High-level audio recording utility.

    Captures audio from microphone using the configured backend.

    Usage:
        recorder = AudioRecorder(config, rate=16000, chunk_frames=1024)
        recorder.start()
        # ... capture audio ...
        data = recorder.read()
        recorder.stop()
        recorder.close()
    """

    _config: Config | None
    _backend: str
    _rate: int
    _chunk_frames: int
    _device_index: int | None
    _stream: AudioInputStream | None
    _running: bool

    def __init__(
        self,
        config: Config | None = None,
        rate: int | None = None,
        chunk_frames: int | None = None,
        device_index: int | None = None,
    ):
        """Initialize audio recorder.

        Args:
            config: Config object for backend and device settings
            rate: Sample rate (defaults to config.audio.input_sample_rate)
            chunk_frames: Frames per chunk (defaults to calculated from config)
            device_index: Device index (defaults to config.audio.input_device_index)
        """
        self._config = config
        self._backend = get_audio_backend()

        if config is not None:
            self._rate = rate or config.audio.input_sample_rate
            self._chunk_frames = chunk_frames or config.audio.input_chunk_size
            self._device_index = (
                device_index if device_index is not None else config.audio.input_device_index
            )
        else:
            self._rate = rate or 16000
            self._chunk_frames = chunk_frames or 1024
            self._device_index = device_index

        self._stream = None
        self._running = False

    def start(self) -> bool:
        """Start recording.

        Returns:
            bool: True if successful
        """
        self.stop()

        try:
            self._stream = SoundDeviceInputStream(
                rate=self._rate,
                chunk_frames=self._chunk_frames,
                device_index=self._device_index,
            )

            if not self._stream.start():
                self._stream = None
                return False

            self._running = True
            logger.info("Recording started")
            return True

        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            self._stream = None
            return False

    def stop(self) -> None:
        """Stop recording."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()

    def read(self) -> bytes | None:
        """Read a chunk of audio data.

        Returns:
            bytes | None: Audio data or None if not running or no data available
        """
        if self._stream is None or not self._running:
            return None
        return self._stream.read(self._chunk_frames)

    def read_numpy(self) -> NDArray[np.int16] | None:
        """Read a chunk of audio data as numpy array.

        Returns:
            NDArray[np.int16] | None: Audio data or None if no data available
        """
        data = self.read()
        if data is None or len(data) == 0:
            return None
        return np.frombuffer(data, dtype=np.int16)

    def read_float(self) -> NDArray[np.float32] | None:
        """Read a chunk of audio data as float32 numpy array.

        Returns:
            NDArray[np.float32] | None: Audio data (normalized to [-1, 1]) or None
        """
        data = self.read_numpy()
        if data is None:
            return None
        return data.astype(np.float32) / 32768.0

    @property
    def sample_rate(self) -> int:
        """Get the sample rate."""
        return self._rate

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._running and self._stream is not None and self._stream.active

    def close(self) -> None:
        """Close the recorder and release resources."""
        self.stop()
        if self._stream is not None:
            self._stream.close()
            self._stream = None


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Keep SoundDeviceAudioStream for backward compatibility
SoundDeviceAudioStream = SoundDeviceInputStream

def create_input_stream(
    rate: int,
    chunk_frames: int,
    device_index: int | None = None,
) -> SoundDeviceInputStream:
    """Factory to create a SoundDeviceInputStream."""
    return SoundDeviceInputStream(
        rate=rate,
        chunk_frames=chunk_frames,
        device_index=device_index,
    )

def create_output_stream(
    rate: int,
    chunk_frames: int,
    device_index: int | None = None,
) -> SoundDeviceOutputStream:
    """Factory to create a SoundDeviceOutputStream."""
    return SoundDeviceOutputStream(
        rate=rate,
        chunk_frames=chunk_frames,
        device_index=device_index,
    )


# Keep existing utility functions for backward compatibility
class AudioUtils:
    """Legacy utility class for audio processing (kept for backward compatibility)."""

    @staticmethod
    def play_audio_file(file_path: str) -> None:
        """Play audio file using system audio player (legacy)."""
        player = AudioPlayer()
        _ = player.play_with_system_player(file_path)
        player.close()

    @staticmethod
    def play_audio_file_to_sd(file_path: str) -> None:
        """Play audio file using SoundDevice (legacy)."""
        player = AudioPlayer()
        _ = setattr(player, "_backend", "sounddevice")
        _ = player.play_file(file_path)
        player.close()

    @staticmethod
    def play_audio_stream(
        audio_stream: NDArray[np.generic],
        sample_rate: int,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        """Play audio stream (legacy)."""
        _ = channels
        player = AudioPlayer()

        # Convert to float32
        if dtype == "int16":
            audio = audio_stream.astype(np.float32) / 32768.0
        else:
            audio = audio_stream.astype(np.float32)

        _ = player.play_data(audio, sample_rate, block=True)
        player.close()

    @staticmethod
    def validate_and_clean_audio(audio_data: NDArray[np.generic]) -> NDArray[np.float32]:
        """Validate and clean audio data (legacy)."""
        return validate_and_clean_audio(audio_data)

    @staticmethod
    def convert_to_int16(audio_float: NDArray[np.generic]) -> NDArray[np.int16]:
        """Convert float32 audio to int16 (legacy)."""
        return convert_to_int16(audio_float)


# Import pathlib for type hints
import pathlib
