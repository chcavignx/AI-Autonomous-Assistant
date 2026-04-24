"""src/audio/audio_utils.py.
=========================
Shared audio processing utilities (validation, conversion, playback).

Extracted from voice_agent_offline.py for reuse across STT, TTS, and VAD modules.
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import sys
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import soundfile as sf


class _SoundDeviceStream(Protocol):
    def start(self) -> None: ...

    def write(self, data: NDArray[np.generic]) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _SoundFileModule(Protocol):
    def read(
        self, file: str, *, dtype: str = ...
    ) -> tuple[NDArray[np.float32], int]: ...


class _SoundDeviceModule(Protocol):
    class PortAudioError(Exception): ...

    def play(self, data: NDArray[np.float32], *, samplerate: int) -> None: ...

    def wait(self) -> None: ...

    def stop(self) -> None: ...

    def OutputStream(
        self, *, samplerate: int, channels: int, dtype: str
    ) -> _SoundDeviceStream: ...


@contextlib.contextmanager
def suppress_pa_stderr():
    """Temporarily redirect fd 2 to /dev/null.

    Suppresses noisy PortAudio/JACK C-library messages that bypass Python's
    logging (e.g. mmap drain errors at teardown, JACK connection attempts).
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def install_alsa_error_handler():
    """Suppress ALSA's noisy enumeration warnings globally."""
    try:
        import ctypes

        alsa_cb_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
        )
        handler = alsa_cb_type(lambda *_: None)
        ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(handler)
        return handler  # Must keep reference
    except Exception:
        return None


class AudioUtils:
    """Utility functions for audio processing."""

    @staticmethod
    def play_audio_file(file_path: str) -> None:
        """Play audio file using system audio player."""
        try:
            player = "aplay" if sys.platform != "darwin" else "afplay"
            subprocess.run([player, file_path], capture_output=True, check=False)
        except (subprocess.SubprocessError, OSError):
            pass

    @staticmethod
    def play_audio_file_to_sd(file_path: str) -> None:
        """Play audio file using SoundDevice."""
        import sounddevice as sd

        sounddevice = cast(_SoundDeviceModule, sd)
        soundfile = cast(_SoundFileModule, sf)
        audio_data: NDArray[np.float32] | None = None
        sample_rate: int | None = None
        try:
            audio_data, sample_rate = soundfile.read(file_path, dtype="float32")
            sounddevice.play(audio_data, samplerate=sample_rate)
            sounddevice.wait()
        except (sf.LibsndfileError, ValueError, sounddevice.PortAudioError, Exception):
            pass
        finally:
            with contextlib.suppress(Exception):
                sounddevice.stop()
            if "audio_data" in locals() and audio_data is not None:
                del audio_data
            if "sample_rate" in locals() and sample_rate is not None:
                del sample_rate

    @staticmethod
    def play_audio_stream(
        audio_stream: NDArray[np.generic],
        sample_rate: int,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        """Play audio stream using SoundDevice."""
        import sounddevice as sd

        sounddevice = cast(_SoundDeviceModule, sd)
        stream = sounddevice.OutputStream(
            samplerate=sample_rate, channels=channels, dtype=dtype
        )
        stream.start()
        try:
            stream.write(audio_stream)
        except (sounddevice.PortAudioError, ValueError, Exception):
            pass
        finally:
            stream.stop()
            stream.close()
            del audio_stream

    @staticmethod
    def validate_and_clean_audio(
        audio_data: NDArray[np.generic],
    ) -> NDArray[np.float32]:
        """Enhanced validation and cleaning of audio data to prevent
        Whisper numerical issues.
        """
        audio_array = np.asarray(audio_data, dtype=np.float32)

        if audio_array.size == 0:
            return np.array([], dtype=np.float32)

        has_invalid = False

        if np.any(np.isnan(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0)
            has_invalid = True

        if np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, posinf=1.0, neginf=-1.0)
            has_invalid = True

        extreme_threshold = 1e10
        if np.any(np.abs(audio_array) > extreme_threshold):
            audio_array = np.clip(audio_array, -1.0, 1.0)
            has_invalid = True

        abs_audio = cast(NDArray[np.float32], np.abs(audio_array))
        max_val = cast(float, abs_audio.max())
        if max_val > 0 and max_val > 0.95:
            audio_array *= 0.95 / max_val
            has_invalid = True

        noise_floor = 1e-8
        silence_threshold = 1e-6
        rms_energy = math.sqrt(float(np.mean(audio_array**2)))
        if rms_energy < silence_threshold:
            dither = np.random.normal(0.0, noise_floor, audio_array.shape).astype(
                np.float32
            )
            audio_array += dither
            has_invalid = True

        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.95, neginf=-0.95)
        audio_array = audio_array.astype(np.float32)

        if has_invalid:
            np.sqrt(np.mean(audio_array**2))

        return audio_array

    @staticmethod
    def convert_to_int16(audio_float: NDArray[np.generic]) -> NDArray[np.int16]:
        """Convert float32 audio to int16 with validation."""
        clean_audio = AudioUtils.validate_and_clean_audio(audio_float)
        return (clean_audio * 32767).astype(np.int16)
