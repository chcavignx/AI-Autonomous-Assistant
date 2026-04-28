"""audio/wake_word.py.
==================
Wake word detection using openWakeWord.

Lightweight (~10MB), fast (<5ms/chunk), fully offline.
Runs in background with configurable cooldown to prevent false positives.

Pre-trained models:
  - hey_jarvis, hey_mycroft, alexa, hey_google, etc.

Dependencies:
  pip install openwakeword
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import queue
import sys
import threading
import time
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import suppress_pa_stderr

if TYPE_CHECKING:
    from collections.abc import Callable

    from utils.config import Config

logger = logging.getLogger(__name__)


class _PyAudioStreamLike(Protocol):
    def read(self, num_frames: int, exception_on_overflow: bool = ...) -> bytes: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...

    def abort_stream(self) -> None: ...


class _PyAudioLike(Protocol):
    def open(
        self,
        *,
        format: int,
        channels: int,
        rate: int,
        input: bool,
        frames_per_buffer: int,
        input_device_index: int | None = ...,
    ) -> _PyAudioStreamLike: ...

    def terminate(self) -> None: ...

    def get_device_info_by_index(self, index: int) -> dict[str, object]: ...


class _WakeWordModelLike(Protocol):
    prediction_buffer: dict[str, list[float]]

    def predict(self, x: NDArray[np.float32]) -> object: ...


class _ResamplePoly(Protocol):
    def __call__(
        self, x: NDArray[np.float32], up: int, down: int
    ) -> NDArray[np.float32]: ...


def _resample_poly(
    audio: NDArray[np.float32], up: int, down: int
) -> NDArray[np.float32]:
    scipy_signal = import_module("scipy.signal")
    resample_poly = cast(_ResamplePoly, getattr(scipy_signal, "resample_poly"))
    return resample_poly(audio, up, down)


class WakeWordDetector:
    """Wake word detection engine (background thread).

    Usage:
        detector = WakeWordDetector(config.audio)
        detector.load()
        detector.start(callback=on_detected)
        # ... app runs ...
        detector.stop()
    """

    # _CHUNK_SAMPLES = 1280  # openWakeWord expects 80ms @ 16kHz
    # _SAMPLE_RATE = 16000

    def __init__(self, config: Config) -> None:
        """Args:
        config: Config object (from utils.config) with wake and audio attributes.
        """
        super().__init__()
        self._config = config
        self._model: _WakeWordModelLike | None = None

        self._running = False
        self._callback: Callable[[], None] | None = None

        self._audio_queue: queue.Queue[NDArray[np.float32] | None] = queue.Queue(
            maxsize=50
        )
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        self._pa: _PyAudioLike | None = None
        self._stream: _PyAudioStreamLike | None = None
        self._native_chunk = 0
        self._capture_rate = 16000
        self._need_resample = False
        self._resample_up = 1
        self._resample_down = 1

        self._last_trigger_time = 0.0

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def load(self) -> None:
        """Initialize wake word model."""
        try:
            from openwakeword.model import Model
        except ImportError:
            msg = "openwakeword not installed. pip install openwakeword"
            raise ImportError(msg)

        model_name = self._config.wake.model_name
        model_path = self._config.wake.full_model_path

        logger.info("Loading wake word model: %s", model_name)
        logger.info("Loading wake word model path: %s", model_path)

        if not pathlib.Path(model_path).is_file():
            logger.error("Wake word model file not found at: %s", model_path)
            raise FileNotFoundError(f"Wake word model file not found: {model_path}")

        self._model = cast(
            _WakeWordModelLike,
            Model(
                wakeword_models=[str(model_path)],
                inference_framework=self._config.wake.inference_framework or "onnx",
                melspec_model_path=str(
                    self._config.wake.download_path / "melspectrogram.onnx"
                ),
                embedding_model_path=str(
                    self._config.wake.download_path / "embedding_model.onnx"
                ),
                # enable_speex_noise_suppression=self._config.wake.noise_suppression,
                # vad_threshold = self._config.vad.threshold
            ),
        )
        logger.info("Wake word detector ready")

    def unload(self) -> None:
        """Release resources."""
        self.stop()
        self._model = None

    # ====================================================================
    # Start / Stop
    # ====================================================================

    def start(self, callback: Callable[[], None]) -> None:
        """Start listening for wake word.

        Args:
            callback: Called when wake word is detected (no args).

        """
        if self._running:
            return

        self._callback = callback
        self._running = True

        if not self._open_input_stream():
            self._running = False
            return

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=False, name="wwd-capture"
        )
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=False, name="wwd-detect"
        )
        self._capture_thread.start()
        self._detect_thread.start()

        logger.info("Waiting for wake word: '%s'", self._config.wake.wake_word)

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        current_thread = threading.current_thread()

        # Stop the stream so any blocking read() can return, then wait for the
        # capture thread to exit before closing the PortAudio stream object.
        if self._stream:
            with contextlib.suppress(Exception):
                self._stream.stop_stream()

        # Join threads BEFORE pa.terminate() to avoid PortAudio segfault
        # and before closing the stream to avoid racing the C backend.
        if self._capture_thread and self._capture_thread is not current_thread:
            self._capture_thread.join(timeout=5.0)
            if self._capture_thread.is_alive() and self._stream:
                logger.warning(
                    "Wake word capture thread did not exit after stop_stream(); aborting stream"
                )
                with contextlib.suppress(Exception):
                    self._stream.abort_stream()
                self._capture_thread.join(timeout=2.0)

        if self._stream and (
            not self._capture_thread or not self._capture_thread.is_alive()
        ):
            with contextlib.suppress(Exception):
                self._stream.close()
        elif self._stream:
            logger.warning(
                "Wake word stream left open because capture thread is still alive"
            )

        if self._detect_thread:
            while True:
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._audio_queue.put(None)  # sentinel
            if self._detect_thread is not current_thread:
                self._detect_thread.join(timeout=5.0)
                if self._detect_thread.is_alive():
                    logger.warning(
                        "Wake word detect thread did not exit before shutdown completed"
                    )

        if self._pa:
            with suppress_pa_stderr(), contextlib.suppress(Exception):
                self._pa.terminate()

        logger.info("Wake word detector stopped")

    # ====================================================================
    # Capture thread
    # ====================================================================

    def _open_input_stream(self) -> bool:
        """Open the microphone stream before worker threads start."""
        try:
            import pyaudio
        except ImportError:
            logger.exception("pyaudio not installed. uv add pyaudio")
            return False

        model_rate = 16000  # openWakeWord STRICTLY requires 16000 Hz
        device_index = self._config.audio.input_device_index
        candidate_rates = [
            self._config.audio.input_sample_rate,
            44100,
            48000,
            16000,
            22050,
            8000,
        ]

        def _resolve_device_candidates(pa: _PyAudioLike) -> list[int | None]:
            candidates: list[int | None] = []
            if device_index is not None:
                try:
                    info = pa.get_device_info_by_index(device_index)
                    if (
                        int(cast(int | float | str, info.get("maxInputChannels", 0)))
                        > 0
                    ):
                        candidates.append(device_index)
                    else:
                        logger.warning(
                            "WDD: configured input device %s has no input channels; using default.",
                            device_index,
                        )
                except Exception:
                    logger.warning(
                        "WDD: configured input device %s is unavailable; using default.",
                        device_index,
                    )
            candidates.append(None)
            return candidates

        def _try_open(
            pa: _PyAudioLike, rate: int, dev_idx: int | None
        ) -> tuple[_PyAudioStreamLike | None, int]:
            native_chunk = max(
                1, round(self._config.audio.input_chunk_size * rate / model_rate)
            )
            try:
                return pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=native_chunk,
                    input_device_index=dev_idx,
                ), native_chunk
            except Exception:
                return None, 0

        try:
            with suppress_pa_stderr():
                self._pa = cast(_PyAudioLike, pyaudio.PyAudio())

            stream, native_chunk, capture_rate = None, 0, model_rate
            tried: list[tuple[int | None, int]] = []
            device_candidates = _resolve_device_candidates(self._pa)

            for dev in device_candidates:
                for rate in candidate_rates:
                    s, ch = _try_open(self._pa, rate, dev)
                    if s is not None:
                        stream, native_chunk, capture_rate = s, ch, rate
                        if dev != device_index:
                            logger.warning(
                                "WDD: device %s unavailable; using system default.",
                                device_index,
                            )
                        if rate != model_rate:
                            logger.info(
                                "WDD: device native rate is %d Hz; will resample to %d Hz.",
                                rate,
                                model_rate,
                            )
                        break
                    tried.append((dev, rate))
                if stream is not None:
                    break

            if stream is None:
                logger.error("WDD: could not open any microphone. Tried: %s", tried)
                if self._pa:
                    with suppress_pa_stderr(), contextlib.suppress(Exception):
                        self._pa.terminate()
                    self._pa = None
                return False

            self._stream = stream
            self._native_chunk = native_chunk
            self._capture_rate = capture_rate
            self._need_resample = capture_rate != model_rate
            if self._need_resample:
                from math import gcd

                g = gcd(model_rate, capture_rate)
                self._resample_up = model_rate // g
                self._resample_down = capture_rate // g

            logger.debug(
                "Wake word capture started (device=%s, rate=%d)",
                device_index,
                capture_rate,
            )
            return True
        except Exception as e:
            logger.exception("WDD capture setup error: %s", e)
            return False

    def _capture_loop(self) -> None:
        """Read microphone, push resampled chunks to queue.

        Auto-detects native device rate. If it differs from 16 kHz, each
        chunk is resampled so openWakeWord always receives 16 kHz audio.
        """
        if self._stream is None:
            logger.error("WDD capture loop started without an open stream")
            return

        while self._running:
            try:
                raw = self._stream.read(self._native_chunk, exception_on_overflow=False)
                if self._need_resample:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    pcm = _resample_poly(pcm, self._resample_up, self._resample_down)
                    audio = pcm / 32768.0
                else:
                    audio = cast(
                        NDArray[np.float32],
                        np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0,
                    )

                if not self._audio_queue.full():
                    self._audio_queue.put(audio)
            except Exception as e:
                if self._running:
                    logger.debug("WDD capture error: %s", e)

    # ====================================================================
    # Detection thread
    # ====================================================================

    def _detect_loop(self) -> None:
        """Consume audio, run model, trigger on wake word."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if chunk is None:  # sentinel
                break

            try:
                if self._model is None:
                    continue
                self._model.predict(chunk)
                scores = self._model.prediction_buffer.get(
                    self._config.wake.wake_word, [0.0]
                )
                score = scores[-1] if scores else 0.0

                now = time.time()
                if (
                    score >= self._config.wake.threshold
                    and (now - self._last_trigger_time)
                    > self._config.wake.cooldown_seconds
                ):
                    self._last_trigger_time = now
                    logger.info("Wake word detected (score=%.2f)", score)
                    if self._callback:
                        self._callback()

            except Exception as e:
                logger.debug("WDD prediction error: %s", e)
