# audio/wake_word.py
"""Wake word detection using openWakeWord.

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

from src.audio.audio_utils import AudioInputStream, open_input_stream_with_fallback

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.utils.config import Config

module_name = __name__
lib_name = module_name.split('.')[1]
logger = logging.getLogger(lib_name)


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

    _config: Config
    _model: _WakeWordModelLike | None
    _running: bool
    _callback: Callable[[], None] | None
    _audio_queue: queue.Queue[NDArray[np.float32] | None]
    _capture_thread: threading.Thread | None
    _detect_thread: threading.Thread | None
    _backend: str
    _stream: AudioInputStream | None
    _native_chunk: int
    _capture_rate: int
    _need_resample: bool
    _resample_up: int
    _resample_down: int
    _last_trigger_time: float

    def __init__(self, config: Config) -> None:
        """Args:
        config: Config object (from utils.config) with wake and audio attributes.
        """
        super().__init__()
        self._config = config
        self._model = None

        self._running = False
        self._callback = None

        self._audio_queue = queue.Queue(
            maxsize=50
        )
        self._capture_thread = None
        self._detect_thread = None

        self._backend = ""
        self._stream = None
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
            cast(
                object,
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
        logger.info(
            "Wake word detector listening with sounddevice backend...",
        )

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        current_thread = threading.current_thread()

        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()

        if self._capture_thread and self._capture_thread is not current_thread:
            self._capture_thread.join(timeout=5.0)

        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None

        if self._detect_thread:
            while True:
                try:
                    _item = self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._audio_queue.put(None)  # sentinel
            if self._detect_thread is not current_thread:
                self._detect_thread.join(timeout=5.0)
                if self._detect_thread.is_alive():
                    logger.warning(
                        "Wake word detect thread did not exit before shutdown completed"
                    )

        logger.info("Wake word detector stopped")

    # ====================================================================
    # Capture thread
    # ====================================================================

    def _open_input_stream(self) -> bool:
        """Open the microphone stream before worker threads start."""
        try:
            opened = open_input_stream_with_fallback(
                rate=16000,
                chunk_ms=self._config.audio.input_chunk_ms,
                device_index=self._config.audio.input_device_index,
                candidate_rates=[44100, 48000, 16000, 22050, 8000],
                dtype="float32",
            )
            if opened is None:
                logger.error("WDD: could not open any microphone.")
                return False

            self._backend = opened.backend
            self._stream = opened.stream
            self._native_chunk = opened.native_chunk_frames
            self._capture_rate = opened.capture_rate
            self._need_resample = opened.need_resample
            self._resample_up = opened.resample_up
            self._resample_down = opened.resample_down

            if opened.device_index != self._config.audio.input_device_index:
                logger.warning(
                    "WDD: device %s unavailable; using system default.",
                    self._config.audio.input_device_index,
                )
            if opened.capture_rate != 16000:
                logger.info(
                    "WDD: device native rate is %d Hz; will resample to 16000 Hz.",
                    opened.capture_rate,
                )

            logger.debug(
                "Wake word capture started (device=%s, rate=%d)",
                opened.device_index,
                opened.capture_rate,
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
                if not raw:
                    continue
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
                _object = self._model.predict(chunk)
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
