# audio/wake_word.py
"""Wake word detection — two available backends.

WakeWordDetector (ONNX Runtime direct, default):
  Lightweight dedicated KWS model, <5 ms/chunk.
  Loads openWakeWord pre-trained models (.onnx) directly via ONNX Runtime.
  Requires:  pip install onnxruntime numpy
  Pre-trained: hey_jarvis, hey_mycroft, alexa, hey_google, …
"""

from __future__ import annotations

import contextlib
from collections import deque
import gc
import logging
import pathlib
import queue
import sys
import threading
import time
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, final, Callable

import numpy as np
from numpy.typing import NDArray

# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import AudioInputStream, open_input_stream_with_fallback

if TYPE_CHECKING:
    from src.utils.config import Config
    import onnxruntime as ort

module_name = __name__
lib_name = module_name.split('.')[1]
logger = logging.getLogger(lib_name)


class _ResamplePoly(Protocol):
    """Protocol for scipy.signal.resample_poly."""

    def __call__(
        self, x: NDArray[np.float32], up: int, down: int
    ) -> NDArray[np.float32]:
        """Resample audio signal using polyphase method."""
        ...


def _resample_poly(
    audio: NDArray[np.float32], up: int, down: int
) -> NDArray[np.float32]:
    scipy_signal = import_module("scipy.signal")
    resample_poly = cast(_ResamplePoly, getattr(scipy_signal, "resample_poly"))
    return resample_poly(audio, up, down)


def _default_transform(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return x / 10 + 2


@final
class ONNXAudioFeatures:
    """Computes audio features using melspectrogram and embedding models via ONNX Runtime.
    See https://github.com/sujitvasanth/openwakeword-simplified.git"""

    melspec_session: ort.InferenceSession
    embedding_session: ort.InferenceSession
    sr: int
    raw_data_buffer: deque[float]
    melspectrogram_buffer: NDArray[np.float32]
    melspectrogram_max_len: int
    accumulated_samples: int
    raw_data_remainder: NDArray[np.float32]
    feature_buffer: NDArray[np.float32]
    feature_buffer_max_len: int

    def __init__(self, melspec_session: ort.InferenceSession, embedding_session: ort.InferenceSession, sr: int = 16000) -> None:
        """Initialize ONNX audio feature extractor with model sessions."""
        self.melspec_session = melspec_session
        self.embedding_session = embedding_session
        self.sr = sr

        self.raw_data_buffer = deque(maxlen=sr * 10)
        self.melspectrogram_buffer = np.ones((76, 32), dtype=np.float32)  # n_frames x num_features
        self.melspectrogram_max_len = 10 * 97  # 97 frames/second of 16kHz audio
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0, dtype=np.float32)

        # Initialize feature buffer with random embeddings to match openWakeWord startup behavior
        random_audio = np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16)
        self.feature_buffer = self._get_embeddings(random_audio)
        self.feature_buffer_max_len = 120  # ~10 seconds of feature history

    def reset(self) -> None:
        """Reset the internal audio and spectrogram buffers."""
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32), dtype=np.float32)
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0, dtype=np.float32)
        random_audio = np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16)
        self.feature_buffer = self._get_embeddings(random_audio)

    def _get_melspectrogram(
        self,
        x: NDArray[np.float32] | NDArray[np.int16] | list[float],
        melspec_transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] = _default_transform
    ) -> NDArray[np.float32]:
        """Compute the log-mel spectrogram of the input audio samples."""
        if isinstance(x, list):
            arr = np.array(x, dtype=np.float32)
        else:
            arr = x.astype(np.float32)

        if np.max(np.abs(arr)) <= 1.01:
            arr = arr * 32767.0

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        outputs = cast("list[NDArray[np.float32]]", self.melspec_session.run(None, {'input': arr}))
        spec = outputs[0]

        if spec.ndim == 4:
            spec = np.squeeze(spec, axis=(0, 1))

        spec = melspec_transform(spec)
        return spec

    def _get_embeddings_from_melspec(self, melspec: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute the Google speech embedding features from a mel-spectrogram."""
        if melspec.ndim == 2:
            melspec = np.expand_dims(melspec, axis=0)
        if melspec.ndim == 3:
            melspec = np.expand_dims(melspec, axis=-1)

        res = cast("list[NDArray[np.float32]]", self.embedding_session.run(None, {'input_1': melspec}))[0]
        return np.reshape(res, (melspec.shape[0], 96))

    def _get_embeddings(self, x: NDArray[np.float32] | NDArray[np.int16], window_size: int = 76, step_size: int = 8) -> NDArray[np.float32]:
        """Compute audio embeddings directly from raw audio samples."""
        spec = self._get_melspectrogram(x)
        windows: list[NDArray[np.float32]] = []
        for i in range(0, spec.shape[0], step_size):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:
                windows.append(window)
        if not windows:
            return np.empty((0, 96), dtype=np.float32)
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        return self._get_embeddings_from_melspec(batch)

    def _buffer_raw_data(self, x: NDArray[np.float32]) -> None:
        """Add raw audio samples to the input queue buffer."""
        self.raw_data_buffer.extend(cast("list[float]", x.tolist()))

    def _streaming_melspectrogram(self, n_samples: int) -> None:
        """Compute the spectrogram for newly accumulated streaming audio samples."""
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

        new_samples = list(self.raw_data_buffer)[-n_samples - 160 * 3:]
        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._get_melspectrogram(new_samples))
        )
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def streaming_features(self, x: NDArray[np.float32]) -> int:
        """Process incoming raw audio chunk, updating spectrograms and embeddings."""
        processed_samples = 0
        x = x.astype(np.float32)

        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0, dtype=np.float32)

        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            else:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0, dtype=np.float32)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:
            self._streaming_melspectrogram(self.accumulated_samples)

            for i in range(self.accumulated_samples // 1280 - 1, -1, -1):
                ndx = -8 * i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                window = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if window.shape[1] == 76:
                    self.feature_buffer = np.vstack(
                        (self.feature_buffer, self._get_embeddings_from_melspec(window))
                    )

            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

        return processed_samples if processed_samples != 0 else self.accumulated_samples

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1) -> NDArray[np.float32]:
        """Retrieve a specific history window of computed audio embedding features."""
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames)
            if start_ndx + n_feature_frames == 0:
                end_ndx = len(self.feature_buffer)
            res = self.feature_buffer[start_ndx:end_ndx, :]
        else:
            res = self.feature_buffer[-int(n_feature_frames):, :]
        return np.expand_dims(res, axis=0).astype(np.float32)

    def __call__(self, x: NDArray[np.float32]) -> int:
        """Call shortcut for streaming_features."""
        return self.streaming_features(x)


@final
class WakeWordDetector:
    """Wake word detection engine using ONNX Runtime directly.

    Usage:
        detector = WakeWordDetector(config)
        detector.load()
        detector.start(callback=on_detected)
        # ... app runs ...
        detector.stop()
    """

    _CHUNK_SAMPLES: int = 1280  # openWakeWord expects 80ms @ 16kHz
    _SAMPLE_RATE: int = 16000

    _config: Config
    _melspec_sess: ort.InferenceSession | None
    _embedding_sess: ort.InferenceSession | None
    _ww_sess: ort.InferenceSession | None
    _preprocessor: ONNXAudioFeatures | None
    _model: object | None
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
    _prediction_count: int

    def __init__(self, config: Config) -> None:
        """Initialize WakeWordDetector using the ONNX backend configuration."""
        super().__init__()
        self._config = config
        self._melspec_sess = None
        self._embedding_sess = None
        self._ww_sess = None
        self._preprocessor = None
        self._model = None

        self._running = False
        self._callback = None

        self._audio_queue = queue.Queue(maxsize=200)
        self._capture_thread = None
        self._detect_thread = None

        self._backend = ""
        self._stream = None
        self._native_chunk = self._CHUNK_SAMPLES
        self._capture_rate = self._SAMPLE_RATE
        self._need_resample = False
        self._resample_up = 1
        self._resample_down = 1

        self._last_trigger_time = 0.0
        self._prediction_count = 0

    def load(self) -> None:
        """Initialize wake word model sessions."""
        # Get paths from config, replacing .tflite with .onnx as needed
        model_path = pathlib.Path(self._config.wake.full_model_path)
        if model_path.suffix == ".tflite":
            model_path = model_path.with_suffix(".onnx")

        # In unit tests, if model_path is not found, we can try to find it under "wakeword" subfolder
        if not model_path.is_file():
            alt_path = model_path.parent / "wakeword" / model_path.name
            if alt_path.is_file():
                model_path = alt_path

        # Raise FileNotFoundError matching regex in tests
        if not model_path.is_file():
            logger.error("Required model file not found: %s", model_path)
            raise FileNotFoundError(f"Wake word model file not found: {model_path}")

        try:
            import onnxruntime as ort
        except ImportError:
            msg = "onnxruntime not installed. pip install onnxruntime"
            raise ImportError(msg)

        melspec_path = pathlib.Path(self._config.wake.melspec_model_path)
        if melspec_path.suffix == ".tflite":
            melspec_path = melspec_path.with_suffix(".onnx")

        embedding_path = pathlib.Path(self._config.wake.embedding_model_path)
        if embedding_path.suffix == ".tflite":
            embedding_path = embedding_path.with_suffix(".onnx")

        logger.info("Loading wake word ONNX sessions...")
        logger.info("  Melspec path: %s", melspec_path)
        logger.info("  Embedding path: %s", embedding_path)
        logger.info("  Wake Word path: %s", model_path)

        for p in [melspec_path, embedding_path, model_path]:
            if not p.is_file():
                logger.error("Required model file not found: %s", p)
                raise FileNotFoundError(f"Wake word model file not found: {p}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self._melspec_sess = ort.InferenceSession(str(melspec_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._embedding_sess = ort.InferenceSession(str(embedding_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._ww_sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

        self._preprocessor = ONNXAudioFeatures(self._melspec_sess, self._embedding_sess)
        self._prediction_count = 0
        logger.info("ONNX Wake word detector ready")

    def unload(self) -> None:
        """Unload loaded ONNX sessions and release resources."""
        self.stop()
        self._melspec_sess = None
        self._embedding_sess = None
        self._ww_sess = None
        self._preprocessor = None

    def start(self, callback: "Callable[[], None]") -> None:
        """Start microphone capture and detector threads."""
        if self._running:
            return

        self._callback = callback
        self._running = True

        # Clear the queue of any leftover items (including None sentinel)
        while not self._audio_queue.empty():
            try:
                _ = self._audio_queue.get_nowait()
            except queue.Empty:
                break

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
        """Stop detector loops, close stream, and join threads."""
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
                    _ = self._audio_queue.get_nowait()
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
        _ = gc.collect()

    def _open_input_stream(self) -> bool:
        """Open input audio stream using fallback sample rates."""
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
        """Read microphone, push resampled chunks to queue."""
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

    def _detect_loop(self) -> None:
        """Consume audio, run model, trigger on wake word."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if chunk is None:
                break

            try:
                if self._ww_sess is not None and self._preprocessor is not None:
                    # Pass normalized float32 chunk (it gets scaled inside ONNXAudioFeatures)
                    _ = self._preprocessor(chunk)

                    # Get the 16 embedding frames [1, 16, 96]
                    features = self._preprocessor.get_features(16)

                    # Predict score
                    inputs = self._ww_sess.get_inputs()
                    ww_input_name = str(inputs[0].name)
                    outputs = self._ww_sess.run(None, {ww_input_name: features})
                    score = float(outputs[0][0][0])  # pyright: ignore[reportIndexIssue]

                    self._prediction_count += 1
                    if self._prediction_count < 5:
                        score = 0.0
                else:
                    continue
                logger.debug("WDD prediction score: %s", score)
                now = time.time()
                if (
                    score >= self._config.wake.threshold
                    and (now - self._last_trigger_time) > self._config.wake.cooldown_seconds
                ):
                    self._last_trigger_time = now
                    logger.info("Wake word detected (score=%.2f)", score)
                    if self._callback:
                        self._callback()

            except Exception as e:
                logger.debug("WDD prediction error: %s", e)
