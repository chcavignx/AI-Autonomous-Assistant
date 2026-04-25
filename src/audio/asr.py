# !/usr/bin/env python3
"""audio/asr.py.

============

Automatic Speech Recognition engine combining STT + VAD.

Architecture:
  1. Microphone → capture thread (PyAudio chunks)
  2. Silero VAD → detect speech vs silence
  3. Accumulate speech chunks
  4. On silence detected → STT (Whisper/Faster-Whisper) → transcript
  5. Callback on result

All offline, no cloud calls. Multiple backend support.
"""

from __future__ import annotations

import contextlib
import io
import math
import logging
import pathlib
import queue
import sys
import threading
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from math import gcd
from typing import BinaryIO, Protocol, cast

import numpy as np
from numpy.typing import NDArray
import faster_whisper
import pyaudio
import whisper


# Ensure 'src' is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import suppress_pa_stderr
from src.utils.config import Config

logger = logging.getLogger(__name__)


class _WhisperSegmentLike(Protocol):
    text: str


class _WhisperResultLike(Protocol):
    segments: Sequence[_WhisperSegmentLike]


class _WhisperModelLike(Protocol):
    def transcribe(
        self, audio: str | BinaryIO | NDArray[np.float32], **kwargs: object
    ) -> object: ...


class _FasterWhisperModelLike(Protocol):
    def transcribe(
        self,
        audio: str | BinaryIO | NDArray[np.float32],
        **kwargs: object,
    ) -> tuple[Iterable[_WhisperSegmentLike], object]: ...


class _VadScoreLike(Protocol):
    def item(self) -> float: ...


class _VadModelLike(Protocol):
    def __call__(self, audio: object, sample_rate: int) -> _VadScoreLike: ...


class _PyAudioStreamLike(Protocol):
    def read(self, num_frames: int, exception_on_overflow: bool = ...) -> bytes: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...


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

    def get_device_info_by_index(self, index: int) -> Mapping[str, object]: ...


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


class ASREngine:
    """Speech recognition + voice activity detection engine.

    Runs capture and processing in background threads. Calls callback on each
    recognized utterance.

    Usage:
        asr = ASREngine(config.audio)
        asr.load()
        asr.start(callback=on_transcript)
        # ... app runs ...
        asr.stop()
        asr.unload()
    """

    def __init__(self, config: Config) -> None:
        """Initialize the ASR engine with the given configuration."""
        super().__init__()
        self._config: Config = config
        self._stt_model: _WhisperModelLike | _FasterWhisperModelLike | None = None
        self._vad_model: _VadModelLike | None = None

        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._transcript_callback: Callable[[str], None] | None = None

        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

        self._pa: _PyAudioLike | None = None
        self._stream: _PyAudioStreamLike | None = None
        self._chunk_frames = 0
        self._capture_rate = 0
        self._need_resample = False
        self._resample_up = 1
        self._resample_down = 1

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def load(self) -> None:
        """Initialize STT and VAD models."""
        self._configure_torch_runtime()
        self._load_stt_model()
        self._load_vad_model()
        logger.info(
            "ASR ready (engine=%s, lang=%s)",
            self._config.asr.engine,
            self._config.asr.language,
        )

    def unload(self) -> None:
        """Release resources."""
        self.stop()
        # NOTE: On some ARM/PortAudio stacks, freeing the Faster-Whisper
        # (CTranslate2) model during shutdown can segfault. Skip releasing
        # that model in-process; rely on OS cleanup at exit.
        engine = self._config.asr.engine.lower()
        if self._config.asr.skip_native_teardown:
            logger.warning("ASR unload skipped by config to avoid native segfaults")
            return

        if engine in {"faster-whisper", "faster_whisper"}:
            logger.warning(
                "ASR unload skipped for faster-whisper to avoid native segfaults"
            )
            return

        self._stt_model = None
        self._vad_model = None

    def _load_stt_model(self) -> None:
        """Load the selected STT backend."""
        try:
            engine = self._config.asr.engine.lower()
            if engine == "whisper":
                self._load_whisper()
            elif engine in {"faster-whisper", "faster_whisper"}:
                self._load_faster_whisper()
            else:
                logger.warning("Unknown STT backend: %s", engine)
                return
        except Exception as e:
            logger.exception("STT model load error: %s", e)
            return

    def _load_whisper(self) -> None:
        """Load openai-whisper."""
        try:
            import whisper

            logger.info("Loading Whisper %s...", self._config.asr.model_size)
            self._stt_model = cast(
                _WhisperModelLike,
                whisper.load_model(
                    name=self._config.asr.model_size,
                    device=self._config.asr.device,
                    download_root=str(self._config.asr.download_path),
                ),
            )
            logger.info("Whisper loaded")
        except ImportError:
            logger.exception("openai-whisper not installed. pip install openai-whisper")
            return

    def _load_faster_whisper(self) -> None:
        """Load faster-whisper (recommended for Pi5)."""
        try:
            import faster_whisper
            logger.info("Loading Faster-Whisper %s...", self._config.asr.model_size)
            self._stt_model = cast(
                _FasterWhisperModelLike,
                faster_whisper.WhisperModel(
                    model_size_or_path=self._config.asr.model_size,
                    device=self._config.asr.device,
                    compute_type=self._config.asr.compute_type,
                    cpu_threads=1,
                    num_workers=1,
                ),
            )
            logger.info("Faster-Whisper loaded")
        except ImportError:
            logger.exception("faster-whisper not installed. pip install faster-whisper")
            return

    @staticmethod
    def _configure_torch_runtime() -> None:
        """Keep Torch execution single-threaded for stability in audio workers."""
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

        try:
            import torch

            torch.set_num_interop_threads(1)
        except Exception:
            pass

    def _load_vad_model(self) -> None:
        """Load Silero VAD (lightweight, offline)."""
        try:
            from silero_vad import load_silero_vad

            self._vad_model = cast(_VadModelLike, load_silero_vad())
            logger.info("Silero VAD loaded")
        except Exception as e:
            logger.exception(
                "Silero VAD load failed (%s), using energy-based fallback", e
            )
            self._vad_model = None

    # ====================================================================
    # Start / Stop
    # ====================================================================

    def start(self, callback: Callable[[str], None]) -> None:
        """Start listening in background threads.

        Args:
            callback: Called with recognized text (one utterance per call).

        """
        if self._running:
            return

        self._transcript_callback = callback
        self._running = True

        if not self._open_input_stream():
            self._running = False
            return

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=False, name="asr-capture"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=False, name="asr-process"
        )
        self._capture_thread.start()
        self._process_thread.start()

        logger.info("ASR listening...")

    def stop(self) -> None:
        """Stop listening and clean up threads."""
        self._running = False
        current_thread = threading.current_thread()

        # Suppress PortAudio/ALSA stderr noise across the entire teardown
        # (mmap drain errors, residual JACK messages, etc. are all cosmetic).
        with suppress_pa_stderr():
            # Stop the stream first so any blocking read() call can unwind,
            # then wait for the capture thread to exit before closing the
            # underlying PortAudio stream object.
            if self._stream:
                with contextlib.suppress(Exception):
                    self._stream.stop_stream()

            if self._capture_thread and self._capture_thread is not current_thread:
                self._capture_thread.join(timeout=5.0)

            if self._stream and (
                not self._capture_thread or not self._capture_thread.is_alive()
            ):
                with contextlib.suppress(Exception):
                    self._stream.close()
                logger.info("Audio stream closed")
            elif self._stream:
                logger.warning(
                    "ASR stream left open because capture thread is still alive"
                )

            # Drop any queued audio so shutdown does not wait for the process
            # thread to chew through an entire backlog before it sees the
            # sentinel. That backlog can be large and keep native libraries
            # alive long enough to crash during interpreter teardown.
            while True:
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Let the processing thread exit immediately after capture stops.
            if self._process_thread:
                self._audio_queue.put(None)  # sentinel
                if self._process_thread is not current_thread:
                    self._process_thread.join(timeout=5.0)
                    if self._process_thread.is_alive():
                        logger.warning(
                            "ASR process thread did not exit before shutdown completed"
                        )

            # Safe to terminate now that no threads are using PortAudio
            if self._pa:
                with contextlib.suppress(Exception):
                    self._pa.terminate()

        logger.info("ASR stopped")

    # ====================================================================
    # Capture thread (PyAudio → queue)
    # ====================================================================
    def _resolve_device_candidates(self) -> list[int | None]:
        if self._pa is None:
            return [None]

        candidates: list[int | None] = []
        if self._config.audio.input_device_index is not None:
            try:
                info = self._pa.get_device_info_by_index(
                    self._config.audio.input_device_index
                )
                max_channels_raw = info.get("maxInputChannels", 0)
                try:
                    max_channels = int(cast(int | float | str, max_channels_raw))
                except (TypeError, ValueError):
                    max_channels = 0

                if max_channels > 0:
                    candidates.append(self._config.audio.input_device_index)
                else:
                    logger.warning(
                        "ASR: configured input device %s has no input channels; using default.",
                        self._config.audio.input_device_index,
                    )
            except Exception:
                logger.warning(
                    "ASR: configured input device %s is unavailable; using default.",
                    self._config.audio.input_device_index,
                )
                candidates.append(None)

        # Always try the system default microphone as a fallback.
        # This is the common case when the config leaves input_device_index unset.
        if None not in candidates:
            candidates.append(None)

        return candidates

    def _try_open(
        self, rate: int, dev_idx: int | None
    ) -> tuple[_PyAudioStreamLike | None, int]:
        chunk = int(rate * self._config.audio.input_chunk_ms / 1000)
        try:
            if self._pa is None:
                return None, 0
            return self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
                input_device_index=dev_idx,
            ), chunk
        except Exception:
            logger.exception("Failed to open audio stream")
            return None, 0

    def _open_input_stream(self) -> bool:
        """Open the microphone stream before worker threads start."""
        model_rate = self._config.audio.input_sample_rate
        device_index = self._config.audio.input_device_index
        candidate_rates = [model_rate, 44100, 48000, 22050, 8000]

        try:
            with suppress_pa_stderr():
                self._pa = cast(_PyAudioLike, pyaudio.PyAudio())

            stream, chunk_frames, capture_rate = None, 0, model_rate
            tried: list[tuple[int | None, int]] = []
            device_candidates = self._resolve_device_candidates()

            for dev in device_candidates:
                for rate in candidate_rates:
                    s, ch = self._try_open(rate, dev)
                    if s is not None:
                        stream, chunk_frames, capture_rate = s, ch, rate
                        if dev != device_index:
                            logger.warning(
                                "ASR: device %s unavailable; using system default.",
                                device_index,
                            )
                        if rate != model_rate:
                            logger.info(
                                "ASR: device native rate is %d Hz; will resample to %d Hz.",
                                rate,
                                model_rate,
                            )
                        break
                    tried.append((dev, rate))
                if stream is not None:
                    break

            if stream is None:
                logger.error("ASR: could not open any microphone. Tried: %s", tried)
                if self._pa:
                    with contextlib.suppress(Exception):
                        self._pa.terminate()
                    self._pa = None
                return False
            self._stream = stream
            self._chunk_frames = chunk_frames
            self._capture_rate = capture_rate
            self._need_resample = capture_rate != model_rate
            if self._need_resample:
                g = gcd(model_rate, capture_rate)
                self._resample_up = model_rate // g
                self._resample_down = capture_rate // g

            logger.debug(
                "ASR capture started (device=%s, capture_rate=%d, chunk=%dms)",
                device_index,
                capture_rate,
                self._config.audio.input_chunk_ms,
            )
            return True

        except Exception as e:
            logger.exception("ASR capture setup error: %s", e)
            return False

    def _capture_loop(self) -> None:
        """Read microphone continuously, push resampled chunks to queue.

        Automatically detects the device's native sample rate. If it differs
        from the model's required 16 kHz, each chunk is resampled before being
        pushed to the processing queue (scipy.signal.resample_poly).
        """
        if self._stream is None:
            logger.error("ASR capture loop started without an open stream")
            return

        while self._running:
            try:
                raw = self._stream.read(
                    num_frames=self._chunk_frames, exception_on_overflow=False
                )
                if self._need_resample:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    pcm = _resample_poly(pcm, self._resample_up, self._resample_down)
                    raw = pcm.astype(np.int16).tobytes()
                self._audio_queue.put(raw)
            except Exception as e:
                if self._running:
                    logger.warning("ASR capture error: %s", e)

    # ====================================================================
    # Process thread (VAD + Whisper)
    # ====================================================================

    def _process_loop(self) -> None:
        """Consume audio queue: detect speech, accumulate, transcribe on silence."""
        speech_buffer: list[bytes] = []
        silence_chunks = 0
        speaking = False

        # Convert silence timeout to chunk count
        silence_chunks_threshold = int(
            self._config.vad.silence_timeout_seconds
            * 1000
            / self._config.audio.input_chunk_ms
        )

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # sentinel
                break

            is_speech = self._detect_speech(chunk)

            if is_speech:
                speaking = True
                silence_chunks = 0
                speech_buffer.append(chunk)
            elif speaking:
                # Extend buffer with silence
                silence_chunks += 1
                speech_buffer.append(chunk)

                # Enough silence = end of utterance
                if silence_chunks >= silence_chunks_threshold:
                    if speech_buffer:
                        audio_bytes = b"".join(speech_buffer)
                        self._transcribe(audio_bytes)
                    speech_buffer = []
                    silence_chunks = 0
                    speaking = False

    def _detect_speech(self, chunk: bytes) -> bool:
        """Detect if chunk contains speech (VAD or energy fallback)."""
        if self._vad_model is None:
            return self._energy_based_vad(chunk)

        try:
            import torch

            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            tensor = torch.as_tensor(audio_float, dtype=torch.float32)

            confidence = self._vad_model(
                tensor, self._config.audio.input_sample_rate
            ).item()
            return confidence >= self._config.vad.threshold

        except Exception:
            return self._energy_based_vad(chunk)

    @staticmethod
    def _energy_based_vad(chunk: bytes) -> bool:
        """Fallback: detect speech by RMS energy threshold."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = math.sqrt(float(np.mean(audio**2)))
        return rms > 300  # empirical threshold for standard USB mic

    def _transcribe(self, audio_bytes: bytes) -> None:
        """Transcribe accumulated audio buffer."""
        if self._stt_model is None:
            return

        try:
            text = self._transcribe_whisper(audio_bytes=audio_bytes)
            if text and len(text.strip()) > 2:
                logger.info("ASR: %s", text)
                if self._transcript_callback:
                    self._transcript_callback(text)

        except Exception as e:
            logger.exception("Transcription error: %s", e)

    def _transcribe_whisper(self, audio_bytes: bytes) -> str:
        """Transcribe using Whisper or Faster-Whisper."""
        # Convert bytes to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._config.audio.input_sample_rate)
            wf.writeframes(audio_bytes)
        wav_buffer.seek(0)

        if self._config.asr.engine in {"faster-whisper", "faster_whisper"}:
            model = cast(_FasterWhisperModelLike | None, self._stt_model)
            if model is None:
                return ""
            segments, _info = model.transcribe(
                audio=wav_buffer,
                language=self._config.asr.language,
                beam_size=3,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            result: object = list(segments)
        else:
            # openai-whisper
            model = cast(_WhisperModelLike | None, self._stt_model)
            if model is None:
                return ""
            result = model.transcribe(
                wav_buffer,
                task="translate" if self._config.asr.translate else "transcribe",
                language=self._config.asr.language,
                temperature=0.0,
                word_timestamps=True,
                fp16=False,
            )
        return self._extract_text_from_result(result)

    def _extract_text_from_result(self, result: object) -> str:
        """Extract text from Whisper/Faster-Whisper result, handling different formats."""
        # Handle different return formats from different Whisper versions
        if isinstance(result, dict):
            # Standard whisper returns a dict with 'segments' key
            result_dict = cast(dict[str, object], result)
            segments_source: object = result_dict.get("segments", [])
        elif isinstance(result, tuple):
            # Some versions return (segments, info) tuple
            result_tuple = cast(tuple[object, ...], result)
            segments_source = result_tuple[0] if len(result_tuple) > 0 else []
        else:
            # Handle case where result is a single Segment or a result object
            if hasattr(result, "segments"):
                segments_source = cast(_WhisperResultLike, result).segments
            elif isinstance(result, list):
                segments_source = cast(list[object], result)
            else:
                segments_source = [result]

        if isinstance(segments_source, list):
            segments = cast(list[object], segments_source)
        elif isinstance(segments_source, tuple):
            segments = list(cast(tuple[object, ...], segments_source))
        elif isinstance(segments_source, Sequence) and not isinstance(
            segments_source, (str, bytes)
        ):
            segments = list(cast(Sequence[object], segments_source))
        elif isinstance(segments_source, Iterable) and not isinstance(
            segments_source, (str, bytes)
        ):
            segments = list(cast(Iterable[object], segments_source))
        else:
            segments = [segments_source]

        # Extract and clean text
        text_segments: list[str] = []
        if segments:
            for segment in segments:
                # Handle different segment formats
                if isinstance(segment, dict):
                    text_raw = cast(dict[str, object], segment).get("text", "")
                elif hasattr(segment, "text"):
                    text_raw = cast(_WhisperSegmentLike, segment).text
                elif isinstance(segment, str):
                    text_raw = segment
                else:
                    text_raw = ""

                if isinstance(text_raw, str) and text_raw.strip():
                    text_segments.append(text_raw.strip())

        return " ".join(text_segments).strip()

    # ====================================================================
    # Static helpers
    # ====================================================================

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a file (for testing/debugging).

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text.

        """
        if self._stt_model is None:
            msg = "ASREngine not loaded"
            raise RuntimeError(msg)

        if self._config.asr.engine.lower() in {"faster-whisper", "faster_whisper"}:
            model = cast(_FasterWhisperModelLike, self._stt_model)
            segments, _info = model.transcribe(
                audio_path,
                language=self._config.asr.language,
            )
            return self._extract_text_from_result(segments)
        else:
            model = cast(_WhisperModelLike, self._stt_model)
            result = model.transcribe(
                audio_path,
                language=self._config.asr.language,
            )
            return self._extract_text_from_result(result)
