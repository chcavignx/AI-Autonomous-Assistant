"""audio/tts.py.
============
Text-to-Speech engine using Piper TTS.

Supports:
  - CLI subprocess (piper binary) — stable on ARM
  - Python API (PiperVoice) — streaming playback via sounddevice

Non-blocking queue-based threading for responsive conversational flow.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import BinaryIO, Protocol, cast

# Ensure 'src' is in sys.path
sys.path.insert(0, str(Path(os.path.join(Path(__file__).parent, "..")).resolve()))

from typing import TYPE_CHECKING
from src.audio.audio_utils import suppress_pa_stderr  # , install_alsa_error_handler
from src.utils.config import Config

logger = logging.getLogger(__name__)

_PIPER_BINARY_CANDIDATES = [
    Path.home() / ".local" / "bin" / "piper",
    Path("/usr/local/bin/piper"),
    Path("/usr/bin/piper"),
]


class _PyAudioStreamLike(Protocol):
    def write(self, data: bytes) -> None: ...

    def stop_stream(self) -> None: ...

    def close(self) -> None: ...


class _PyAudioLike(Protocol):
    def open(
        self,
        *,
        format: int,
        channels: int,
        rate: int,
        output: bool,
        output_device_index: int | None = ...,
    ) -> _PyAudioStreamLike: ...

    def terminate(self) -> None: ...

    def get_format_from_width(self, width: int) -> int: ...


class _PiperVoiceLike(Protocol):
    def synthesize_wav(
        self, *, text: str, wav_file: BinaryIO, set_wav_format: bool
    ) -> object: ...


class TTSEngine:
    """Text-to-speech synthesis + playback using Piper.

    Non-blocking: queue-based, separate playback thread.

    Usage:
        tts = TTSEngine(config.audio)
        tts.load()
        tts.speak("Hello world")
        tts.speak("Another phrase")  # queued, non-blocking
        tts.wait()  # block until queue empty
        tts.unload()
    """

    def __init__(self, config: Config) -> None:
        """Args:
        config: Config object (from utils.config) with tts and audio attributes.

        """
        super().__init__()
        self._config: Config = config
        self._piper_bin: Path | None = None
        self._engine = self._config.tts.engine
        self._model_name: str = self._config.tts.model_name
        self._model_path: Path = self._config.tts.full_model_path
        self._piper_voice: _PiperVoiceLike | None = None

        self._pa: _PyAudioLike | None = None
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._playback_thread: threading.Thread | None = None
        self._running = False
        self._is_speaking = threading.Event()

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def load(self) -> None:
        """Initialize TTS engine."""
        if not self._config.tts.cli_mode:
            self._load_piper_python()
        else:
            self._find_piper_binary()

        try:
            import pyaudio

            # if not hasattr(self, "_alsa_error_handler"):
            #     self._alsa_error_handler = install_alsa_error_handler()

            with suppress_pa_stderr():
                self._pa = cast(_PyAudioLike, pyaudio.PyAudio())
        except ImportError:
            msg = "pyaudio not installed. uv add pyaudio"
            raise ImportError(msg)

        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=False, name="tts-playback"
        )
        self._playback_thread.start()

        logger.info(
            "TTS ready (model=%s, use_cli=%s)",
            self._config.tts.model_name,
            self._config.tts.cli_mode,
        )

    def unload(self) -> None:
        """Shutdown TTS engine."""
        self._running = False
        self._tts_queue.put(None)  # sentinel
        if self._playback_thread:
            self._playback_thread.join(timeout=3.0)
            if self._playback_thread.is_alive():
                logger.warning("TTS playback thread did not terminate in time")
                return  # Don't terminate PyAudio while thread may still use it
        if self._pa:
            with suppress_pa_stderr():
                self._pa.terminate()
        self._piper_voice = None
        logger.info("TTS stopped")

    def _find_piper_binary(self) -> None:
        """Locate piper binary for CLI mode."""
        if self._config.tts.cli_mode:
            # Note: binary_path is not in TTSConfig, check if needed for CLI mode
            for candidate in _PIPER_BINARY_CANDIDATES:
                if candidate.exists():
                    self._piper_bin = candidate
                    logger.info("Piper binary: %s", candidate)
                    return

            msg = "Piper binary not found. Install piper via: https://github.com/rhasspy/piper"
            raise FileNotFoundError(msg)

    def _load_piper_python(self) -> None:
        """Load Piper Python API."""
        try:
            from piper import PiperVoice
        except ImportError:
            msg = "piper-tts not installed. pip install piper-tts"
            raise ImportError(msg)

        # Determine models directory from config
        models_dir = Path(self._config.tts.full_model_path).parent
        model_file = models_dir / self._config.tts.model_name

        if not model_file.exists():
            msg = f"Piper model not found: {model_file}\nDownload from huggingface or run setup script."
            raise FileNotFoundError(msg)

        self._model_path = model_file
        self._piper_voice = cast(_PiperVoiceLike, PiperVoice.load(str(model_file)))
        logger.info("Piper voice loaded: %s", model_file)

    # ====================================================================
    # Public API
    # ====================================================================

    def speak(self, text: str, blocking: bool = False) -> None:
        """Queue text for synthesis.

        Args:
            text: Text to synthesize (<500 chars recommended).
            blocking: If True, wait for playback to finish.

        """
        if not text.strip():
            return

        # Truncate long text for conversational responsiveness
        if len(text) > self._config.audio.output_chunk_size:
            logger.warning("TTS text truncated (%d chars)", len(text))
            text = text[: self._config.audio.output_chunk_size] + "..."

        self._tts_queue.put(text)

        if blocking:
            self.wait()

    def interrupt(self) -> None:
        """Clear queue and stop current playback."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("TTS interrupted")

    def wait(self) -> None:
        """Block until queue is empty."""
        self._tts_queue.join()

    @property
    def is_speaking(self) -> bool:
        """Check if currently playing audio."""
        return self._is_speaking.is_set()

    # ====================================================================
    # Playback thread
    # ====================================================================

    def _playback_loop(self) -> None:
        """Consume TTS queue and play audio."""
        while self._running:
            try:
                text = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # sentinel
                self._tts_queue.task_done()
                break

            try:
                self._is_speaking.set()
                wav_bytes = self._synthesize(text)
                if wav_bytes:
                    self._play_wav(wav_bytes)
            except Exception:
                logger.exception("TTS playback error")
            finally:
                self._is_speaking.clear()
                self._tts_queue.task_done()

    # ====================================================================
    # Synthesis
    # ====================================================================

    def _synthesize(self, text: str) -> bytes | None:
        """Synthesize text to WAV bytes.

        Returns:
            WAV bytes, or None on error.

        """
        if self._config.tts.cli_mode:
            return self._synthesize_via_cli(text)
        return self._synthesize_via_api(text)

    def _synthesize_via_cli(self, text: str) -> bytes | None:
        """Piper CLI subprocess → WAV bytes."""
        if not self._piper_bin:
            logger.error("Piper binary not found")
            return None

        model_file = self._config.tts.full_model_path

        # Defensive check for speed to avoid ZeroDivisionError or nonsense values
        speed = max(self._config.tts.speed, 0.01)

        cmd = [
            str(self._piper_bin),
            "--model",
            str(model_file),
            "--output_raw",
            "--length_scale",
            str(1.0 / speed),
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=15.0,
            )
            if result.returncode != 0:
                logger.error("Piper error: %s", result.stderr.decode())
                return None

            # Piper --output_raw → PCM s16le 22050Hz mono
            return self._pcm_to_wav(result.stdout, sample_rate=22050)

        except subprocess.TimeoutExpired:
            logger.exception("Piper synthesis timeout")
            return None
        except Exception as e:
            logger.exception("Piper subprocess error: %s", e)
            return None

    def _synthesize_via_api(self, text: str) -> bytes | None:
        """Piper Python API → WAV bytes."""
        if not self._piper_voice:
            logger.error("Piper voice not initialized")
            return None

        try:
            wav_buffer = io.BytesIO()
            self._piper_voice.synthesize_wav(
                text=text,
                wav_file=wav_buffer,
                set_wav_format=True,
            )
            return wav_buffer.getvalue()
        except Exception as e:
            logger.exception("Piper synthesis error: %s", e)
            return None

    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 22050) -> bytes:
        """Wrap raw PCM s16le in WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # ====================================================================
    # Playback
    # ====================================================================

    def _play_wav(self, wav_bytes: bytes) -> None:
        """Play WAV bytes via PyAudio."""
        if self._pa is None:
            return

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            with suppress_pa_stderr():
                stream = self._pa.open(
                    format=self._pa.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self._config.audio.output_device_index,
                )

            chunk_size = self._config.audio.output_chunk_size
            try:
                data = wf.readframes(chunk_size)
                while data and self._running:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
            finally:
                with suppress_pa_stderr():
                    stream.stop_stream()
                    stream.close()
