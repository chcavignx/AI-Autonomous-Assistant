"""audio/tts.py.
============
Text-to-Speech engine using Piper TTS.

Supports:
  - CLI subprocess (piper binary) — stable on ARM
  - Python API (PiperVoice) — streaming playback via sounddevice only

Non-blocking queue-based threading for responsive conversational flow.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import sys

import threading
import wave
from pathlib import Path
from typing import BinaryIO, Protocol, cast

# Ensure 'src' is in sys.path
sys.path.insert(0, str(Path(os.path.join(Path(__file__).parent, "..")).resolve()))

from src.audio.audio_utils import AudioPlayer
from src.utils.config import Config

module_name = __name__
lib_name = module_name.split('.')[1]
logger = logging.getLogger(lib_name)

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

    _config: Config
    _engine: str
    _model_name: str
    _model_path: Path
    _piper_voice: _PiperVoiceLike | None
    _audio_player: AudioPlayer
    _tts_queue: queue.Queue[str | None]
    _playback_thread: threading.Thread | None
    _running: bool
    _is_speaking: threading.Event

    def __init__(self, config: Config) -> None:
        """Args:
        config: Config object (from utils.config) with tts and audio attributes.

        """
        super().__init__()
        self._config = config
        self._engine = self._config.tts.engine
        self._model_name = self._config.tts.model_name
        self._model_path = self._config.tts.full_model_path
        self._piper_voice = None
        self._audio_player = AudioPlayer(self._config)
        self._tts_queue = queue.Queue()
        self._playback_thread = None
        self._running = False
        self._is_speaking = threading.Event()

    # ====================================================================
    # Lifecycle
    # ====================================================================

    def load(self) -> None:
        """Initialize TTS engine."""
        self._load_piper_python()
        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=False, name="tts-playback"
        )
        self._playback_thread.start()

        logger.info(
            "TTS ready (model=%s, backend=sounddevice)",
            self._config.tts.model_name,
        )

    def unload(self) -> None:
        """Shutdown TTS engine."""
        self._running = False
        self._tts_queue.put(None)  # sentinel
        if self._playback_thread:
            self._playback_thread.join(timeout=3.0)
            if self._playback_thread.is_alive():
                logger.warning("TTS playback thread did not terminate in time")
                return  # Don't close resources while thread may still use them
        self._audio_player.close()
        self._piper_voice = None
        logger.info("TTS stopped")

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
        self._piper_voice = cast(_PiperVoiceLike, cast(object, PiperVoice.load(str(model_file))))
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
                _item = self._tts_queue.get_nowait()
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
        """Piper Python API → WAV bytes."""
        if not self._piper_voice:
            logger.error("Piper voice not initialized")
            return None

        try:
            wav_buffer = io.BytesIO()
            # Piper expects a wave.Wave_write object, not raw BytesIO
            with wave.open(wav_buffer, "wb") as wav_file:
                _object = self._piper_voice.synthesize_wav(
                    text=text,
                    wav_file=cast(BinaryIO, cast(object, wav_file)),
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
        """Play WAV bytes using the active backend."""
        if self._running:
            _play = self._audio_player.play_wav_bytes(wav_bytes, block=True)
