"""Whisper ASR Hailo-8L accelerated streamer module."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Generator, Optional # pyright: ignore[reportDeprecated]

import numpy as np
import psutil
import sounddevice as sd

from utils.config import MODELS_DIR, config # pyright: ignore[reportUnusedImport]
from utils.metrics import LatencyMeter # pyright: ignore[reportUnusedImport]
from vision.hailo_backend import HailoExecutor, HailoModelConfig # pyright: ignore[reportMissingImports]


@dataclass
class HailoAsrChunkResult:
    """ASR result details for a chunk."""

    text: str
    rtf: float
    cpu_percent: float
    ram_mb: float


@dataclass
class WhisperHailoConfig:
    """Configuration settings for WhisperHailoStreamer."""

    hef_path: str = str(MODELS_DIR / "whisper_tiny_hailo.hef")
    sample_rate: int = 16000
    chunk_duration_s: float = 1.0


class WhisperHailoStreamer:
    """ASR transcription streamer accelerated by Hailo-8L co-processor."""

    def __init__(self, cfg: WhisperHailoConfig | None = None) -> None:
        """Initialize the streamer and load the Whisper HEF.

        Args:
            cfg: Configuration for the streamer.
        """
        self.cfg = cfg or WhisperHailoConfig()
        self.executor = HailoExecutor(HailoModelConfig(hef_path=self.cfg.hef_path))
        self.sample_rate = self.cfg.sample_rate
        self._proc = psutil.Process()

    def record_stream(self) -> Generator[np.ndarray, None, None]:
        """Record audio input chunk by chunk from default input device.

        Yields:
            Captured float32 audio numpy array.
        """
        frames_per_chunk = int(self.sample_rate * self.cfg.chunk_duration_s)
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
        ):
            while True:
                audio = sd.rec(frames_per_chunk, samplerate=self.sample_rate, channels=1, dtype="float32")
                sd.wait()
                yield audio.reshape(-1)

    def _run_model(self, audio: np.ndarray) -> str:
        """Run inference on the audio data.

        Args:
            audio: Sound data.

        Returns:
            Decoded text output.
        """
        # Fit input shape and normalize
        inp = np.expand_dims(audio.astype(np.float32), axis=0)
        outputs = self.executor.run({self.executor.input_name: inp})
        out = outputs[self.executor.output_name] # pyright: ignore[reportUnusedVariable]

        # Placeholder: Return empty string in the integration wrapper.
        # This will be mapped to text tokens using specific Whisper tokenizer in deployment.
        return ""

    def transcribe_chunk(self, audio: np.ndarray) -> Optional[HailoAsrChunkResult]: # pyright: ignore[reportDeprecated]
        """Transcribe an audio chunk and measure system statistics.

        Args:
            audio: Audio segment.

        Returns:
            HailoAsrChunkResult containing transcribed text and CPU/RTF KPIs.
        """
        cpu_before = psutil.cpu_percent(interval=None) # pyright: ignore[reportUnusedVariable]
        mem_before = self._proc.memory_info().rss / (1024 * 1024) # pyright: ignore[reportUnusedVariable]

        t0 = perf_counter()
        text = self._run_model(audio)
        t1 = perf_counter()

        # If empty text is returned (e.g. silence or placeholder), skip reporting
        if not text:
            return None

        audio_dur = len(audio) / self.sample_rate
        rtf = (t1 - t0) / max(audio_dur, 1e-6)

        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = self._proc.memory_info().rss / (1024 * 1024)

        return HailoAsrChunkResult(
            text=text,
            rtf=rtf,
            cpu_percent=cpu_after,
            ram_mb=mem_after,
        )
