"""src/audio/vad.py.
=================
Voice Activity Detection using Silero VAD.

Accepts an AgentConfig (from voice_agent_offline) and provides:
  - is_speech_detected(audio_data) -> bool
  - get_speech_segments(audio_data) -> List[Dict]

Also applies Raspberry Pi CPU tuning on load.

Dependencies:
  pip install silero-vad torch
"""

from __future__ import annotations

import pathlib
import sys
from importlib import import_module
from torch._tensor import Tensor
from torch._tensor import Tensor
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

# Ensure 'src' is in sys.path before any local imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

WHISPER_MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
]
ENGLISH_ONLY_COMPATIBLE = ["tiny", "base", "small", "medium"]


from utils.sysutils import detect_raspberry_pi_model, limit_cpu_for_multiprocessing

if TYPE_CHECKING:
    from utils.config import Config


class _VadModelLike(Protocol):
    def __call__(self, audio: object, sample_rate: int) -> torch.Tensor: ...


class _GetSpeechTimestamps(Protocol):
    def __call__(
        self,
        audio: torch.Tensor,
        model: _VadModelLike,
        **kwargs: object,
    ) -> list[object]: ...


def _load_silero_vad_model() -> _VadModelLike:
    silero_vad = import_module("silero_vad")
    return cast(_VadModelLike, getattr(silero_vad, "load_silero_vad")())


def _get_speech_timestamps(
    audio: torch.Tensor,
    model: _VadModelLike,
    **kwargs: object,
) -> list[object]:
    silero_vad = import_module("silero_vad")
    get_timestamps = cast(
        _GetSpeechTimestamps, getattr(silero_vad, "get_speech_timestamps")
    )
    return get_timestamps(audio, model, **kwargs)


load_silero_vad = _load_silero_vad_model
get_speech_timestamps = _get_speech_timestamps


class VADEngine:
    """Voice Activity Detection using Silero VAD.

    Usage:
        vad = VADEngine(agent_config)
        if vad.is_speech_detected(audio_array):
            segments = vad.get_speech_segments(audio_array)
    """

    def __init__(self, vad_config: Config) -> None:
        """Args:
        vad_config: AgentConfig (or any object with sample_rate,
                    min_speech_duration_ms, min_silence_duration_ms,
                    stt_language, stt_model_size, faster_stt_model_size,
                    cpu_cores).

        """
        super().__init__()
        self.config: Config = vad_config
        self.model: _VadModelLike = load_silero_vad()
        self._apply_platform_tuning()

    # ------------------------------------------------------------------
    # Platform tuning
    # ------------------------------------------------------------------

    def _apply_platform_tuning(self) -> None:
        """Adjust model sizes and CPU limits for Raspberry Pi vs desktop."""
        if detect_raspberry_pi_model():
            limit_cpu_for_multiprocessing(self.config.cpu_cores)
            self._suffix_model_size("stt_model_size")
            self._suffix_model_size("faster_stt_model_size")
        else:
            limit_cpu_for_multiprocessing()  # use all cores
            self.config.stt_model_size = "base"

    def _suffix_model_size(self, attr: str) -> None:
        """Append '.en' suffix when language is English, if not already present."""
        value = cast(str, getattr(self.config, attr))
        if value in ENGLISH_ONLY_COMPATIBLE and not value.endswith(".en"):
            suffix = ".en" if self.config.stt_language == "en" else ""
            setattr(self.config, attr, f"{value}{suffix}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_speech_detected(self, audio_data: NDArray[np.float32]) -> bool:
        """Return True if the audio buffer contains speech."""
        if len(audio_data) < self.config.sample_rate // 4:  # need ≥ 250ms
            return False
        try:
            audio_tensor: Tensor = torch.as_tensor(data=audio_data, dtype=torch.float32)
            timestamps: list[object] = get_speech_timestamps(
                audio=audio_tensor,
                model=self.model,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
            )
            return bool(timestamps)
        except (RuntimeError, ValueError):
            return False

    def get_speech_segments(
        self, audio_data: NDArray[np.float32]
    ) -> list[dict[str, object]]:
        """Return detailed speech segments with timestamps (in seconds)."""
        try:
            audio_tensor: Tensor = torch.as_tensor(data=audio_data, dtype=torch.float32)
            return cast(
                list[dict[str, object]],
                get_speech_timestamps(
                    audio=audio_tensor,
                    model=self.model,
                    min_speech_duration_ms=self.config.min_speech_duration_ms,
                    min_silence_duration_ms=self.config.min_silence_duration_ms,
                    return_seconds=True,
                ),
            )
        except (RuntimeError, ValueError):
            return []
