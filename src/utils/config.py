"""Configuration for the voice agent."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, Field

from .sysutils import detect_raspberry_pi_model, limit_cpu_for_multiprocessing

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
USER_DIR = Path.home()


class PathConfig(BaseModel):
    """Configuration for path settings."""

    src: str = "src"
    data: str = "data"
    cache: str = ".cache"
    models: str = "models"

    @property
    def src_path(self) -> Path:
        """Returns the path to the source directory."""
        return ROOT_DIR / self.src

    @property
    def data_path(self) -> Path:
        """Returns the path to the data directory."""
        return ROOT_DIR / self.data

    @property
    def cache_path(self) -> Path:
        """Returns the path to the cache directory."""
        return ROOT_DIR / self.cache

    @property
    def models_path(self) -> Path:
        """Returns the path to the models directory."""
        return self.cache_path / self.models

    @property
    def models_audio_path(self) -> Path:
        """Returns the path to the models directory."""
        return self.cache_path / "audio" / self.models

    @property
    def models_vision_path(self) -> Path:
        """Returns the path to the models directory."""
        return self.cache_path / "vision" / self.models


class ASRConfig(PathConfig):
    """Configuration for ASR (Automatic Speech Recognition / Speech-to-Text) settings."""

    engine: str = "faster-whisper"
    model_size: str = "tiny"
    faster_model_size: str = "small"
    language: str = "en"
    translate: bool = False
    transformers: bool = False
    transformers_engine: str = "huggingface"
    download_root: str | None = None
    device: str = "cpu"  # Pi5 : CPU (or "hailo")
    compute_type: str = "int8"  # INT8 = 2x plus rapide sur ARM
    skip_native_teardown: bool = False

    @property
    def download_path(self) -> Path:
        """Returns the download path for the ASR model."""
        if self.download_root:
            p = ROOT_DIR / self.download_root
            # Avoid doubling engine name if already in path
            if p.name == self.engine:
                return p
            return p / self.engine
        return self.models_audio_path / (self.transformers_engine if self.transformers else self.engine)

    @property
    def full_model_path(self) -> Path:
        """Returns the full path to the ASR model."""
        return self.download_path / f"{self.language}-{self.model_size}.onnx"


class TTSConfig(PathConfig):
    """Configuration for TTS (Text-to-Speech) settings."""

    engine: str = "piper"
    model_name: str = "en_US-hfc_female-medium.onnx"
    model_path: str | None = None
    cli_mode: bool = False
    device: str = "cpu"  # Pi5 : CPU (or "hailo")
    length_scale: float = 1.0  # speed control: <1.0 slower, >1.0 faster
    noise_scale: float = 1.0  # more audio variation
    noise_w_scale: float = 1.0  # more speaking variation
    normalize_audio: bool = True  # use raw audio from voice
    speed: float = Field(default=1.0, gt=0)  # alias for length_scale
    volume: float = 0.5  # output volume level

    @property
    def full_model_path(self) -> Path:
        """Returns the full path to the TTS model."""
        if self.model_path:
            p = ROOT_DIR / self.model_path
            # If it's already a file path, return it
            if p.suffix in {".onnx", ".bin", ".pt"}:
                return p
            return p / self.engine / self.model_name
        return self.models_audio_path / self.engine / self.model_name


class WakeConfig(PathConfig):
    """Configuration for Wake Word detection settings."""

    wake_word: str = "hey_jarvis"
    model_name: str = "hey_jarvis"
    model_path: str | None = None
    inference_framework: str = "onnx"  # or "pytorch" if using a PyTorch model
    threshold: float = 0.4
    cooldown_seconds: float = 2.0  # minimum seconds between detections
    download_root: str | None = None
    noise_suppression: bool = False
    vad_threshold: float = 0.6

    @property
    def download_path(self) -> Path:
        """Returns the download path for the ASR model."""
        if self.download_root:
            p = ROOT_DIR / self.download_root
            # Avoid doubling engine name if already in path
            if p.suffix in {".onnx", ".tflite"}:
                return p
            return p / "wakeword"
        return self.models_audio_path / "wakeword"

    @property
    def full_model_path(self) -> Path:
        """Returns the full path to the wakeword model."""
        return self.download_path / f"{self.model_name}.{self.inference_framework}"


class VADConfig(PathConfig):
    """Configuration for Voice Activity Detection (VAD) settings."""

    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 500
    silence_timeout_seconds: int = 1
    max_recording_seconds: int = 15
    threshold: float = 0.6
    sample_rate: int = 16000


class AudioConfig(PathConfig):
    """Configuration for Audio Input/Output settings."""

    input_sample_rate: int = 22050
    input_chunk_ms: int = 30  # taille des chunks audio en ms
    input_chunk_size: int = 500
    input_device_index: int | None = None  # None = périphérique système par défaut
    volume: float = 0.5  # half as loud
    output_device_index: int | None = None  # None = périphérique système par défaut
    output_sample_rate: int = 22050
    output_chunk_ms: int = 30  # taille des chunks audio en ms
    output_chunk_size: int = 500
    output_device_name: str | None = None  # Optional name of output device to select (overrides index if found)


class PlatformConfig(PathConfig):
    """Configuration for platform-specific tuning."""

    cpu_cores: int | None = 2  # Limit CPU cores for multiprocessing on Pi5
    pi: bool | None = False  # Automatically detect Raspberry Pi and apply tuning

    def is_raspberry_pi(self) -> bool:
        """Detect whether the system is running on a Raspberry Pi 5.

        Returns:
            `True` when the host appears to be a Raspberry Pi 5.

        """
        self.pi = detect_raspberry_pi_model()
        return self.pi

    def cpu_limit(self) -> int:
        """Set and return the CPU core limit for multiprocessing.

        Returns:
            The CPU core limit selected for multiprocessing.

        """
        self.cpu_cores = limit_cpu_for_multiprocessing(self.cpu_cores)
        return self.cpu_cores

    def __post_init__(self) -> None:
        """Apply platform-specific tuning after initialization."""
        self.is_raspberry_pi()
        self.cpu_limit()


class Config:
    """Configuration for the voice agent."""

    def __init__(self, **data: object) -> None:
        """Build a configuration object from keyword data."""
        super().__init__()
        self.paths = PathConfig.model_validate(data.get("paths", {}))
        self.asr = ASRConfig.model_validate(data.get("asr", {}))
        self.tts = TTSConfig.model_validate(data.get("tts", {}))
        self.wake = WakeConfig.model_validate(data.get("wake", {}))
        self.vad = VADConfig.model_validate(data.get("vad", {}))
        self.audio = AudioConfig.model_validate(data.get("audio", {}))
        self.platform = PlatformConfig.model_validate(data.get("platform", {}))

    @staticmethod
    def from_mapping(data: Mapping[str, object]) -> Config:
        """Build a config object from a string-keyed mapping."""
        return Config(**{str(section_key): section_value for section_key, section_value in data.items()})

    @property
    def cpu_cores(self) -> int | None:
        """CPU cores limit for multiprocessing."""
        return self.platform.cpu_cores

    @property
    def sample_rate(self) -> int:
        """Sample rate for VAD."""
        return self.vad.sample_rate

    @property
    def min_speech_duration_ms(self) -> int:
        """Minimum speech duration in ms."""
        return self.vad.min_speech_duration_ms

    @property
    def min_silence_duration_ms(self) -> int:
        """Minimum silence duration in ms."""
        return self.vad.min_silence_duration_ms

    @property
    def stt_language(self) -> str:
        """STT language."""
        return self.asr.language

    @property
    def stt_model_size(self) -> str:
        """STT model size."""
        return self.asr.model_size

    @stt_model_size.setter
    def stt_model_size(self, value: str) -> None:
        self.asr.model_size = value

    @property
    def faster_stt_model_size(self) -> str:
        """Faster STT model size."""
        return self.asr.faster_model_size

    @faster_stt_model_size.setter
    def faster_stt_model_size(self, value: str) -> None:
        self.asr.faster_model_size = value


def load_config(config_path: Path | None = None) -> Config:
    """Load the configuration from a YAML file.

    Returns:
        The parsed voice-agent configuration.

    Raises:
        ValueError: If the config path resolves outside the allowed roots.

    """
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"

    # Security: Resolve paths and validate that the config is within allowed directories
    try:
        abs_config_path = config_path.resolve()
        abs_root_dir = ROOT_DIR.resolve()
        abs_user_dir = USER_DIR.resolve()

        # Check if the config path is within the project root or user directory
        is_safe = (
            abs_config_path in {abs_root_dir, abs_user_dir}
            or abs_root_dir in abs_config_path.parents
            or abs_user_dir in abs_config_path.parents
        )

        if not is_safe:
            msg = f"Security error: Configuration path {config_path} is outside allowed directories."
            raise ValueError(msg)
    except (OSError, RuntimeError):
        # If path cannot be resolved, but we are trying to open it, that's a risk.
        # However, if it doesn't exist, the .exists() check below handles the UI.
        # We only block if we CAN resolve it and it's unsafe.
        pass

    if not config_path.exists():
        return Config()

    # Safe to open: path has been validated above to be within allowed directories
    # nosec B301 - Path validation prevents file inclusion attacks
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = cast("object", yaml.safe_load(f))

    if config_dict is None:
        return Config()

    if isinstance(config_dict, Mapping):
        return Config.from_mapping(cast("Mapping[str, object]", config_dict))

    return Config()


# Global config instance
config: Config = load_config()


# Helper to ensure src is in sys.path
def setup_python_path() -> None:
    """Add the src directory to `sys.path` so local imports work."""
    src_path = str(config.paths.src_path)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Also add root for imports like utils.sysutils if src is not the package root
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
