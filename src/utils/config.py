"""Configuration for the voice agent."""

from __future__ import annotations

import json
import logging.config
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

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
    tmp: str = ".tmp"

    @property
    def src_path(self) -> Path:
        """Get the path to the source directory."""
        return ROOT_DIR / self.src

    @property
    def data_path(self) -> Path:
        """Get the path to the data directory."""
        return ROOT_DIR / self.data

    @property
    def cache_path(self) -> Path:
        """Get the path to the cache directory."""
        return ROOT_DIR / self.cache

    @property
    def models_path(self) -> Path:
        """Get the path to the models directory."""
        return self.cache_path / self.models

    @property
    def models_audio_path(self) -> Path:
        """Get the path to the models directory."""
        return self.cache_path / "audio" / self.models

    @property
    def models_vision_path(self) -> Path:
        """Get the path to the models directory."""
        return self.cache_path / "vision" / self.models

    @property
    def dataset_vision_path(self) -> Path:
        """Get the path to the models directory."""
        return self.cache_path / "vision" / self.data

    @property
    def tmp_path(self) -> Path:
        """Get the path to the tmp directory."""
        return ROOT_DIR / self.tmp


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
    compute_type: str = "int8"  # INT8 = 2x faster on ARM
    skip_native_teardown: bool = False
    store_audio: bool = False
    store_audio_path: str | None = None

    @property
    def download_path(self) -> Path:
        """Get the download path for the ASR model."""
        if self.download_root:
            p = ROOT_DIR / self.download_root
            if p.name == self.engine:
                return p.resolve()
            return (p / self.engine).resolve()
        return self.models_audio_path / (self.transformers_engine if self.transformers else self.engine)

    @property
    def full_model_path(self) -> Path:
        """Get the full path to the ASR model."""
        return self.download_path / f"{self.language}-{self.model_size}.onnx"


class TTSConfig(PathConfig):
    """Configuration for TTS (Text-to-Speech) settings."""

    engine: str = "piper"
    model_name: str = "jarvis-medium.onnx"
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
        """Get the full path to the TTS model."""
        if self.model_path:
            p = ROOT_DIR / self.model_path
            # If it's already a file path, return it
            if p.suffix in {".onnx", ".bin", ".pt", ".tflite"}:
                return p.resolve()
            return (p / self.engine / self.model_name).resolve()
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
    melspec_model: str = "melspectrogram"
    embedding_model: str = "embedding_model"
    silero_vad_model: str = "silero_vad"
    backend: str = "wakeword"

    @property
    def download_path(self) -> Path:
        """Get the download path for the wakeword model."""
        if self.download_root:
            return (ROOT_DIR / self.download_root).resolve()
        return self.models_audio_path / "wakeword"

    @property
    def full_model_path(self) -> Path:
        """Get the full path to the wakeword model."""
        return self.download_path / f"{self.model_name}.{self.inference_framework}"

    @property
    def embedding_model_path(self) -> Path:
        """Get the full path to the embedding model."""
        return self.download_path / f"{self.embedding_model}.{self.inference_framework}"

    @property
    def melspec_model_path(self) -> Path:
        """Get the full path to the melspec model."""
        return self.download_path / f"{self.melspec_model}.{self.inference_framework}"

    @property
    def silero_vad_model_path(self) -> Path:
        """Get the full path to the silero vad model."""
        return self.download_path / f"{self.silero_vad_model}.{self.inference_framework}"


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
    input_chunk_ms: int = 30  # chunks size for audio in ms
    input_chunk_size: int = 500
    input_device_index: int | None = None  # None = default input device
    input_device_name: str | None = None  # Optional name of input device to select (overrides index if found)
    volume: float = 0.5  # half as loud
    output_device_index: int | None = None  # None = default output device
    output_sample_rate: int = 22050
    output_chunk_ms: int = 30  # chunks size for audio in ms
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
        _ = self.is_raspberry_pi()
        _ = self.cpu_limit()


class CameraConfig(BaseModel):
    """Configuration for Camera settings."""

    camera_index: int = 0
    frame_width: int = 1080
    frame_height: int = 720
    format: str = "RGB888"  # "YUV420" or "XRGB8888"
    lores_frame_width: int = 640
    lores_frame_height: int = 480
    lores_format: str = "YUV420"
    imx500_frame_width: int = 640
    imx500_frame_height: int = 480


class VisionConfig(PathConfig):
    """Configuration for Vision settings."""

    model_config: ClassVar[dict[str, Any]] = {"populate_by_name": True}  # pyright: ignore[reportUndefinedVariable]

    face_detector_type: str = "cascade"  # "cascade", "insightface", "imx500"
    face_model_name: str = "haarcascade_frontalface_default.xml"  # "buffalo_l" for insightface
    face_model_path: str | None = None
    face_recognition_threshold: float = 0.4
    post_processing_enabled: bool = False
    post_processing_model_name: str = "arcface_r100_v1.onnx"  # "buffalo_l" for insightface
    post_processing_model_type: str = (
        "insightface"  # "insightface" (for openvino arcface_r100_v1.onnx), "buffalo_l" for buffalo_l, "hailo" for hailo
    )
    post_processing_model_path: str | None = None
    post_processing_image_size: int = 640
    post_processing_inference_framework: str = "onnx"  # "hef" for Hailo
    object_model_type: str = "yolo"  # "yolo" (CPU), "yolo_hailo", "yolo_imx500", "libreyolo" (CPU)
    object_model_name: str = "yolo26n.onnx"
    object_model_path: str | None = None
    object_device: str = "cpu"  # CPU or "hailo"
    object_inference_framework: str = "onnx"  # "hef" for Hailo
    object_nms: bool = False  # Enable NMS in Python (after export)
    object_image_size: int = 640  # Image size for YOLO model
    object_recognition_threshold: float = 0.25
    enable_face_detection: bool = True
    enable_face_recognition: bool = True
    enable_object_detection: bool = True
    face_dataset_path: str | None = None
    object_dataset_path: str | None = None

    camera: CameraConfig = CameraConfig()

    @property
    def object_model_full_path(self) -> Path:
        """Get the resolved full path to the object detection model."""
        raw_path = self.object_model_path  # Access Pydantic field value
        if raw_path:
            p = ROOT_DIR / raw_path
            if p.suffix in {".onnx", ".pt", ".hef", ".rpk"}:
                return p.resolve()
            return (p / self.object_model_type / self.object_model_name).resolve()
        base_name = (
            Path(self.object_model_name).stem
            if Path(self.object_model_name).suffix in {".onnx", ".pt", ".hef", ".rpk"}
            else self.object_model_name
        )
        if self.object_inference_framework in {"onnx", "hef", "rpk"}:
            return (
                self.models_vision_path / self.object_model_type / f"{base_name}.{self.object_inference_framework}"
            ).resolve()
        if self.object_inference_framework == "ncnn":
            return (self.models_vision_path / self.object_model_type / f"{base_name}_ncnn_model").resolve()
        msg = f"Unknown inference framework: {self.object_inference_framework}"
        raise ValueError(msg)

    @property
    def face_detector_model_path(self) -> Path:
        """Get the full path to the detector model."""
        if self.face_model_path:
            p = ROOT_DIR / self.face_model_path
            if p.suffix in {".onnx", ".hef", ".rpk"}:
                return p.resolve()
        return self.models_vision_path / self.face_detector_type / self.face_model_name

    @property
    def post_processing_model_full_path(self) -> Path:
        """Get the full path to the post-processing model."""
        raw_path = self.post_processing_model_path
        if raw_path:
            p = ROOT_DIR / raw_path
            if p.suffix in {".onnx", ".hef"}:
                return p.resolve()
            return (p / self.post_processing_model_type / self.post_processing_model_name).resolve()
        return (self.models_vision_path / self.post_processing_model_type / self.post_processing_model_name).resolve()


class LLMConfig(BaseModel):
    """Configuration for local or cloud LLM integration."""

    api_type: str = "ollama"
    model: str = "llama3.2:latest"
    url: str = "http://127.0.0.1:11434/api/generate"
    timeout: float = 5.0
    api_key: str | None = None


class Config:
    """Configuration for the voice agent."""

    paths: PathConfig
    asr: ASRConfig
    tts: TTSConfig
    wake: WakeConfig
    vad: VADConfig
    audio: AudioConfig
    platform: PlatformConfig
    vision: VisionConfig
    llm: LLMConfig

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
        self.vision = VisionConfig.model_validate(data.get("vision", {}))
        self.llm = LLMConfig.model_validate(data.get("llm", {}))

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
    is_safe = True
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
    except (OSError, RuntimeError):
        # If path cannot be resolved, but we are trying to open it, that's a risk.
        # However, if it doesn't exist, the .exists() check below handles the UI.
        # We only block if we CAN resolve it and it's unsafe.
        pass

    if not is_safe:
        msg = f"Security error: Configuration path {config_path} is outside allowed directories."
        raise ValueError(msg)

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


def setup_config_logging() -> None:
    """Set up the logging configuration module."""
    log_file = ROOT_DIR / "log.json"

    log_path = Path(config.paths.tmp + "/filename.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(log_file).open("r", encoding="utf-8") as f:
        logging.config.dictConfig(json.load(f))


setup_python_path()
setup_config_logging()
