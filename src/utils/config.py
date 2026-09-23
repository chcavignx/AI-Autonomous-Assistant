"""Configuration for the voice agent."""

from __future__ import annotations

import json
import logging.config
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .sysutils import detect_raspberry_pi_model, limit_cpu_for_multiprocessing

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
USER_DIR = Path.home()

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


class PathConfig(BaseModel):
    """Configuration for path settings."""

    src: str = "src"
    data: str = "data"
    cache: str = ".cache"
    models: str = "models"
    tmp: str = ".tmp"
    system: str = "/usr/share"

    @property
    def src_path(self) -> Path:
        """The path to the source directory."""
        return ROOT_DIR / self.src

    @property
    def data_path(self) -> Path:
        """The path to the data directory."""
        return ROOT_DIR / self.data

    @property
    def cache_path(self) -> Path:
        """The path to the cache directory."""
        return ROOT_DIR / self.cache

    @property
    def models_path(self) -> Path:
        """The path to the models directory."""
        return self.cache_path / self.models

    @property
    def models_audio_path(self) -> Path:
        """The path to the models directory."""
        return self.cache_path / "audio" / self.models

    @property
    def models_vision_path(self) -> Path:
        """The path to the models directory."""
        return self.cache_path / "vision" / self.models

    @property
    def dataset_vision_path(self) -> Path:
        """The path to the models directory."""
        return self.cache_path / "vision" / self.data

    @property
    def tmp_path(self) -> Path:
        """The path to the tmp directory."""
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
        """The download path for the ASR model."""
        if self.download_root:
            p = ROOT_DIR / self.download_root
            if p.name == self.engine:
                return p.resolve()
            return (p / self.engine).resolve()
        return self.models_audio_path / (self.transformers_engine if self.transformers else self.engine)

    @property
    def full_model_path(self) -> Path:
        """The full path to the ASR model."""
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
        """The full path to the TTS model."""
        if self.model_path:
            p = Path(self.model_path)
            if p.suffix in {".onnx", ".bin", ".pt", ".tflite"}:
                return p.resolve()
            p = p / self.model_name
            if p.suffix in {".onnx", ".bin", ".pt", ".tflite"}:
                return p.resolve()
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
        """The download path for the wakeword model."""
        if self.download_root:
            return Path(self.download_root).resolve()
        return self.models_audio_path / "wakeword"

    @property
    def full_model_path(self) -> Path:
        """The full path to the wakeword model."""
        return self.download_path / f"{self.model_name}.{self.inference_framework}"

    @property
    def embedding_model_path(self) -> Path:
        """The full path to the embedding model."""
        return self.download_path / f"{self.embedding_model}.{self.inference_framework}"

    @property
    def melspec_model_path(self) -> Path:
        """The full path to the melspec model."""
        return self.download_path / f"{self.melspec_model}.{self.inference_framework}"

    @property
    def silero_vad_model_path(self) -> Path:
        """The full path to the silero vad model."""
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


class ModelResolution(BaseModel):
    """Configuration for model resolution settings."""

    width: int = 640
    height: int = 640


class CameraConfig(BaseModel):
    """Configuration for Camera settings."""

    camera_index: int = 0
    frame_width: int = 1080
    frame_height: int = 720
    format: str = "RGB888"  # "YUV420" or "XRGB8888"
    lores_frame_width: int = 640
    lores_frame_height: int = 480
    lores_format: str = "YUV420"


class VisionConfig(PathConfig):
    """Configuration for Vision settings."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # --- Face Detection & Recognition Configuration ---
    face_detector_type: str = "insightface"  # "cascade", "insightface", "hailo", "imx500"
    face_model_name: str = "buffalo_l"  # "haarcascade_frontalface_default.xml", "buffalo_l", "scrfd_2.5g.hef"
    face_model_path: str | None = None
    face_recognition_threshold: float = 0.4
    face_detector_device: str = "cpu"  # "cpu", "hailo", "imx500"
    face_detector_inference_framework: str = "onnx"  # "onnx", "ncnn", "hef", "rpk"
    enable_face_detection: bool = True
    enable_face_recognition: bool = True
    face_dataset_path: str | None = None

    # --- Face Post-Processing & Embeddings (ArcFace) ---
    post_processing_enabled: bool = False
    post_processing_model_name: str = Field(
        default="buffalo_l",
        validation_alias=AliasChoices("post_processing_model_name", "post_processing_model"),
    )
    post_processing_model_type: str = "insightface"  # "insightface", "hailo"
    post_processing_model_path: str | None = None
    post_processing_image_size: int = 640
    post_processing_model_device: str = "cpu"  # "cpu", "hailo", "imx500"
    post_processing_inference_framework: str = "onnx"  # "onnx", "hef"
    async_face_recognition: bool = True
    face_recognition_interval: float = 1.5

    # --- Object Detection Configuration ---
    object_model_type: str = "yolo"  # "yolo", "yolo_hailo", "yolo_imx500", "libreyolo"
    object_model_name: str = "yolo26n.onnx"
    object_model_path: str | None = None
    object_label_path: str | None = None
    object_device: str = "cpu"  # "cpu", "hailo", "imx500"
    object_inference_framework: str = "onnx"  # "onnx", "ncnn", "hef"
    object_nms: bool = False  # Enable NMS in Python (after export)
    object_image_size: int = 640  # Image size for YOLO model
    object_recognition_threshold: float = 0.25
    enable_object_detection: bool = True
    enable_object_recognition: bool = True
    object_dataset_path: str | None = None

    # --- Object Post-Processing ---
    object_post_processing_enabled: bool = False
    object_post_processing_model_name: str = "buffalo_l"
    object_post_processing_model_type: str = "insightface"
    object_post_processing_model_device: str = "cpu"  # "cpu", "hailo", "imx500"
    object_post_processing_model_path: str | None = None
    object_post_processing_image_size: int = 640
    object_post_processing_inference_framework: str = "onnx"

    # --- Resolution & Camera Configuration ---
    model_resolutions: dict[str, ModelResolution] = Field(
        default_factory=lambda: {
            "LibreYOLOXn.onnx": ModelResolution(width=416, height=416),
            "LibreYOLOXn": ModelResolution(width=416, height=416),
            "yolo26n.onnx": ModelResolution(width=640, height=640),
            "yolo26n": ModelResolution(width=640, height=640),
            "yolo11n.onnx": ModelResolution(width=640, height=640),
            "yolo11n": ModelResolution(width=640, height=640),
            "yolov8n.hef": ModelResolution(width=640, height=640),
            "yolov8s_h8l.hef": ModelResolution(width=640, height=640),
            "yolo11n.hef": ModelResolution(width=640, height=640),
            "scrfd_2.5g.hef": ModelResolution(width=640, height=640),
            "scrfd_2.5g_h8l.hef": ModelResolution(width=640, height=640),
            "buffalo_l": ModelResolution(width=640, height=640),
            "imx500": ModelResolution(width=640, height=480),
            "hailo": ModelResolution(width=640, height=640),
            "default": ModelResolution(width=1080, height=720),
        }
    )

    camera: CameraConfig = CameraConfig()

    def get_model_resolution(self, model_name: str | None = None) -> tuple[int, int]:
        """Resolve (width, height) resolution for a specific model or current vision model setting.

        Args:
            model_name: Optional model name string. If None, checks object_model_name, face_model_name,
                        or object_model_type/face_detector_type.


        Returns:
            Tuple of (width, height).

        """
        candidates: list[str] = []
        if model_name:
            candidates.append(model_name)
        else:
            if self.object_model_name:
                candidates.append(self.object_model_name)
            if self.face_model_name:
                candidates.append(self.face_model_name)
            if self.object_model_type:
                candidates.append(self.object_model_type)
            if self.face_detector_type:
                candidates.append(self.face_detector_type)

        # 1. Exact match
        for candidate in candidates:
            if candidate in self.model_resolutions:
                res = self.model_resolutions[candidate]
                return res.width, res.height

        # 2. Case-insensitive exact match
        for candidate in candidates:
            cand_lower = candidate.lower()
            for key, res in self.model_resolutions.items():
                if key.lower() == cand_lower:
                    return res.width, res.height

        # 3. Stem match (e.g. 'yolo26n.onnx' -> 'yolo26n')
        for candidate in candidates:
            stem = Path(candidate).stem.lower()
            for key, res in self.model_resolutions.items():
                if Path(key).stem.lower() == stem:
                    return res.width, res.height

        # 4. Fallback to default in table if available, else camera config defaults
        if "default" in self.model_resolutions:
            res = self.model_resolutions["default"]
            return res.width, res.height
        width = getattr(self.camera, "frame_width", 1080)
        height = getattr(self.camera, "frame_height", 720)
        return width, height

    @property
    def object_model_full_path(self) -> Path:
        """The resolved full path to the object detection model."""
        raw_path = self.object_model_path
        if raw_path:
            p = Path(raw_path)
            if p.suffix in {".onnx", ".pt", ".hef", ".rpk"}:
                return p.resolve() if p.is_absolute() else (ROOT_DIR / p).resolve()

        base_name = (
            Path(self.object_model_name).stem
            if Path(self.object_model_name).suffix in {".onnx", ".pt", ".hef", ".rpk"}
            else self.object_model_name
        )
        base_dir = (
            (Path(raw_path).resolve() if Path(raw_path).is_absolute() else (ROOT_DIR / raw_path).resolve())
            if raw_path
            else self.models_vision_path
        )

        if self.object_inference_framework in {"onnx", "hef", "rpk"}:
            cand_device = (
                base_dir
                / self.object_model_type
                / self.object_device
                / f"{base_name}.{self.object_inference_framework}"
            )
            if cand_device.exists():
                return cand_device.resolve()
            return (base_dir / self.object_model_type / f"{base_name}.{self.object_inference_framework}").resolve()
        if self.object_inference_framework == "ncnn":
            cand_device = base_dir / self.object_model_type / self.object_device / f"{base_name}_ncnn_model"
            if cand_device.exists():
                return cand_device.resolve()
            return (base_dir / self.object_model_type / f"{base_name}_ncnn_model").resolve()
        msg = f"Unable to resolve object model path for inference framework: {self.object_model_name}"
        raise ValueError(msg)

    @property
    def object_label_full_path(self) -> Path | None:
        """The resolved full path to the object detection labels file."""
        if self.object_label_path:
            p = Path(self.object_label_path)
            return p.resolve() if p.is_absolute() else (ROOT_DIR / self.object_label_path).resolve()
        return None

    @property
    def face_detector_model_path(self) -> Path:
        """The full path to the detector model."""
        if self.face_model_path:
            p = Path(self.face_model_path)
            if p.suffix in {".onnx", ".hef", ".rpk", ".xml"}:
                return p.resolve()
            p = p / self.face_model_name
            if p.suffix in {".onnx", ".hef", ".rpk", ".xml"}:
                return p.resolve()
        cand_device = (
            self.models_vision_path / self.face_detector_type / self.face_detector_device / self.face_model_name
        )
        if cand_device.exists():
            return cand_device.resolve()
        return (self.models_vision_path / self.face_detector_type / self.face_model_name).resolve()

    @property
    def face_model_full_path(self) -> Path:
        """The full path to the face model."""
        return self.face_detector_model_path

    @property
    def post_processing_model_full_path(self) -> Path:
        """The full path to the post-processing model."""
        raw_path = self.post_processing_model_path
        if raw_path:
            p = Path(raw_path, self.post_processing_model_name)
            if p.suffix in {".onnx", ".hef"}:
                return p.resolve()
        for base in [
            self.models_vision_path,
            ROOT_DIR / ".cache" / "vision" / "models",
            ROOT_DIR / "data" / "models" / "vision",
        ]:
            cand_device = (
                base
                / self.post_processing_model_type
                / self.post_processing_model_device
                / self.post_processing_model_name
            )
            if cand_device.exists():
                return cand_device.resolve()
            cand = base / self.post_processing_model_type / self.post_processing_model_name
            if cand.exists():
                return cand.resolve()
            # If directory or alias is configured (e.g. buffalo_l)
            cand_r50_device = (
                base
                / self.post_processing_model_type
                / self.post_processing_model_device
                / "buffalo_l"
                / "w600k_r50.onnx"
            )
            if cand_r50_device.exists():
                return cand_r50_device.resolve()
            cand_r50 = base / self.post_processing_model_type / "buffalo_l" / "w600k_r50.onnx"
            if cand_r50.exists():
                return cand_r50.resolve()

        return (self.models_vision_path / self.post_processing_model_type / self.post_processing_model_name).resolve()

    @property
    def post_processing_object_model_full_path(self) -> Path:
        """The full path to the post-processing object model."""
        raw_path = self.object_post_processing_model_path
        if raw_path:
            p = Path(raw_path, self.object_post_processing_model_name)
            if p.suffix in {".onnx", ".hef"}:
                return p.resolve()
        for base in [
            self.models_vision_path,
            ROOT_DIR / ".cache" / "vision" / "models",
            ROOT_DIR / "data" / "models" / "vision",
        ]:
            cand_device = (
                base
                / self.object_post_processing_model_type
                / self.object_post_processing_model_device
                / self.object_post_processing_model_name
            )
            if cand_device.exists():
                return cand_device.resolve()
            cand = base / self.object_post_processing_model_type / self.object_post_processing_model_name
            if cand.exists():
                return cand.resolve()
            # If directory or alias is configured (e.g. buffalo_l)
            cand_r50_device = (
                base
                / self.object_post_processing_model_type
                / self.object_post_processing_model_device
                / "buffalo_l"
                / "w600k_r50.onnx"
            )
            if cand_r50_device.exists():
                return cand_r50_device.resolve()
            cand_r50 = base / self.object_post_processing_model_type / "buffalo_l" / "w600k_r50.onnx"
            if cand_r50.exists():
                return cand_r50.resolve()

        return (
            self.models_vision_path / self.object_post_processing_model_type / self.object_post_processing_model_name
        ).resolve()

    @property
    def face_dataset_full_path(self) -> Path | None:
        """Resolved path to face dataset directory."""
        if self.face_dataset_path:
            p = Path(self.face_dataset_path)
            return p.resolve() if p.is_absolute() else (ROOT_DIR / self.face_dataset_path).resolve()
        return None

    @property
    def object_dataset_full_path(self) -> Path | None:
        """Resolved path to object dataset directory."""
        if self.object_dataset_path:
            p = Path(self.object_dataset_path)
            return p.resolve() if p.is_absolute() else (ROOT_DIR / self.object_dataset_path).resolve()
        return None


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


def _deep_merge_dicts(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge_dicts(result[key], cast("Mapping[str, Any]", value))
        else:
            result[key] = value
    return result


def _resolve_subconfig_path(sub_path_str: str, parent_dir: Path) -> Path:
    """Resolve sub-config path relative to parent directory, config/vision, or project root."""
    cand = Path(sub_path_str)
    if cand.is_absolute():
        return cand.resolve()

    cand_parent = (parent_dir / sub_path_str).resolve()
    if cand_parent.exists():
        return cand_parent

    cand_root = (ROOT_DIR / sub_path_str).resolve()
    if cand_root.exists():
        return cand_root

    cand_vision = (ROOT_DIR / "config" / "vision" / sub_path_str).resolve()
    if cand_vision.exists():
        return cand_vision

    return cand_root


def load_config(config_path: Path | None = None) -> Config:
    """Load the configuration from a YAML file (supports vision_config profiles and includes).

    Returns:
        The parsed voice/vision agent configuration.

    Raises:
        ValueError: If the config path resolves outside the allowed roots.

    """
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    # Security: Resolve paths and validate that the config is within allowed directories
    is_safe = True
    try:
        abs_config_path = config_path.resolve()

        import tempfile

        abs_root_dir = ROOT_DIR.resolve()
        abs_user_dir = USER_DIR.resolve()
        temp_dir = Path(tempfile.gettempdir()).resolve()

        # Check if the config path is within the project root, user directory, or system temp
        is_safe = (
            abs_config_path in {abs_root_dir, abs_user_dir, temp_dir}
            or abs_root_dir in abs_config_path.parents
            or abs_user_dir in abs_config_path.parents
            or temp_dir in abs_config_path.parents
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

    if not isinstance(config_dict, Mapping):
        return Config()

    raw_dict: dict[str, Any] = dict(config_dict)

    # 1. Process generic includes: [...]
    merged_root: dict[str, Any] = {}
    includes = raw_dict.pop("includes", None)
    if isinstance(includes, list):
        for inc in includes:
            inc_path = _resolve_subconfig_path(str(inc), config_path.parent)
            if inc_path.exists():
                with inc_path.open("r", encoding="utf-8") as inc_f:
                    inc_data = yaml.safe_load(inc_f)
                    if isinstance(inc_data, Mapping):
                        merged_root = _deep_merge_dicts(merged_root, cast("Mapping[str, Any]", inc_data))

    merged_root = _deep_merge_dicts(merged_root, raw_dict)

    # 2. Process vision_config modular profile (e.g. config/vision/config_vision_hailo.yaml)
    vision_sub = merged_root.pop("vision_config", None)
    if vision_sub and isinstance(vision_sub, str):
        vis_path = _resolve_subconfig_path(vision_sub, config_path.parent)
        if vis_path.exists():
            with vis_path.open("r", encoding="utf-8") as vis_f:
                vis_data = yaml.safe_load(vis_f)
                if isinstance(vis_data, Mapping):
                    inline_vision = merged_root.get("vision", {})
                    if isinstance(inline_vision, Mapping):
                        merged_vision = _deep_merge_dicts(dict(vis_data), cast("Mapping[str, Any]", inline_vision))
                    else:
                        merged_vision = dict(vis_data)
                    merged_root["vision"] = merged_vision

    return Config.from_mapping(cast("Mapping[str, object]", merged_root))


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
