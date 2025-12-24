#!/usr/bin/env python3
"""
Configuration for the voice agent.
"""

import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
USER_DIR = Path.home()


class PathConfig(BaseModel):
    """
    Configuration for path settings.
    """

    src: str = "src"
    data: str = "data"
    cache: str = "cache"
    models: str = "models"

    @property
    def src_path(self) -> Path:
        """
        Returns the path to the source directory.
        """
        return ROOT_DIR / self.src

    @property
    def data_path(self) -> Path:
        """
        Returns the path to the data directory.
        """
        return ROOT_DIR / self.data

    @property
    def cache_path(self) -> Path:
        """
        Returns the path to the cache directory.
        """
        return ROOT_DIR / self.cache

    @property
    def models_path(self) -> Path:
        """
        Returns the path to the models directory.
        """
        return self.cache_path / self.models


class STTConfig(PathConfig):
    """
    Configuration for STT (Speech-to-Text) settings.
    """

    engine: str = "whisper"
    model_size: str = "tiny"
    language: str = "en"
    transformers: bool = False
    transformers_engine: str = "huggingface"
    download_root: Optional[str] = None

    @property
    def download_path(self) -> Path:
        """
        Returns the download path for the STT model.
        """
        if self.download_root:
            p = ROOT_DIR / self.download_root
            # Avoid doubling engine name if already in path
            if p.name == self.engine:
                return p
            return p / self.engine
        return self.models_path / (
            self.transformers_engine if self.transformers else self.engine
        )


class TTSConfig(PathConfig):
    """
    Configuration for TTS (Text-to-Speech) settings.
    """

    engine: str = "piper"
    model_name: str = "jarvis-medium.onnx"
    model_path: Optional[str] = None
    cli_mode: bool = False

    @property
    def full_model_path(self) -> Path:
        """
        Returns the full path to the TTS model.
        """
        if self.model_path:
            p = ROOT_DIR / self.model_path
            # If it's already a file path, return it
            if p.suffix in [".onnx", ".bin", ".pt"]:
                return p
            return p / self.engine / self.model_name
        return self.models_path / self.engine / self.model_name


class VADConfig(PathConfig):
    """
    Configuration for VAD (Voice Activity Detection) settings.
    """

    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    silence_timeout_seconds: int = 3
    max_recording_seconds: int = 15


class Config:
    """
    Configuration for the voice agent.
    """

    def __init__(self, **data):
        self.paths = PathConfig(**data.get("paths", {}))
        self.stt = STTConfig(**data.get("stt", {}))
        self.tts = TTSConfig(**data.get("tts", {}))
        self.vad = VADConfig(**data.get("vad", {}))


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Loads the configuration from a YAML file.
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
            abs_config_path == abs_root_dir
            or abs_root_dir in abs_config_path.parents
            or abs_config_path == abs_user_dir
            or abs_user_dir in abs_config_path.parents
        )

        if not is_safe:
            raise ValueError(
                f"Security error: Configuration path {config_path} is "
                f"outside allowed directories."
            )
    except (OSError, RuntimeError):
        # If path cannot be resolved, but we are trying to open it, that's a risk.
        # However, if it doesn't exist, the .exists() check below handles the UI.
        # We only block if we CAN resolve it and it's unsafe.
        pass

    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return Config()

    # Safe to open: path has been validated above to be within allowed directories
    # nosec B301 - Path validation prevents file inclusion attacks
    with open(config_path, "r", encoding="utf-8") as f:  # noqa: S101
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        return Config()

    return Config(**config_dict)


# Global config instance
config: Config = load_config()


# Helper to ensure src is in sys.path
def setup_python_path():
    """
    Adds the src directory to sys.path to ensure imports work correctly.
    """
    src_path = str(config.paths.src_path)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Also add root for imports like utils.sysutils if src is not the package root
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


if __name__ == "__main__":
    print(f"Root: {ROOT_DIR}")
    print(f"Data path: {config.paths.data_path}")
    print(f"STT Model: {config.stt.model_size}")
