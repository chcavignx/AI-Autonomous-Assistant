#!/usr/bin/env python3
"""Script to download and save Hugging Face models, tokenizers, processors,
and their associated datasets to a local backup in your user cache directory.
"""

import pathlib
import sys

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from huggingface_hub import snapshot_download
from models_check import model_exists

from src.utils.config import config
from src.utils.sysutils import detect_raspberry_pi_model

MODELS_NAMES_BASE = (
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-small.en",
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-distil-whisper-small.en",
)

MODELS_NAMES_EXTENDED = (
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-base.en",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-medium.en",
    "Systran/faster-whisper-large-v3",
    "Systran/faster-distil-whisper-large-v3",
)

CACHE_DIR = str(config.asr.download_path)


def get_models_to_download() -> tuple:
    """Select which Hugging Face model identifiers should be downloaded for the current platform.

    Returns:
        tuple: Tuple of model identifier strings — on Raspberry Pi this is the base models tuple, otherwise the base models concatenated with the extended models tuple.

    """
    # Add larger models if not on Raspberry Pi
    if not detect_raspberry_pi_model():
        return MODELS_NAMES_BASE + MODELS_NAMES_EXTENDED
    return MODELS_NAMES_BASE


def run() -> None:
    """Download the selected Hugging Face models and store them in the user's local cache.

    Selects models appropriate for the current platform, skips models that are already present in the cache, downloads any missing models into the configured cache directory, and prints progress messages for each model.
    """
    models_to_download = get_models_to_download()
    for model_name in models_to_download:
        if model_exists(model_name, CACHE_DIR):
            continue

        snapshot_download(repo_id=model_name, repo_type="model", cache_dir=CACHE_DIR)


if __name__ == "__main__":
    run()
