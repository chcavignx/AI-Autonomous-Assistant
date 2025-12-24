#!/usr/bin/env python3
import os

from huggingface_hub import snapshot_download
from models_check import model_exists

from src.utils.sysutils import detect_raspberry_pi_model

MODELS_NAMES = (
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-small.en",
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-distil-whisper-small.en",
)
# Add larger models if not on Raspberry Pi
if not detect_raspberry_pi_model():
    MODELS_NAMES += (
        "Systran/faster-whisper-base",
        "Systran/faster-whisper-base.en",
        "Systran/faster-whisper-medium",
        "Systran/faster-whisper-medium.en",
        "Systran/faster-whisper-large-v3",
        "Systran/faster-distil-whisper-large-v3",
    )

cache_dir = os.path.join(os.path.expanduser("~"), "cache/models/huggingface")


def run() -> None:
    """Downloads and saves Hugging Face models, tokenizers, processors,
    and their associated datasets to a local backup in your user cache directory."""
    for model_name in MODELS_NAMES:
        if model_exists(model_name, cache_dir):
            print(f"Model {model_name} already exists.")
            continue
        print(f"Downloading and saving {model_name} to {cache_dir}")

        snapshot_download(repo_id=model_name, repo_type="model", cache_dir=cache_dir)
        print(f"Model saved to: {os.path.join(cache_dir, model_name)}")
    print("All fast-whisper models have been downloaded and saved.")


if __name__ == "__main__":
    run()
