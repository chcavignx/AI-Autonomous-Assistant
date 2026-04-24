#!/usr/bin/env python3
"""Script to download and extract Vosk speech recognition models
to a local cache directory.
"""

import zipfile

import pathlib
import sys

import requests
from models_check import model_exists

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import config

# Model URLs
MODELS = {
    "en-us": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr-pguyot": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
}


# Target directory
CACHE_DIR = config.paths.models_audio_path / "vosk"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract(model_name, url) -> None:
    """Downloads and extracts a Vosk model."""
    filename = url.split("/")[-1]
    filepath = CACHE_DIR / filename

    # Download the file
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with filepath.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the file
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(CACHE_DIR)

    # Remove the zip file
    filepath.unlink()


def run() -> None:
    """Downloads and extracts all Vosk models."""
    for model_name, url in MODELS.items():
        if model_exists(model_name, CACHE_DIR):
            pass
        else:
            download_and_extract(model_name, url)


if __name__ == "__main__":
    run()
