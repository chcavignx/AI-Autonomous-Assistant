#!/usr/bin/env python3
"""Script to download and extract Vosk speech recognition models
to a local cache directory."""

import zipfile

import requests
from models_check import model_exists

from src.utils.config import config

# Model URLs
MODELS = {
    "en-us": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr-pguyot": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
}


# Target directory
CACHE_DIR = config.paths.models_path / "vosk"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract(model_name, url) -> None:
    """Downloads and extracts a Vosk model."""
    filename = url.split("/")[-1]
    filepath = CACHE_DIR / filename
    print(f"Downloading {model_name} model...")

    # Download the file
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with filepath.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename} successfully")

    # Extract the file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(CACHE_DIR)
    print(f"Extracted {filename} successfully")

    # Remove the zip file
    filepath.unlink()
    print(f"Removed {filename}")


def run() -> None:
    """Downloads and extracts all Vosk models."""
    for model_name, url in MODELS.items():
        print(f"Processing {model_name}...")
        if model_exists(model_name, CACHE_DIR):
            print(f"Model '{model_name}' already exists, skipping download.")
        else:
            download_and_extract(model_name, url)
        print("---")
    print("All vosk models processed!")


if __name__ == "__main__":
    run()
