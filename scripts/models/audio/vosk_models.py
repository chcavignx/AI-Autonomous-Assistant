#!/usr/bin/env python3
"""Script to download and save Hugging Face models, tokenizers, processors,
and their associated datasets to a local backup in your user cache directory."""

import os
import zipfile

import requests
from models_check import model_exists

# Model URLs
MODELS = {
    "en-us": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr-pguyot": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
}

# Target directory
cache_dir = os.path.join(os.path.expanduser("~"), "cache/models/vosk")
os.makedirs(cache_dir, exist_ok=True)


def download_and_extract(model_name, url):
    filename = os.path.basename(url)
    filepath = os.path.join(cache_dir, filename)
    print(f"Downloading {model_name} model...")

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename} successfully")

    # Extract the file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(cache_dir)
    print(f"Extracted {filename} successfully")

    # Remove the zip file
    os.remove(filepath)
    print(f"Removed {filename}")


def run():
    for model_name, url in MODELS.items():
        print(f"Processing {model_name}...")
        if model_exists(model_name, cache_dir):
            print(f"Model '{model_name}' already exists, skipping download.")
        else:
            download_and_extract(model_name, url)
        print("---")
    print("All vosk models processed!")


if __name__ == "__main__":
    run()
