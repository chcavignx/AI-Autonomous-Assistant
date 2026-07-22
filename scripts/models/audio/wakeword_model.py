#!/usr/bin/env python3
"""Script to download and extract wakeword models to a local cache directory."""

import pathlib
import sys
import urllib.request

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import config

# Target directory
CACHE_DIR = config.wake.download_path
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def run() -> None:
    models = {
        "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "hey_jarvis_v0.1.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
    }

    for model_name, url in models.items():
        dest = CACHE_DIR / model_name
        if dest.exists():
            print(f"{model_name} already exists at {dest}, skipping.")
        else:
            print(f"Downloading {model_name} from {url}...")
            urllib.request.urlretrieve(url, dest)
            print(f"Successfully downloaded {model_name} to {dest}.")


if __name__ == "__main__":
    run()
