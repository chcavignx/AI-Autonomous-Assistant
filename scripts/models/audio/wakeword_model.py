#!/usr/bin/env python3
"""Script to download and extract wakeword models to a local cache directory."""

import pathlib
import sys

import openwakeword

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import config

# Target directory
CACHE_DIR = config.wake.download_path
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# One-time download of all pre-trained models (or only select models)


def run() -> None:
    openwakeword.utils.download_models(
        model_names=["hey_jarvis_v0.1"], target_directory=CACHE_DIR
    )


if __name__ == "__main__":
    run()
