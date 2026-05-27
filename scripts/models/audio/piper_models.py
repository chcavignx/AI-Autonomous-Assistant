# !/usr/bin/env python3
"""Script to move data from data/models/piper to cache/models/piper."""

import os
import pathlib
import sys

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import config

LOCAL_PIPER_DIR = "models/piper"
DEST_PIPER_DIR = "piper"


def run() -> None:
    """Function to move data from data/models/piper to cache/audio/models/piper."""
    piper_dir = pathlib.Path(os.path.join(config.paths.data_path, LOCAL_PIPER_DIR))
    cache_piper_dir = pathlib.Path(
        os.path.join(config.paths.models_audio_path, DEST_PIPER_DIR)
    )
    pathlib.Path(cache_piper_dir).mkdir(exist_ok=True, parents=True)

    for model in os.listdir(piper_dir):
        model_path = pathlib.Path(os.path.join(piper_dir, str(model)))
        cache_model_path = pathlib.Path(os.path.join(cache_piper_dir, str(model)))
        if pathlib.Path(cache_model_path).exists():
            continue
        # os.rename(model_path, cache_model_path)
        pathlib.Path(cache_model_path).symlink_to(model_path)


if __name__ == "__main__":
    run()
