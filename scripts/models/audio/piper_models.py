# !/usr/bin/env python3

import os

from utils.config import setup_python_path, load_config

setup_python_path()
config = load_config()

PIPER_DIR = "models/piper"


def run() -> None:
    """Function to move data from data/models/piper to cache/models/piper"""
    cache_dir = config.paths.cache_path
    data_dir = config.paths.data_path
    piper_dir = data_dir / PIPER_DIR
    cache_piper_dir = cache_dir / PIPER_DIR
    os.makedirs(cache_piper_dir, exist_ok=True)

    for model in os.listdir(piper_dir):
        model_path = piper_dir / model
        cache_model_path = cache_piper_dir / model
        if os.path.exists(cache_model_path):
            print(f"Model {model} already exists in {cache_piper_dir}. Skipping.")
            continue
        print(f"Moving {model} to {cache_piper_dir}")
        os.rename(model_path, cache_model_path)
    print(f"All models have been moved to {cache_piper_dir}")


if __name__ == "__main__":
    run()
