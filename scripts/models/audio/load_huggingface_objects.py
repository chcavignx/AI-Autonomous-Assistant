#!/usr/bin/env python3
"""Script to download and save Hugging Face models, tokenizers, processors,
and their associated datasets to a local backup in your user cache directory."""

import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from models_check import model_exists

from src.utils.config import config

MODEL_NAMES = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-tiny",
    "distil-whisper/distil-large-v3",
    # "distil-whisper/distil-large-v3.5",
)
DATA_SET_NAMES = (
    "hf-internal-testing/librispeech_asr_dummy",
    "distil-whisper/librispeech_long",
)

CACHE_DIR = str(config.paths.models_path / "huggingface")


def run() -> None:
    """
    Download and save configured Hugging Face models, tokenizers, processors, and datasets to the local user cache.

    This function iterates over the module-level `model_names` and `data_set_names`, skipping entries already present in `cache_dir`. For each missing repository it attempts to download a snapshot into `cache_dir` and prints progress and completion messages. If a repository is not found or is gated, it prints a corresponding message and continues with the next item.
    """
    # repo_type="model" if None is by default "model" - Not mandatory but for clarity
    for model_name in MODEL_NAMES:
        if model_exists(model_name, CACHE_DIR):
            print(f"Model {model_name} already exists.")
            continue
        print(f"Downloading and saving {model_name} to {CACHE_DIR}")
        try:
            snapshot_download(
                repo_id=model_name, repo_type="model", cache_dir=CACHE_DIR
            )
            print(f"Model saved to: {os.path.join(CACHE_DIR, model_name)}")
        except RepositoryNotFoundError:
            print(f"Model {model_name} not found on Hugging Face.")
        except GatedRepoError:
            print(f"Model {model_name} is gated and requires authentication.")
    print("All huggingface models have been downloaded and saved.")

    for data_set_name in DATA_SET_NAMES:
        if model_exists(data_set_name, CACHE_DIR):
            print(f"Data_set {data_set_name} already exists.")
            continue
        print(f"Downloading and saving {data_set_name} to {CACHE_DIR}")
        # Load a hosted dataset
        try:
            snapshot_download(
                repo_id=data_set_name, repo_type="dataset", cache_dir=CACHE_DIR
            )
            print(f"Data_sets saved to: {os.path.join(CACHE_DIR, data_set_name)}")
        except RepositoryNotFoundError:
            print(f"Data_set {data_set_name} not found on Hugging Face.")
        except GatedRepoError:
            print(f"Data_set {data_set_name} is gated and requires authentication.")
    print("All data_sets have been downloaded and saved.")


if __name__ == "__main__":
    run()
