#!/usr/bin/env python3
"""Script to download and save Hugging Face models, tokenizers, processors,
and their associated datasets to a local backup in your user cache directory.
"""

import os
import pathlib
import sys

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from models_check import model_exists

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Charge le fichier .env dans les variables d'environnement
load_dotenv()  # par défaut, cherche un fichier .env dans le répertoire courant

# Configuration with defaults and .env overrides
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

import contextlib

from src.utils.config import config

MODEL_NAMES = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-tiny",
    # "distil-whisper/distil-large-v3",
    # "distil-whisper/distil-large-v3.5",
)
DATA_SET_NAMES = (
    "hf-internal-testing/librispeech_asr_dummy",
    "distil-whisper/librispeech_long",
)

CACHE_DIR = str(config.paths.models_audio_path / "huggingface")


def run() -> None:
    """Download and save configured Hugging Face models, tokenizers, processors, and datasets to the local user cache.

    This function iterates over the module-level `model_names` and `data_set_names`, skipping entries already present in `cache_dir`. For each missing repository it attempts to download a snapshot into `cache_dir` and prints progress and completion messages. If a repository is not found or is gated, it prints a corresponding message and continues with the next item.
    """
    # repo_type="model" if None is by default "model" - Not mandatory but for clarity
    for model_name in MODEL_NAMES:
        if model_exists(model_name, CACHE_DIR):
            continue
        with contextlib.suppress(RepositoryNotFoundError, GatedRepoError):
            snapshot_download(
                repo_id=model_name, repo_type="model", cache_dir=CACHE_DIR
            )

    for data_set_name in DATA_SET_NAMES:
        if model_exists(data_set_name, CACHE_DIR):
            continue
        # Load a hosted dataset
        with contextlib.suppress(RepositoryNotFoundError, GatedRepoError):
            snapshot_download(
                repo_id=data_set_name, repo_type="dataset", cache_dir=CACHE_DIR
            )


if __name__ == "__main__":
    run()
