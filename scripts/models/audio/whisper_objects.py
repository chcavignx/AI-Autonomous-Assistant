#!/usr/bin/env python3
"""Script to download and save Hugging Face models, tokenizers, processors,
and their associated datasets to a local backup in your user cache directory."""

import os
import urllib.error
import urllib.request
from pathlib import Path

import whisper
from models_check import model_exists

from src.utils.sysutils import detect_raspberry_pi_model

# Define the target directory
cache_dir = os.path.join(os.path.expanduser("~"), "cache/models/whisper")

# Base models for all platforms (including Raspberry Pi)
MODELS_BASE = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

# Extended models for non-Raspberry Pi platforms
MODELS_EXTENDED = {
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

# Add GPT-2 models if needed
GPT2 = [
    "https://github.com/graykode/gpt-2-Pytorch/tree/master/GPT2/vocab.bpe",
    "https://github.com/graykode/gpt-2-Pytorch/tree/master/GPT2/encoder.json",
]


def get_models_to_download() -> dict:
    """
    Selects the set of Whisper model download mappings appropriate for the current platform.
    
    Returns:
        dict: Mapping of model names to their download URLs. On Raspberry Pi systems returns `MODELS_BASE`; on other platforms returns a merged mapping of `MODELS_BASE` and `MODELS_EXTENDED`.
    """
    # Add larger models if not on Raspberry Pi
    if not detect_raspberry_pi_model():
        return {**MODELS_BASE, **MODELS_EXTENDED}
    return MODELS_BASE


def download_file(url: str, target_dir: str, filename: str = None) -> None:
    """
    Download a file from a URL into a target directory, skipping or resuming as appropriate.
    
    Checks for an existing model/file using `model_exists` and skips download if present. Ensures the target directory exists (resolving symlinks), then downloads the URL to the given filename (defaults to the URL's final path segment). If a partial file is present, attempts to resume using HTTP Range requests; if the server does not support resuming, restarts the download. Handles HTTP 416 as an already-complete file and reports network or filesystem errors via printed messages.
    
    Parameters:
        url (str): The source URL of the file to download.
        target_dir (str): Directory path where the file will be saved; created if missing.
        filename (str, optional): Filename to use for the saved file. Defaults to the last path segment of `url`.
    """
    if not filename:
        filename = url.split("/")[-1]

    # Use model_exists to check if the file/model already exists
    if model_exists(filename, target_dir):
        print(
            f"Model/File {filename} already exists in {target_dir}. Skipping download."
        )
        return

    target_dir_path = Path(target_dir)

    # Handle directory creation, respecting symlinks
    if target_dir_path.is_symlink():
        # If it's a symlink, ensure the target directory exists
        resolved_path = target_dir_path.resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
    else:
        target_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = target_dir_path / filename

    # Check if file exists to resume
    downloaded = 0
    if file_path.exists():
        downloaded = file_path.stat().st_size
        print(f"Resuming download for {filename} from {downloaded} bytes...")
    else:
        print(f"Downloading {filename}...")

    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")

    try:
        with urllib.request.urlopen(req) as response:
            # Check if server supports partial content
            if downloaded > 0 and response.status == 206:
                mode = "ab"
            elif downloaded > 0 and response.status == 200:
                print("Server does not support resume. Restarting download.")
                mode = "wb"
                downloaded = 0  # Reset if restarting
            else:
                mode = "wb"

            with open(file_path, mode) as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"Finished {filename}")
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print(f"File {filename} already fully downloaded.")
        else:
            print(f"Error downloading {url}: {e}")
    except urllib.error.URLError as e:
        print(f"A network error occurred: {e}")
    except OSError as e:
        print(f"A file system error occurred: {e}")


def run() -> None:
    """
    Download Whisper models and related tokenizer and processor files into the module cache directory.
    
    Uses the whisper library to fetch models returned by get_models_to_download and saves them under cache_dir; if the library download fails, falls back to manually downloading model weight files. Also downloads configured GPT-2 support files into the same cache location.
    """
    print(f"Target directory: {cache_dir}")

    models_to_download = get_models_to_download()
    try:
        print("Using whisper library to download models...")
        for model_name, model_url in models_to_download.items():
            if model_exists(model_name, cache_dir):
                print(f"Model {model_name} already exists.")
                continue
            print(f"Downloading {model_name} via whisper.load_model...")
            whisper.load_model(model_name, download_dir=cache_dir)
    except RuntimeError as e:
        print(
            f"Whisper library download failed: {e}. Falling back to manual download..."
        )
        for model_name, model_url in models_to_download.items():
            download_file(model_url, cache_dir, filename=f"{model_name}.pt")

    for url in GPT2:
        download_file(url, cache_dir)

    print(f"✅ All whisper models have been downloaded into {cache_dir}")


if __name__ == "__main__":
    run()