#!/usr/bin/env python3
"""Script to download LibreYOLO models and datasets from Hugging Face."""

from __future__ import annotations

import os
import pathlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import config

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# Target directory
CACHE_DIR = config.paths.models_vision_path / "yolo"
if DRY_RUN:
    print(f"[DRY-RUN] Would create directory at {CACHE_DIR}")
else:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = config.paths.dataset_vision_path
# Ensure directories exist
if DRY_RUN:
    print(f"[DRY-RUN] Would create directory at {DATASET_DIR}")
else:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

# URL mapping
URLS = {
    "LibreYOLOXn.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXn/resolve/main/LibreYOLOXn.onnx",
    "LibreYOLOXn.pt": "https://huggingface.co/LibreYOLO/LibreYOLOXn/resolve/main/LibreYOLOXn.pt",
    "coco8.zip": "https://huggingface.co/datasets/LibreYOLO/coco8/resolve/main/coco8.zip",
    "coco128.zip": "https://huggingface.co/datasets/LibreYOLO/coco128/resolve/main/coco128.zip",
    "coco8.yaml": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml",
    "coco128.yaml": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml",
}


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress reporting."""
    if DRY_RUN:
        print(f"[DRY-RUN] Would download from {url} to {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")

    def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100.0, (block_num * block_size / total_size) * 100)
            print(f"\rProgress: {percent:.1f}%", end="")
        else:
            print(f"\rDownloaded {block_num * block_size} bytes", end="")

    urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_hook)
    print("\nDownload complete.")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive to the target directory and remove the zip file."""
    if DRY_RUN:
        print(f"[DRY-RUN] Would extract {zip_path} to {extract_dir} and delete zip archive {zip_path}")
        return

    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete. Cleaning up zip archive...")
    zip_path.unlink()


def run() -> None:
    # 1. Download models
    for model_name in ["LibreYOLOXn.onnx", "LibreYOLOXn.pt"]:
        dest = CACHE_DIR / model_name
        if not DRY_RUN and dest.exists():
            print(f"{model_name} already exists at {dest}. Skipping.")
        else:
            download_file(URLS[model_name], dest)

    # 2. Download and extract datasets
    for zip_name in ["coco8.zip", "coco128.zip"]:
        dest = DATASET_DIR / zip_name
        extract_target = DATASET_DIR / zip_name.replace(".zip", "")

        # If directory already exists, skip
        if not DRY_RUN and extract_target.exists():
            print(f"Dataset directory {extract_target} already exists. Skipping.")
        else:
            download_file(URLS[zip_name], dest)
            extract_zip(dest, DATASET_DIR)
    # 3. Download YAML files
    for yml_name in ["coco8.yaml", "coco128.yaml"]:
        dest = DATASET_DIR / yml_name
        if not DRY_RUN and dest.exists():
            print(f"{yml_name} already exists at {dest}. Skipping.")
        else:
            download_file(URLS[yml_name], dest)

if __name__ == "__main__":
    run()
