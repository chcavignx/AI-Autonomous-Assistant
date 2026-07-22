#!/usr/bin/env python3
"""Script to download face detector models and datasets."""

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
INSIGHTFACE_CACHE_DIR = config.paths.models_vision_path / "insightface"
if DRY_RUN:
    print(f"[DRY-RUN] Would create directory at {INSIGHTFACE_CACHE_DIR}")
else:
    INSIGHTFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_CACHE_DIR = config.paths.models_vision_path / "cascade"
# Ensure directories exist
if DRY_RUN:
    print(f"[DRY-RUN] Would create directory at {CASCADE_CACHE_DIR}")
else:
    CASCADE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARCFACE_CACHE_DIR = INSIGHTFACE_CACHE_DIR
if DRY_RUN:
    print(f"[DRY-RUN] Would create directory at {ARCFACE_CACHE_DIR}")
else:
    ARCFACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# arcface models
#  "https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/face-recognition-resnet100-arcface-onnx/model.yml"
#   download yaml flle and get value file.name and file.source link
#   https://omz-ai-edge.intel.com/openvino-api/2024.2.0/public/face-recognition-resnet100-arcface-onnx/face-recognition-resnet100-arcface-onnx.xml
# https://github.com/openvinotoolkit/openvino/blob/2024.2.0/demos/face_recognition_demo/python/README.md
# https://github.com/openvinotoolkit/openvino/blob/2024.2.0/samples/python/face-recognition/README.md
# https://github.com/openvinotoolkit/openvino/blob/2024.2.0/docs/tutorials/pytorch/face_recognition/face_recognition.html

# URL mapping
URLS = {
    "buffalo_l.zip":
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
    "haarcascade_profileface.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml",
    "arcface_r100_v1.onnx":
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx"

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
    for model_name in ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml"]:
        dest = CASCADE_CACHE_DIR / model_name
        if not DRY_RUN and dest.exists():
            print(f"{model_name} already exists at {dest}. Skipping.")
        else:
            download_file(URLS[model_name], dest)

    # 2. Download and extract datasets
    for zip_name in ["buffalo_l.zip"]:
        dest = INSIGHTFACE_CACHE_DIR / zip_name
        extract_target = INSIGHTFACE_CACHE_DIR / zip_name.replace(".zip", "")

        # If directory already exists, skip
        if not DRY_RUN and extract_target.exists():
            print(f"Dataset directory {extract_target} already exists. Skipping.")
        else:
            download_file(URLS[zip_name], dest)
            extract_zip(dest, extract_target)

    # 3. Download arcface models
    for model_name in ["arcface_r100_v1.onnx"]:
        dest = ARCFACE_CACHE_DIR / model_name
        if not DRY_RUN and dest.exists():
            print(f"{model_name} already exists at {dest}. Skipping.")
        else:
            download_file(URLS[model_name], dest)


if __name__ == "__main__":
    run()
