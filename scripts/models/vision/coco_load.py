#!/usr/bin/env python3
"""Script to download COCO 2017 dataset labels and images."""

from __future__ import annotations

import os
import pathlib
import sys
import urllib.request
import zipfile
import yaml
from pathlib import Path

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

NEED_COC0_DATA = False
# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import config

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
#see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml.
DATASET_DIR = config.paths.data_path / "dataset_vision"
COCO_YAML_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml"
COCO_DATA_DIR = config.paths.dataset_vision_path
COCO_YAML_PATH = COCO_DATA_DIR / "coco.yaml"


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

    if DRY_RUN:
        print(f"[DRY-RUN] Would create directory at {COCO_DATA_DIR}")
        print(f"[DRY-RUN] Would download yaml file from {COCO_YAML_URL} to {COCO_DATA_DIR}")
    else:
        COCO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading yaml file from {COCO_YAML_URL} to {COCO_DATA_DIR}...")
        download_file(COCO_YAML_URL, COCO_YAML_PATH)

    if not COCO_YAML_PATH.exists() and not DRY_RUN:
        raise FileNotFoundError(f"coco.yaml not found at {COCO_YAML_PATH}")

    if DRY_RUN:
        print(f"[DRY-RUN] Opening and loading yaml file from {COCO_YAML_PATH}")
        coco_data = {"path": "coco"}
    else:
        with open(COCO_YAML_PATH, "r", encoding="utf-8") as f:
            coco_data = yaml.safe_load(f)

    # dir = "path" value in coco.yaml file.
    # download to dataset path
    DATASET_PATH = COCO_DATA_DIR / coco_data["path"]
    if DRY_RUN:
        print(f"[DRY-RUN] Would create directory at {DATASET_PATH}")
    else:
        DATASET_PATH.mkdir(parents=True, exist_ok=True)


    # Download labels
    segments = False  # segment or box labels
    DATASET_LABELS_PATH = DATASET_PATH / ("segments" if segments else "labels")
    if DRY_RUN:
        print(f"[DRY-RUN] Would create directory at {DATASET_LABELS_PATH}")
    else:
        DATASET_LABELS_PATH.mkdir(parents=True, exist_ok=True)

    zip_name = "coco2017labels-segments.zip" if segments else "coco2017labels.zip"
    url = ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")
    dest = DATASET_PATH / zip_name

    if DRY_RUN:
        print(f"[DRY-RUN] Would download labels from {url} to {dest}")
        print(f"[DRY-RUN] Would extract {dest} to {COCO_DATA_DIR} and delete zip archive {dest}")
    else:
        if (DATASET_LABELS_PATH / "val2017").exists():
            print(f"Labels directory {DATASET_LABELS_PATH} already exists. Skipping.")
        else:
            download_file(url, dest)
            extract_zip(dest, COCO_DATA_DIR)

    # Download data
    DATASET_IMAGES_PATH = DATASET_PATH / "images"
    if DRY_RUN:
        print(f"[DRY-RUN] Would create directory at {DATASET_IMAGES_PATH}")
    else:
        DATASET_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    data_urls = {
        "train": "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
        "val": "http://images.cocodataset.org/zips/val2017.zip",    # 1G, 5k images
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", # 250MB
    }
    if NEED_COC0_DATA:
        for name in ["train", "val"]:
            zip_name = f"{name}2017.zip"
            dest = DATASET_IMAGES_PATH / zip_name
            extract_target = DATASET_IMAGES_PATH / f"{name}"

            # If directory already exists, skip
            if not DRY_RUN and extract_target.exists():
                print(f"Dataset directory {extract_target} already exists. Skipping.")
            else:
                download_file(data_urls[name], dest)
                extract_zip(dest, DATASET_IMAGES_PATH)

if __name__ == "__main__":
    run()
