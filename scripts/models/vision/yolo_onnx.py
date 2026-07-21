#!/usr/bin/env python3
"""
    Download YOLOv26 models from Ultralytics repository.
    Convert YOLOv26 models to ONNX format.
    Run benchmark on the ONNX models.
"""
import os
import pathlib
import sys
import urllib.request
from pathlib import Path
from ultralytics import YOLO

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

# URL mapping
URLS = {
    "yolo26n-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt",
    "yolo26n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
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


def convert_to_onnx(model_path: Path, dest_path: Path) -> Path:
    """Convert a YOLO model to ONNX format."""
    if DRY_RUN:
        print(f"[DRY-RUN] Would convert from {model_path} to {dest_path}")
        return dest_path
    model = YOLO(model_path)
    return model.export(format="onnx", device="cpu")  # creates 'yolo26n.onnx'

def convert_to_ncnn(model_path: Path, dest_path: Path) -> Path:
    """Convert a YOLO model to NCNN format."""
    if DRY_RUN:
        print(f"[DRY-RUN] Would convert from {model_path} to {dest_path}")
        return dest_path
    model = YOLO(model_path)
    return model.export(format="ncnn", device="cpu")  # creates 'yolo26n.ncnn'

def run() -> None:
    # 1. Download models
    for model_name in ["yolo26n-cls.pt", "yolo26n.pt"]:
        dest = CACHE_DIR / model_name
        if not DRY_RUN and dest.exists():
            print(f"{model_name} already exists at {dest}. Skipping.")
        else:
            download_file(URLS[model_name], dest)
    #2. Convert model to ONNX
        dest_path = CACHE_DIR / f"{model_name.replace('.pt', '.onnx')}"
        _deth_path = convert_to_onnx(dest, dest_path)
        if DRY_RUN:
            print(f"[DRY-RUN] Would save ONNX model to {_deth_path}")
        else:
            print(f"ONNX model saved to {_deth_path}")

    #2. Convert model to NCNN
        dest_path = CACHE_DIR / f"{model_name.replace('.pt', '.ncnn')}"
        _deth_path = convert_to_ncnn(dest, dest_path)
        if DRY_RUN:
            print(f"[DRY-RUN] Would save NCNN model to {_deth_path}")
        else:
            print(f"NCNN model saved to {_deth_path}")

if __name__ == "__main__":
    run()
