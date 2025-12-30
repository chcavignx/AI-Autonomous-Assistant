#!/usr/bin/env python3
"""Script to check if a model exists in the target directory."""

from pathlib import Path


def model_exists(model_name: str, target_dir: str) -> bool:
    """Check if a model exists in the target directory.
    Handles symlinks and checks for both directories and files (e.g., .pt files).
    """
    target_path = Path(target_dir).resolve()
    if not target_path.exists():
        return False

    # Check if any entry in the directory contains the model name and exists
    for entry in target_path.iterdir():
        if model_name in entry.name:
            # entry.exists() follows symlinks by default
            if entry.is_dir() or entry.is_file():
                return True
    return False
