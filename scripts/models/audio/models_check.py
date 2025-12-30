#!/usr/bin/env python3
"""Script to check if a model exists in the target directory."""

from pathlib import Path


def model_exists(model_name: str, target_dir: str) -> bool:
    """
    Determine whether a model whose name contains the given substring exists in the target directory.

    Searches the resolved target directory for any entry whose name contains model_name. Symlinks are followed and both directories and regular files (e.g., model files like `.pt`) are considered matches.

    Returns:
        True if a matching file or directory exists in target_dir, False otherwise.
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
