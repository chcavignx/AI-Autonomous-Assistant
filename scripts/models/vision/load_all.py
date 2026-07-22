#!/usr/bin/env python3
"""Script to load all models."""

import pathlib
import sys

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import yolo_onnx
import libre_yolo_onnx
import coco_load
import detector_load



def main() -> None:
    """Orchestrates loading of all vision-related models in a fixed sequence."""
    phases = [
        ("YOLO ONNX Models", yolo_onnx.run),
        ("Libre YOLO ONNX Models", libre_yolo_onnx.run),
        ("Detector Models", detector_load.run),
        ("COCO Dataset", coco_load.run),
    ]

    failed_phases = []
    success_count = 0

    for name, run_func in phases:
        try:
            run_func()
            success_count += 1
        except Exception:  # pylint: disable=broad-except
            # We catch the general Exception here to ensure that a failure in one
            # model loading phase doesn't prevent other phases from running.
            failed_phases.append(name)

    if failed_phases:
        sys.exit(1)


if __name__ == "__main__":
    main()
