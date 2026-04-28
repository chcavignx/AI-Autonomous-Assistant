#!/usr/bin/env python3
"""Script to load all models."""

import pathlib
import sys

# Add project root to sys.path
root_path = pathlib.Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import fast_whisper_objects
import load_huggingface_objects
import piper_models
import wakeword_model
import whisper_objects


def main() -> None:
    """Orchestrates loading of all audio-related models in a fixed sequence.

    Prints progress banners for each phase and invokes the model-loading routines for Whisper, Fast Whisper, Hugging Face objects, Vosk, and Piper in order.
    """
    phases = [
        ("Whisper Models", whisper_objects.run),
        ("Fast Whisper Models", fast_whisper_objects.run),
        ("Hugging Face Objects", load_huggingface_objects.run),
        ("Piper Models", piper_models.run),
        ("Wakeword Model", wakeword_model.run),
        # ("Vosk Models", vosk_models.run),
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
