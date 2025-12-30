#!/usr/bin/env python3
"""Script to load all models."""

import sys
import traceback

import fast_whisper_objects
import load_huggingface_objects
import piper_models
import vosk_models
import whisper_objects


def main() -> None:
    """
    Orchestrates loading of all audio-related models in a fixed sequence.

    Prints progress banners for each phase and invokes the model-loading routines for Whisper, Fast Whisper, Hugging Face objects, Vosk, and Piper in order.
    """
    print("==================================================")
    print("🚀 Starting Master Model Loading Process")
    print("==================================================")

    phases = [
        ("Whisper Models", whisper_objects.run),
        ("Fast Whisper Models", fast_whisper_objects.run),
        ("Hugging Face Objects", load_huggingface_objects.run),
        ("Vosk Models", vosk_models.run),
        ("Piper Models", piper_models.run),
    ]

    failed_phases = []
    success_count = 0

    for name, run_func in phases:
        print(f"\n--- Phase {success_count + len(failed_phases) + 1}: {name} ---")
        try:
            run_func()
            success_count += 1
        except Exception:  # pylint: disable=broad-except
            # We catch the general Exception here to ensure that a failure in one
            # model loading phase doesn't prevent other phases from running.
            print(f"❌ Error in phase '{name}':")
            print(traceback.format_exc())
            failed_phases.append(name)

    print("\n==================================================")
    print("📋 Final Summary")
    print(f"✅ Successful phases: {success_count}/{len(phases)}")
    if failed_phases:
        print(f"❌ Failed phases: {', '.join(failed_phases)}")
        print("==================================================")
        sys.exit(1)
    else:
        print("✅ All model loading tasks completed successfully!")
        print("==================================================")


if __name__ == "__main__":
    main()
