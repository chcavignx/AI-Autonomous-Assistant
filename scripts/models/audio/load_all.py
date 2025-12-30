#!/usr/bin/env python3
"""Script to load all models."""

import fast_whisper_objects
import load_huggingface_objects
import piper_models
import vosk_models
import whisper_objects


def main() -> None:
    """Function to load all models"""
    print("==================================================")
    print("🚀 Starting Master Model Loading Process")
    print("==================================================")

    print("\n--- Phase 1: Whisper Models ---")
    whisper_objects.run()

    print("\n--- Phase 2: Fast Whisper Models ---")
    fast_whisper_objects.run()

    print("\n--- Phase 3: Hugging Face Objects ---")
    load_huggingface_objects.run()

    print("\n--- Phase 4: Vosk Models ---")
    vosk_models.run()

    print("\n--- Phase 5: Piper Models ---")
    piper_models.run()

    print("\n==================================================")
    print("✅ All model loading tasks completed!")
    print("==================================================")


if __name__ == "__main__":
    main()
