#!/usr/bin/env python3
"""Integration test: Standalone WakeWordDetector.

Validates the WakeWordDetector engine in isolation, testing model loading,
hardware capture, and background thread orchestration.

Run with:
  python examples/test_wake_word_standalone.py
"""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.wake_word import WakeWordDetector
from src.utils.config import load_config

app_name = 'test_wake_word_standalone'
logger = logging.getLogger(app_name)

def test_wake_word_standalone() -> bool:

    config = load_config()

    detector = WakeWordDetector(config)
    try:
        detector.load()
    except Exception:
        return False

    trigger_count = 0

    def on_detected() -> None:
        nonlocal trigger_count
        trigger_count += 1

    try:
        detector.start(callback=on_detected)

        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(1)

        detector.stop()

        detector.unload()

        return True

    except Exception:
        return False


if __name__ == "__main__":
    success = test_wake_word_standalone()
    logger.info(f"Wake word standalone test {'passed' if success else 'failed'}")
    # Explicit exit to prevent potential ALSA/PortAudio teardown segfaults
    os._exit(0 if success else 1)
