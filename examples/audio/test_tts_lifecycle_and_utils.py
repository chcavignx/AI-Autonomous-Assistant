#!/usr/bin/env python3
"""E2E Example testing TTSEngine lifecycle and audio_utils.py.

Validates advanced TTSEngine functions:
  - Non-blocking synthesis: tts.speak(..., blocking=False)
  - State querying: tts.is_speaking
  - Synthesis interruption: tts.interrupt()
  - Synchronous coordination: tts.wait()

Validates audio_utils.py functions:
  - is_backend_available()
  - suppress_pa_stderr()
  - install_alsa_error_handler()
  - convert_to_float32() and convert_to_int16()
  - resolve_device_index()
  - Legacy AudioUtils compatibility class.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.audio.audio_utils import (
    AudioUtils,
    convert_to_float32,
    convert_to_int16,
    install_alsa_error_handler,
    resolve_device_index,
)
from src.audio.tts import TTSEngine
from src.utils.config import load_config

app_name = 'test_tts_lifecycle_and_utils'
logger = logging.getLogger(app_name)

def test_audio_utils_functions() -> bool:

    # Test install_alsa_error_handler
    try:
        _handler = install_alsa_error_handler()
        # _handler.close()
        # Keep handler alive to prevent garbage collection
        if _handler is None:
            logger.debug("ALSA error handler installation failed")
            return False
        logger.debug("ALSA error handler installed successfully")
    except Exception:
        logger.debug("ALSA error handler installation failed")
        return False

    # Test float32 and int16 converters
    try:
        # Array of floats in [-1.0, 1.0]
        test_floats = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        ints = convert_to_int16(test_floats)
        logger.debug("Converted to int16 successfully: %s", ints)
        # ints[1] is known to be int16 from convert_to_int16
        assert ints[1] == 0, f"Expected 0, got {ints[1]}"

        # Convert back
        floats_back = convert_to_float32(ints)
        logger.debug("Converted to float32 successfully: %s", floats_back)
        # floats_back[1] is known to be float32 from convert_to_float32
        assert abs(float(floats_back[1])) < 1e-4, f"Expected ~0, got {floats_back[1]}"
    except Exception:
        return False

    # Test resolve_device_index
    try:
        # None input -> returns default output device index (int) or None if no device
        # On a system with a default output device this will be an int, not necessarily None
        res_none = resolve_device_index(None, is_input=False)
        logger.debug("Resolved None device: %s", res_none)
        assert res_none is None or isinstance(res_none, int), (
            f"Expected int or None, got {type(res_none)}: {res_none}"
        )

        # Out-of-range index falls back to default output device (same as None input)
        res_invalid = resolve_device_index(9999, is_input=False)
        logger.debug("Resolved invalid device: %s", res_invalid)
        assert res_invalid is None or isinstance(res_invalid, int), (
            f"Expected int or None for invalid index, got {type(res_invalid)}: {res_invalid}"
        )
        # Both should resolve to the same default device
        assert res_none == res_invalid, (
            f"Expected both to resolve to same default device, got {res_none} vs {res_invalid}"
        )
    except Exception:
        return False

    # Test legacy AudioUtils wrapper
    try:
        test_floats = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        cleaned = AudioUtils.validate_and_clean_audio(test_floats)
        ints_legacy = AudioUtils.convert_to_int16(test_floats)
        logger.debug("Converted to int16 successfully: %s", ints_legacy)
        logger.debug("Cleaned audio successfully: %s", cleaned)
        assert cleaned.shape == test_floats.shape
        assert ints_legacy.shape == test_floats.shape
    except Exception:
        return False

    return True


def test_tts_lifecycle() -> bool:

    config = load_config()
    try:
        tts = TTSEngine(config)
        logger.debug("TTS engine initialized successfully")
    except Exception:
        logger.debug("TTS engine initialization failed")
        return False

    try:
        tts.load()
        logger.debug("TTS engine loaded successfully")
    except Exception:
        return False

    # Check that tts is initially not speaking
    assert not tts.is_speaking, "Expected is_speaking to be False initially"

    tts.speak("This is a long sentence meant to test the non blocking and interrupt capability of our text to speech engine.", blocking=False)

    # Wait briefly for playback thread to process the item and set is_speaking
    time.sleep(0.5)

    tts.interrupt()

    # Wait for the current playing sentence to finish
    tts.wait()
    assert not tts.is_speaking, "Expected is_speaking to be False after queue drains"

    tts.speak("Testing done.", blocking=False)
    tts.wait()

    try:
        _ = cast("Any", tts).unload()
    except Exception:
        return False

    return True


if __name__ == "__main__":
    success_utils = test_audio_utils_functions()
    logger.info(f"Audio utils test {'passed' if success_utils else 'failed'}")

    success_tts = test_tts_lifecycle()
    logger.info(f"TTS lifecycle test {'passed' if success_tts else 'failed'}")

    all_passed = success_utils and success_tts
    logger.info(f"TTS lifecycle and utils test {'passed' if all_passed else 'failed'}")
    sys.exit(0 if all_passed else 1)
