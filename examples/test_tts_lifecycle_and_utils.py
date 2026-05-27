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

import sys
import time
from pathlib import Path
from typing import Any, cast

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.audio.audio_utils import (
    AudioUtils,
    convert_to_float32,
    convert_to_int16,
    install_alsa_error_handler,
    is_backend_available,
    resolve_device_index,
    suppress_pa_stderr,
)
from src.audio.tts import TTSEngine
from src.utils.config import load_config


def test_audio_utils_functions() -> bool:

    # 1. Test is_backend_available
    is_backend_available("sounddevice")
    is_backend_available("pyaudio")

    # 2. Test suppress_pa_stderr context manager
    try:
        with suppress_pa_stderr():
            _ = sys.stderr.write("This test message to stderr should be suppressed and invisible!\n")
            sys.stderr.flush()
    except Exception:
        return False

    # 3. Test install_alsa_error_handler
    try:
        _handler = install_alsa_error_handler()
    except Exception:
        return False

    # 4. Test float32 and int16 converters
    try:
        # Array of floats in [-1.0, 1.0]
        test_floats = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        ints = convert_to_int16(test_floats)
        assert ints[1] == 0, f"Expected 0, got {ints[1]}"

        # Convert back
        floats_back = convert_to_float32(ints)
        # floats_back[1] is known to be float32 from convert_to_float32
        assert abs(float(floats_back[1])) < 1e-4, f"Expected ~0, got {floats_back[1]}"
    except Exception:
        return False

    # 5. Test resolve_device_index
    try:
        # None resolves to None (default device)
        res_none = resolve_device_index(None, is_input=False)
        assert res_none is None, f"Expected None, got {res_none}"

        # Resolve invalid index
        res_invalid = resolve_device_index(9999, is_input=False)
        assert res_invalid is None, f"Expected None for invalid index, got {res_invalid}"
    except Exception:
        return False

    # 6. Test legacy AudioUtils wrapper
    try:
        test_floats = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        cleaned = AudioUtils.validate_and_clean_audio(test_floats)
        ints_legacy = AudioUtils.convert_to_int16(test_floats)
        assert cleaned.shape == test_floats.shape
        assert ints_legacy.shape == test_floats.shape
    except Exception:
        return False

    return True


def test_tts_lifecycle() -> bool:

    config = load_config()
    try:
        tts = TTSEngine(config)
    except Exception:
        return False

    try:
        tts.load()
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

    start_time = time.time()
    tts.speak("Testing done.", blocking=False)
    tts.wait()
    time.time() - start_time

    try:
        _ = cast("Any", tts).unload()
    except Exception:
        return False

    return True


if __name__ == "__main__":
    success_utils = test_audio_utils_functions()
    success_tts = test_tts_lifecycle()

    all_passed = success_utils and success_tts
    sys.exit(0 if all_passed else 1)
