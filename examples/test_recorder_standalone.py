#!/usr/bin/env python3
"""Integration test: Standalone AudioRecorder.

Validates the simplified AudioRecorder utility from audio_utils.py,
testing capture lifecycle, numpy/float reading, and state management.

Run with:
  python examples/test_recorder_standalone.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import AudioRecorder
from src.utils.config import load_config


def test_recorder_lifecycle() -> bool:

    config = load_config()

    recorder = AudioRecorder(config)

    try:
        # 1. Test start
        if not recorder.start():
            return False

        assert recorder.is_recording, "is_recording should be True after start()"

        # 2. Test reading numpy (int16)
        chunks_captured = 0
        start_time = time.time()
        while time.time() - start_time < 1.0:
            data = recorder.read_numpy()
            if data is not None:
                assert data.dtype == np.int16, f"Expected int16, got {data.dtype}"
                chunks_captured += 1
            time.sleep(0.05)

        if chunks_captured == 0:
            return False

        # 3. Test reading float32
        float_chunks = 0
        start_time = time.time()
        while time.time() - start_time < 1.0:
            data = recorder.read_float()
            if data is not None:
                assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
                assert np.all(data <= 1.0) and np.all(data >= -1.0), "Float data out of range [-1, 1]"
                float_chunks += 1
            time.sleep(0.05)

        if float_chunks == 0:
            return False

        # 4. Test stop
        recorder.stop()

        assert not recorder.is_recording, "is_recording should be False after stop()"

        # 5. Test state after stop
        data = recorder.read()
        if data is not None:
            return False

        # 6. Test close
        recorder.close()

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False
    finally:
        recorder.close()


if __name__ == "__main__":
    success = test_recorder_lifecycle()
    # Explicit exit to prevent potential ALSA/PortAudio teardown segfaults
    import os
    os._exit(0 if success else 1)
