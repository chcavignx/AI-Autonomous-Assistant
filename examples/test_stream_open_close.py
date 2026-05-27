#!/usr/bin/env python3
"""Integration test: Audio Stream Open/Close.

Tests that audio streams can be opened, used, and closed safely.
Validates codec configuration and frame buffer sizing.

Run with:
  python examples/test_stream_open_close.py
"""
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import (
    create_input_stream,
    create_output_stream,
    get_audio_backend,
    get_chunk_frames,
    get_default_input_device,
    get_default_output_device,
)
from src.utils.config import load_config


def test_input_stream_open_close() -> bool | None:
    """Test opening and closing an input stream."""
    config = load_config()
    backend = get_audio_backend(config)
    device = get_default_input_device(backend)

    if device is None:
        return False

    # Try common sample rates for input
    sample_rate = None
    stream = None
    for sr in [16000, 44100, 48000]:
        try:
            chunk_frames = get_chunk_frames(sr, config.audio.input_chunk_ms)
            stream = create_input_stream(
                rate=sr,
                chunk_frames=chunk_frames,
                device_index=device.index,
                backend=backend
            )
            if stream.start():
                stream.close()
                sample_rate = sr
                break
        except (OSError, ValueError, ImportError):
            continue

    if sample_rate is None:
        return False

    chunk_frames = get_chunk_frames(sample_rate, config.audio.input_chunk_ms)

    try:
        # Open stream
        stream = create_input_stream(
            rate=sample_rate,
            chunk_frames=chunk_frames,
            device_index=device.index,
            backend=backend
        )
        success = stream.start()
        assert success, "Failed to start stream"

        # Verify stream is active
        assert stream.active, "Stream is not active"

        # Read one chunk
        data = stream.read(chunk_frames)
        assert data is not None, "No data read from stream"
        assert len(data) > 0, "Empty data read from stream"

        # Stop stream
        stream.stop()

        # Verify stream is stopped
        assert not stream.active, "Stream is still active after stop"

        return True

    except Exception:
        return False
    finally:
        if stream:
            stream.close()


def test_output_stream_open_close() -> bool | None:
    """Test opening and closing an output stream."""
    config = load_config()
    backend = get_audio_backend(config)
    device = get_default_output_device(backend)

    if device is None:
        return False

    sample_rate = 16000
    chunk_frames = 512

    stream = None
    try:
        # Open stream
        stream = create_output_stream(
            rate=sample_rate,
            chunk_frames=chunk_frames,
            device_index=device.index,
            backend=backend
        )
        success = stream.start()
        assert success, "Failed to start stream"

        # Verify stream is active
        assert stream.active, "Stream is not active"

        # Write silence
        silence = np.zeros(chunk_frames, dtype=np.float32)
        stream.write(silence)

        # Stop stream
        stream.stop()

        # Verify stream is stopped
        assert not stream.active, "Stream is still active after stop"

        return True

    except Exception:
        return False
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    success1 = test_input_stream_open_close()
    success2 = test_output_stream_open_close()

    all_passed = success1 and success2

    sys.exit(0 if all_passed else 1)
