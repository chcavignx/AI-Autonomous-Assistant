#!/usr/bin/env python3
"""Integration test: Audio Stream Open/Close.

Tests that audio streams can be opened, used, and closed safely.
Validates codec configuration and frame buffer sizing.

Run with:
  python examples/test_stream_open_close.py
"""
import pyaudio
import sys
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def get_first_input_device():
    """Get first available input device index."""
    pa: PyAudio = pyaudio.PyAudio()
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_input = info.get("maxInputChannels", 0)
            if max_input and max_input > 0:
                return idx
    finally:
        pa.terminate()
    return None


def get_first_output_device():
    """Get first available output device index."""
    pa: PyAudio = pyaudio.PyAudio()
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_output = info.get("maxOutputChannels", 0)
            if max_output and max_output > 0:
                return idx
    finally:
        pa.terminate()
    return None


def test_input_stream_open_close():
    """Test opening and closing an input stream."""
    print("\n" + "=" * 60)
    print("TEST: Input Stream Open/Close")
    print("=" * 60)

    config: Config = load_config()
    device_idx: int | None = get_first_input_device()

    if device_idx is None:
        print("⊘ SKIPPED: No input device available")
        return True

    # Try common sample rates for input
    sample_rate = None
    for sr in [16000, 44100, 48000]:
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=512,
                input_device_index=device_idx,
            )
            stream.close()
            pa.terminate()
            sample_rate = sr
            break
        except (OSError, ValueError):
            pa.terminate()
            continue

    if sample_rate is None:
        print(f"⊘ SKIPPED: No supported sample rate found for device {device_idx}")
        return True

    channels = 1
    frames_per_buffer = int(sample_rate * config.audio.input_chunk_ms / 1000)

    print(f"Device index: {device_idx}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Frames per buffer: {frames_per_buffer}")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        # Open stream
        print("\nOpening input stream...", end=" ", flush=True)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
            input_device_index=device_idx,
        )
        print("✓")

        # Verify stream is active
        print("Verifying stream is active...", end=" ", flush=True)
        assert stream.is_active(), "Stream is not active"
        print("✓")

        # Read one chunk
        print("Reading one audio chunk...", end=" ", flush=True)
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        assert len(data) > 0, "No data read from stream"
        print(f"✓ ({len(data)} bytes)")

        # Stop stream
        print("Stopping stream...", end=" ", flush=True)
        stream.stop_stream()
        print("✓")

        # Verify stream is stopped
        print("Verifying stream is stopped...", end=" ", flush=True)
        assert not stream.is_active(), "Stream is still active after stop"
        print("✓")

        print("\n✓ PASSED: Input stream open/close successful")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False
    finally:
        if stream:
            try:
                stream.close()
            except OSError:
                pass
        pa.terminate()


def test_output_stream_open_close():
    """Test opening and closing an output stream."""
    print("\n" + "=" * 60)
    print("TEST: Output Stream Open/Close")
    print("=" * 60)

    device_idx = get_first_output_device()

    if device_idx is None:
        print("⊘ SKIPPED: No output device available")
        return True

    sample_rate = 16000
    channels = 1
    frames_per_buffer = 512

    print(f"Device index: {device_idx}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Frames per buffer: {frames_per_buffer}")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        # Open stream
        print("\nOpening output stream...", end=" ", flush=True)
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=frames_per_buffer,
            output_device_index=device_idx,
        )
        print("✓")

        # Verify stream is active
        print("Verifying stream is active...", end=" ", flush=True)
        assert stream.is_active(), "Stream is not active"
        print("✓")

        # Write silence
        import numpy as np
        print("Writing silence chunk...", end=" ", flush=True)
        silence = np.zeros(frames_per_buffer, dtype=np.float32).tobytes()
        stream.write(silence)
        print("✓")

        # Stop stream
        print("Stopping stream...", end=" ", flush=True)
        stream.stop_stream()
        print("✓")

        # Verify stream is stopped
        print("Verifying stream is stopped...", end=" ", flush=True)
        assert not stream.is_active(), "Stream is still active after stop"
        print("✓")

        print("\n✓ PASSED: Output stream open/close successful")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False
    finally:
        if stream:
            try:
                stream.close()
            except OSError:
                pass
        pa.terminate()


if __name__ == "__main__":
    success1 = test_input_stream_open_close()
    success2 = test_output_stream_open_close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = success1 and success2
    print(f"Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")

    sys.exit(0 if all_passed else 1)
