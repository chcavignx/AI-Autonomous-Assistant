#!/usr/bin/env python3
"""Integration test: Simple Audio Playback.

Tests that audio data can be generated (sine wave), sent to output stream,
and played back through hardware.

Run with:
  python examples/test_playback.py
"""

import sys
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyaudio
import time


def find_first_output_device():
    """Get first available output device index."""
    pa = pyaudio.PyAudio()
    try:
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        max_output = info.get("maxOutputChannels", 0)
        if max_output and max_output > 0:
            return idx
    finally:
        pa.terminate()
    return None


def generate_sine_wave(frequency: int, duration_s: int, sample_rate: int):
    """Generate a sine wave at specified frequency.

    Args:
        frequency: Hz (e.g., 440 for A note)
        duration_s: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        numpy array of float32 samples
    """
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples, False)
    wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return wave


def test_playback_sine_wave():
    """Test playback of a generated sine wave."""
    print("\n" + "=" * 60)
    print("TEST: Sine Wave Playback")
    print("=" * 60)

    device_idx = find_first_output_device()

    if device_idx is None:
        print("⊘ SKIPPED: No output device available")
        return True

    sample_rate = 16000
    channels = 1
    duration_s = 2  # 2 seconds
    frequency = 440  # A note

    print(f"Device index: {device_idx}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_s}s")
    print(f"Frequency: {frequency} Hz (sine wave)")

    # Generate sine wave
    print("\nGenerating sine wave...", end=" ", flush=True)
    wave = generate_sine_wave(frequency, duration_s, sample_rate)
    print(f"✓ ({len(wave)} samples)")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        # Open output stream
        print("Opening output stream...", end=" ", flush=True)
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=512,
            output_device_index=device_idx,
        )
        print("✓")

        # Play the sine wave
        print("Playing sine wave...", end=" ", flush=True)
        stream.write(wave.tobytes())
        print("✓ (data sent to output buffer)")

        # Wait for playback to complete
        print("Waiting for playback...", end=" ", flush=True)
        time.sleep(duration_s + 0.5)  # Extra time to ensure playback
        print("✓")

        print("\n✓ PASSED: Sine wave playback successful")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except OSError:
                pass
        pa.terminate()


def test_playback_silence():
    """Test playback of silence (zeros)."""
    print("\n" + "=" * 60)
    print("TEST: Silence Playback")
    print("=" * 60)

    device_idx = find_first_output_device()

    if device_idx is None:
        print("⊘ SKIPPED: No output device available")
        return True

    sample_rate = 16000
    channels = 1
    duration_s = 1

    print(f"Device index: {device_idx}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_s}s")

    # Generate silence
    print("\nGenerating silence...", end=" ", flush=True)
    silence = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    print(f"✓ ({len(silence)} samples)")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        # Open output stream
        print("Opening output stream...", end=" ", flush=True)
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=512,
            output_device_index=device_idx,
        )
        print("✓")

        # Play silence
        print("Playing silence...", end=" ", flush=True)
        stream.write(silence.tobytes())
        print("✓ (silence sent to output buffer)")

        # Wait for playback
        print("Waiting for playback...", end=" ", flush=True)
        time.sleep(duration_s + 0.2)
        print("✓")

        print("\n✓ PASSED: Silence playback successful")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except OSError:
                pass
        pa.terminate()


if __name__ == "__main__":
    success1 = test_playback_sine_wave()
    success2 = test_playback_silence()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = success1 and success2
    print(f"Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")

    sys.exit(0 if all_passed else 1)
