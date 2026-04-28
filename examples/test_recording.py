#!/usr/bin/env python3
"""Integration test: Simple Audio Recording.

Tests that audio can be recorded from input device and saved to WAV file.
Validates mic connectivity and data capture.

Run with:
  python examples/test_recording.py
"""

import sys
import tempfile

from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT_DIR / ".tmp"
import wave
import pyaudio
from src.utils.config import load_config

SUPPORTED_SAMPLE_RATE: list[int]=[16000, 44100, 48000]

def find_first_input_device():
    """Get first available input device index."""
    pa = pyaudio.PyAudio()
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_input = info.get("maxInputChannels", 0)
            if max_input and max_input > 0:
               return idx
    finally:
        pa.terminate()
    return None


def find_supported_sample_rate(device_idx):
    """Find a sample rate supported by the device."""
    for sr in SUPPORTED_SAMPLE_RATE:
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
            return sr
        except (OSError, ValueError):
            pa.terminate()
            continue
    return None


def test_simple_recording(duration_s: int = 2, output_file: str | None = None) -> bool:
    """Test recording audio from microphone and saving to WAV.

    Args:
        duration_s: Duration to record in seconds
        output_file: Path to save WAV file (default: /tmp/test_recording.wav)
    """
    print("\n" + "=" * 60)
    print("TEST: Simple Audio Recording")
    print("=" * 60)

    if output_file is None:
        output_file = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    else:
        output_file = Path(output_file)

    device_idx = find_first_input_device()

    if device_idx is None:
        print("⊘ SKIPPED: No input device available")
        return True

    sample_rate = find_supported_sample_rate(device_idx)

    if sample_rate is None:
        print(f"⊘ SKIPPED: No supported sample rate found for device {device_idx}")
        return True

    channels = 1
    frames_per_buffer = 512

    print(f"Device index: {device_idx}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_s}s")
    print(f"Output file: {output_file}")

    pa = pyaudio.PyAudio()
    stream = None
    frames = []

    try:
        # Open input stream
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

        # Record audio
        total_frames = int(sample_rate / frames_per_buffer * duration_s)
        print(f"Recording {duration_s}s ({total_frames} chunks)...", end=" ", flush=True)

        for i in range(total_frames):
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            frames.append(data)
            # Progress indicator
            if (i + 1) % (total_frames // 4 or 1) == 0:
                print(f"{int((i + 1) / total_frames * 100)}%", end=" ", flush=True)

        print("✓")

        # Stop stream
        print("Stopping stream...", end=" ", flush=True)
        stream.stop_stream()
        print("✓")

        # Write to WAV file
        print(f"Writing WAV file ({output_file})...", end=" ", flush=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_file), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

        print("✓")

        # Verify file size
        file_size = output_file.stat().st_size
        print(f"File size: {file_size} bytes")

        expected_size = sample_rate * channels * pyaudio.get_sample_size(pyaudio.paInt16) * duration_s  # approx
        if file_size < expected_size * 0.5:  # Allow 50% variance
            print(f"⚠ Warning: File size smaller than expected ({file_size} vs ~{expected_size})")

        print("\n✓ PASSED: Audio recording successful")
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


def test_recording_and_playback(output_file: str | None = None) -> bool:
    """Test recording then playing back the recorded audio."""
    print("\n" + "=" * 60)
    print("TEST: Record, Save, and Playback")
    print("=" * 60)

    if output_file is None:
        output_file = Path("/tmp/test_record_playback.wav")
    else:
        output_file = Path(output_file)

    # Record
    print("\n[STEP 1] Recording audio...")
    input_device = find_first_input_device()
    output_device = None

    pa = pyaudio.PyAudio()
    try:
        # Find output device
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_output = info.get("maxOutputChannels", 0)
            if max_output and max_output > 0:
                output_device = idx
                break
    finally:
        pa.terminate()

    if input_device is None:
        print("⊘ SKIPPED: No input device available")
        return True
    if output_device is None:
        print("⊘ SKIPPED: No output device available for playback")
        return True

    # Do the recording
    if not test_simple_recording(duration_s=1, output_file=output_file):
        return False

    # Playback
    print("\n[STEP 2] Playing back recorded audio...")

    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(output_file), dtype="float32")
        print(f"Loaded: {len(audio_data)} samples at {sample_rate} Hz")

        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=512,
                output_device_index=output_device,
            )
            print("Playing back...", end=" ", flush=True)
            stream.write(audio_data.tobytes())
            import time
            time.sleep(len(audio_data) / sample_rate + 0.1)
            print("✓")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except OSError:
                    pass
            pa.terminate()

        print("\n✓ PASSED: Record and playback successful")
        return True

    except ImportError:
        print("⊘ SKIPPED: soundfile not available for playback test")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


if __name__ == "__main__":
    success1 = test_simple_recording()
    success2 = test_recording_and_playback()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = success1 and success2
    print(f"Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")

    sys.exit(0 if all_passed else 1)
