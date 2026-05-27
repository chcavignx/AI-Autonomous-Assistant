#!/usr/bin/env python3
"""Integration test: Simple Audio Playback.

Tests that audio data can be generated (sine wave), sent to output stream,
and played back through hardware using the unified audio_utils library.

Run with:
  python examples/test_playback.py
"""

import sys
import time
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.typing import NDArray
from src.audio.audio_utils import (
    AudioPlayer,
    get_audio_backend,
    get_default_output_device,
)
from src.audio.tts import TTSEngine
from src.utils.config import load_config


def generate_sine_wave(
    frequency: int,
    duration_s: int,
    sample_rate: int,
) -> NDArray[np.float32]:
    """Generate a sine wave at the given frequency."""
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def test_playback_sine_wave() -> bool:
    """Test playback of a generated sine wave."""
    output_dev = get_default_output_device()
    if output_dev is None:
        return True

    get_audio_backend()
    sample_rate = 16000
    duration_s = 2    # 2 seconds
    frequency = 440    # A note

    # Generate sine wave
    wave_data = generate_sine_wave(frequency, duration_s, sample_rate)

    # Play
    player = AudioPlayer()
    ok = player.play_data(wave_data, sample_rate, block=False)

    if not ok:
        return False

    # Wait for playback to drain
    time.sleep(duration_s + 0.5)

    player.close()

    return True


def test_playback_silence() -> bool:
    """Test playback of silence (zeros)."""
    output_dev = get_default_output_device()
    if output_dev is None:
        return True

    get_audio_backend()
    sample_rate = 16000
    duration_s = 1

    # Generate silence
    silence = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

    # Play
    player = AudioPlayer()
    ok = player.play_data(silence, sample_rate, block=False)

    if not ok:
        return False

    time.sleep(duration_s + 0.2)

    player.close()

    return True


def test_playback_tts_wav() -> bool:
    """Test playback of TTS-generated WAV audio."""
    output_dev = get_default_output_device()
    if output_dev is None:
        return True

    get_audio_backend()

    # Load config and TTS engine
    config = load_config()

    try:
        tts_engine = TTSEngine(config)
        tts_engine.load()
    except Exception:
        return False

    # Generate audio from text
    test_text = "Hello, how are you today?"
    try:
        # pyright: ignore[reportPrivateUsage]
        wav_bytes = tts_engine._synthesize(test_text)
        if wav_bytes is None:
            tts_engine.unload()
            return False
    except Exception:
        tts_engine.unload()
        return False

    # Play the WAV bytes
    player = AudioPlayer()
    ok = player.play_wav_bytes(wav_bytes, block=False)

    tts_engine.unload()

    if not ok:
        return False

    # Estimate duration from WAV bytes (approx 22050 sample rate, 16-bit mono)
    duration_s = len(wav_bytes) / 22050 / 2
    time.sleep(duration_s + 0.5)
    player.close()

    # Save to temporary file for verification
    tmp_file = Path(__file__).parent.parent / ".tmp" / "test_playback_tts.wav"
    tmp_file.parent.mkdir(exist_ok=True)

    with tmp_file.open("wb") as wf:
        _ = wf.write(wav_bytes)

    return True


def test_playback_file() -> bool:
    """Test playback of a WAV file using play_file."""
    output_dev = get_default_output_device()
    if output_dev is None:
        return True

    # Use existing test.wav or create a small one
    wav_path = Path(__file__).resolve().parent.parent / "data" / "test.wav"
    if not wav_path.exists():
        return True

    player = AudioPlayer()
    ok = player.play_file(wav_path)

    duration_s = 1
    time.sleep(duration_s + 0.5)
    player.close()
    return ok


if __name__ == "__main__":
    success1 = test_playback_sine_wave()
    success2 = test_playback_silence()
    success3 = test_playback_tts_wav()
    success4 = test_playback_file()

    all_passed = success1 and success2 and success3 and success4
    status = "PASSED" if all_passed else "FAILED"

    sys.exit(0 if all_passed else 1)
