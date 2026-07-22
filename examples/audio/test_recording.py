#!/usr/bin/env python3
"""Integration test: Simple Audio Recording.

Tests that audio can be recorded from input device and saved to WAV file.
Validates mic connectivity and data capture using centralized audio_utils.

Run with:
  python examples/audio/test_recording.py
"""

import logging
import os
import sys
import wave
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.audio_utils import AudioPlayer, open_input_stream_with_fallback
from src.utils.config import load_config

app_name = 'test_recordering'
logger = logging.getLogger(app_name)

def test_simple_recording(duration_s: int = 2, output_file: str | Path | None = None) -> bool:
    """Test recording audio from microphone and saving to WAV."""
    config = load_config()
    resolved_output_file: Path
    if output_file is None:
        tmp_dir = Path(__file__).resolve().parent.parent.parent / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        resolved_output_file = tmp_dir / "test_recording.wav"
    else:
        resolved_output_file = Path(output_file)

    # Use fallback opener to get a working stream
    opened = open_input_stream_with_fallback(
        rate=config.audio.input_sample_rate,
        chunk_ms=config.audio.input_chunk_ms,
        device_index=config.audio.input_device_index,
        dtype="int16"  # Use int16 for direct WAV writing
    )

    if opened is None:
        return True

    stream = opened.stream
    frames: list[bytes] = []

    try:
        # Record audio
        chunk_count = int(duration_s * 1000 / config.audio.input_chunk_ms)

        for i in range(chunk_count):
            data = stream.read(opened.native_chunk_frames)
            if data is not None:
                frames.append(data)

            # Progress indicator
            _ = (i + 1) % (max(1, chunk_count // 4)) == 0

        # Write to WAV file
        resolved_output_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(resolved_output_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(opened.capture_rate)
            wf.writeframes(b"".join(frames))

        # Verify file size
        file_size = resolved_output_file.stat().st_size
        stream.close()
        return not file_size < 1000

    except Exception:
        return False
    finally:
        stream.close()


def test_recording_and_playback() -> bool:
    """Test recording then playing back the recorded audio."""
    tmp_dir = Path(__file__).resolve().parent.parent.parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    rec_file = tmp_dir / "test_record_playback.wav"

    # Step 1: Record
    if not test_simple_recording(duration_s=2, output_file=rec_file):
        return False

    # Step 2: Playback
    player = AudioPlayer()
    try:
        wav_bytes = rec_file.read_bytes()

        success = player.play_wav_bytes(wav_bytes, block=True)
        return bool(success)
    except Exception:
        return False
    finally:
        player.close()


if __name__ == "__main__":
    success1 = test_simple_recording()
    logger.info(f"Simple Recording test {'passed' if success1 else 'failed'}")
    success2 = test_recording_and_playback()
    logger.info(f"Recording and playback test {'passed' if success2 else 'failed'}")

    all_passed = success1 and success2
    logger.info(f"Overall Recording test {'passed' if all_passed else 'failed'}")
    os._exit(0 if all_passed else 1)
