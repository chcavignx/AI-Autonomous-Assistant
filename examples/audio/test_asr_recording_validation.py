#!/usr/bin/env python3
"""Integration test: ASREngine Recording Validation.

Uses the production ASREngine to capture audio and validates that the
captured data is valid PCM by checking the stored WAV file.

The test does NOT require speech/transcription - it only validates that
the ASR pipeline captures audio and writes a valid WAV file.

Run with:
  python examples/test_asr_recording_validation.py
"""

import logging
import os
import sys
import time
import wave
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.asr import ASREngine
from src.utils.config import load_config

app_name = 'test_asr_recording_validation'
logger = logging.getLogger(app_name)


def test_asr_recording() -> bool:

    # 1. Load configuration
    config = load_config()

    # 2. Resolve the store_audio_path - ASREngine writes captured audio here
    store_path = Path(config.asr.store_audio_path)
    if not store_path.is_absolute():
        store_path = Path(__file__).resolve().parent.parent.parent / store_path

    # Remove any stale file from a previous run so we can detect a fresh write
    if store_path.exists():
        store_path.unlink()

    # 3. Initialize Engine
    engine = ASREngine(config)
    try:
        engine.load()
    except Exception:
        logger.exception("Failed to load ASR engine")
        return False

    # 4. Collect transcripts (bonus validation - not required for pass)
    transcripts: list[str] = []

    def on_transcript(text: str) -> None:
        if text:
            transcripts.append(text)

    # 5. Run capture for a few seconds (silence is fine - we just test the pipeline)
    duration = 3

    try:
        engine.start(callback=on_transcript)
        logger.info('ASR start listening ...')

        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.5)

        engine.stop()
        logger.info('ASR stop listening ...')

    except Exception:
        engine.stop()
        logger.exception('Error during ASR capture')
        return False

    # 6. Validate: the ASR engine must have written the WAV file
    if not store_path.exists():
        logger.warning(
            "ASR recording validation failed: store file not found at %s. \
            Check that config.asr.store_audio=true and store_audio_path is set.",
            store_path,
        )
        return False

    file_size = store_path.stat().st_size
    if file_size < 44:  # WAV header alone is 44 bytes
        logger.warning(
            "ASR recording validation failed: store file too small (%d bytes)", file_size
        )
        return False

    # 7. Validate WAV file structure
    try:
        with wave.open(str(store_path), "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            logger.info(
                "WAV validation OK: %d frames @ %d Hz, %d ch, %d bytes/sample",
                n_frames, rate, channels, sampwidth,
            )
            if n_frames == 0:
                logger.warning("WAV file has 0 frames")
                return False
    except wave.Error as e:
        logger.warning("WAV file is invalid: %s", e)
        return False

    # 8. Log transcripts if any came through (informational only)
    if transcripts:
        logger.info("Bonus: transcripts received: %s", transcripts)
    else:
        logger.info(
            "No transcripts (silence during test) - recording pipeline validation still passed"
        )

    return True


if __name__ == "__main__":
    success = test_asr_recording()
    logger.info(f"ASR recording validation test {'passed' if success else 'failed'}")
    os._exit(0 if success else 1)
