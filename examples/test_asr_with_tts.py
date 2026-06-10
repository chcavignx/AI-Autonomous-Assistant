#!/usr/bin/env python3
"""Test ASR transcription using TTS-generated audio.

This validates the complete ASR+TTS integration flow by:
1. Generating audio using TTS (Piper) via _synthesize()
2. Saving it to a temporary WAV file
3. Transcribing it using ASREngine.transcribe_file()
4. Comparing the result to the original text

Run with:
  python examples/test_asr_with_tts_integration.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is accessible BEFORE any src.* imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.utils.config import load_config

app_name = 'test_asr_with_tts'
logger = logging.getLogger(app_name)


def test_asr_with_tts() -> bool:
    """Test ASR transcription using TTS-generated audio (offline synthesis + file transcription)."""
    config = load_config()

    # Test text - use English since config is set to English
    test_texts = [
        "Hello, how are you today?",
        "I am a voice assistant.",
        "Testing voice recognition.",
    ]

    # Initialize TTS engine
    try:
        tts_engine = TTSEngine(config)
        tts_engine.load()
    except Exception:
        logger.exception("Failed to load TTS engine")
        return False

    # Initialize ASR engine
    try:
        asr_engine = ASREngine(config)
        asr_engine.load()
    except Exception:
        logger.exception("Failed to load ASR engine")
        tts_engine.unload()
        return False

    # Test each text
    all_passed = True
    for test_text in test_texts:
        tmp_path: str | None = None
        try:
            # Step 1: Synthesize text to WAV bytes
            wav_bytes = tts_engine._synthesize(test_text)
            if wav_bytes is None:
                logger.warning("TTS synthesis returned None for: '%s'", test_text)
                all_passed = False
                continue

            # Step 2: Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                _ = tmp_file.write(wav_bytes)
                tmp_path = tmp_file.name

            # Step 3: Transcribe the audio file
            transcribed_text = asr_engine.transcribe_file(tmp_path)

            # Step 4: Compare against original text
            if transcribed_text and len(transcribed_text.strip()) > 0:
                original_words = set(test_text.lower().split())
                transcribed_words = set(transcribed_text.lower().split())
                matching_words = original_words & transcribed_words

                if matching_words:
                    logger.info(
                        "Text '%s' -> transcript '%s' (matched: %s)",
                        test_text, transcribed_text, matching_words,
                    )
                else:
                    logger.warning(
                        "Text '%s' -> no matching words in transcript: '%s'",
                        test_text, transcribed_text,
                    )
                    all_passed = False
            else:
                logger.warning("Text '%s' -> empty transcription", test_text)
                all_passed = False

        except Exception:
            import traceback
            traceback.print_exc()
            all_passed = False
        finally:
            # Clean up temp file
            if tmp_path is not None and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    # Cleanup engines
    try:
        asr_engine.unload()
        tts_engine.unload()
    except Exception:
        pass

    return bool(all_passed)


if __name__ == "__main__":
    success = test_asr_with_tts()
    logger.info(f"ASR with TTS test {'passed' if success else 'failed'}")
    os._exit(0 if success else 1)
