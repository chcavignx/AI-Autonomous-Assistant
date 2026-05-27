#!/usr/bin/env python3
"""Test ASR transcription using TTS-generated audio.

This validates the complete ASR flow by:
1. Generating audio using TTS (Piper)
2. Saving it to a temporary WAV file
3. Transcribing it using ASREngine.transcribe_file()
4. Comparing the result to the original text

Run with:
  python examples/test_asr_with_tts.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.utils.config import load_config


def test_asr_with_tts() -> bool:
    """Test ASR transcription using TTS-generated audio."""
    config = load_config()

    # Test text - use English since config is set to English
    test_texts = [
        "Hello, how are you today?",
        "I am a voice assistant.",
        "Testing voice recognition.",
    ]

    # Initialize engines
    try:
        tts_engine = TTSEngine(config)
        tts_engine.load()
    except Exception:
        return False

    try:
        asr_engine = ASREngine(config)
        asr_engine.load()
    except Exception:
        tts_engine.unload()
        return False

    # Test each text
    all_passed = True
    for test_text in test_texts:

        # Generate audio with TTS
        try:
            # Use TTS engine's internal synthesis to get WAV bytes
            # pyright: ignore[reportPrivateUsage]
            wav_bytes = tts_engine._synthesize(test_text)
            if wav_bytes is None:
                all_passed = False
                continue

            # Save to temporary file for transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                _ = tmp_file.write(wav_bytes)
                tmp_path = tmp_file.name

            try:
                # Transcribe the audio file
                transcribed_text = asr_engine.transcribe_file(tmp_path)

                # Check if transcription contains some of the original text
                # (we don't expect perfect match due to ASR limitations)
                if transcribed_text and len(transcribed_text.strip()) > 0:
                    # Check if at least some words match
                    original_words = set(test_text.lower().split())
                    transcribed_words = set(transcribed_text.lower().split())
                    matching_words = original_words & transcribed_words

                    if matching_words:
                        pass
                    else:
                        all_passed = False
                else:
                    all_passed = False

            except Exception:
                all_passed = False
            finally:
                # Clean up temp file
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()

        except Exception:
            all_passed = False

    # Cleanup
    try:
        asr_engine.unload()
        tts_engine.unload()
    except Exception:
        pass

    return bool(all_passed)


if __name__ == "__main__":
    success = test_asr_with_tts()
    os._exit(0 if success else 1)
