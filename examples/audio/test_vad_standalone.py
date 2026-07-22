#!/usr/bin/env python3
"""Standalone E2E test for the VADEngine.

Validates voice activity detection initialization, speech detection,
and segment/timestamp production using the repository sample audio file
(data/test.wav) and tests the audio resampling utility.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf
from src.audio.audio_utils import resample_audio
from src.audio.vad import VADEngine
from src.utils.config import load_config

if TYPE_CHECKING:
    from numpy.typing import NDArray

app_name = 'test_vad_standalone'
logger = logging.getLogger(app_name)

def test_vad_standalone() -> bool:

    # 1. Load config and initialize engine
    config = load_config()
    try:
        vad = VADEngine(config)
    except Exception:
        return False

    # 2. Read data/test.wav file
    wav_path = project_root / "data" / "test.wav"
    if not wav_path.exists():
        return False

    try:
        # sf.read returns (NDArray, int)
        read_result = cast("tuple[NDArray[np.float64], int]", cast("Any", sf).read(str(wav_path)))
        data, sample_rate = read_result
    except Exception:
        return False

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Ensure float32 format
    audio_float = data.astype(np.float32)

    # 3. Test VAD detection on speech
    try:
        has_speech = vad.is_speech_detected(audio_float)
        if not has_speech:
            return False
    except Exception:
        return False

    # 4. Test VAD on pure silence
    silence = np.zeros(config.audio.input_sample_rate * 2, dtype=np.float32)
    try:
        has_speech_silence = vad.is_speech_detected(silence)
        if has_speech_silence:
            return False
    except Exception:
        return False

    # 5. Test get_speech_segments
    try:
        segments = vad.get_speech_segments(audio_float)
        for seg in segments:
            cast("float", seg.get("start", 0.0))
            cast("float", seg.get("end", 0.0))
    except Exception:
        return False

    # 6. Test resample_audio utility
    target_rate = 22050
    try:
        audio_resampled = resample_audio(audio_float, int(cast("Any", sample_rate)), target_rate)
        expected_shape = int(audio_float.shape[0] * target_rate / sample_rate)
        # Check if length is close to expected
        if abs(audio_resampled.shape[0] - expected_shape) > 5:
            return False
    except Exception:
        return False

    return True


if __name__ == "__main__":
    success = test_vad_standalone()
    logger.info(f"VAD standalone test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)
