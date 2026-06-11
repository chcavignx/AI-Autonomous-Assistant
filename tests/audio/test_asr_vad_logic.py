from __future__ import annotations

import numpy as np
import pytest
from src.audio.asr import ASREngine
from src.utils.config import load_config


@pytest.mark.basic
def _chunk_from_int16(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def test_energy_based_vad_detects_loud_chunk() -> None:
    chunk = _chunk_from_int16([0, 0, 5000, -5000, 0])
    assert ASREngine._energy_based_vad(chunk)


def test_detect_speech_falls_back_when_vad_missing() -> None:
    config = load_config()
    asr = ASREngine(config)
    asr._vad_model = None
    chunk = _chunk_from_int16([0, 0, 5000, -5000, 0])
    assert asr._detect_speech(chunk) == ASREngine._energy_based_vad(chunk)
