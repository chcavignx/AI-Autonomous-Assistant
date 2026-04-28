from __future__ import annotations

import pathlib

import pytest
from src.audio.asr import ASREngine
from src.utils.config import load_config

from tests.audio._helpers import has_asr_model_cached


@pytest.mark.integration
def test_repository_sample_audio_transcribes():
    config = load_config()
    if not has_asr_model_cached(config):
        pytest.skip("ASR model not cached; skipping sample audio transcription")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    audio_path = repo_root / "data" / "test.wav"
    assert audio_path.exists(), f"Sample audio not found at {audio_path}"

    asr = ASREngine(config)
    asr._configure_torch_runtime()
    asr._load_stt_model()
    assert asr._stt_model is not None

    text = asr.transcribe_file(str(audio_path)).strip()
    assert text, "Expected transcription from sample audio file"
