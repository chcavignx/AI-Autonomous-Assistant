from __future__ import annotations

import re
import wave

import pytest
from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.utils.config import load_config

from tests.audio._helpers import has_asr_model_cached, has_tts_model


@pytest.mark.integration
def test_tts_to_wav_to_asr_transcription(tmp_path):
    config = load_config()

    if not has_tts_model(config):
        pytest.skip("TTS model not found in cache; skipping E2E")
    if not has_asr_model_cached(config):
        pytest.skip("ASR model not cached; skipping E2E")

    # Synthesize WAV bytes by rendering to a real WAV file handle.
    tts = TTSEngine(config)
    tts.load()

    wav_path = tmp_path / "tts.wav"
    with wave.open(str(wav_path), "wb") as wf:
        tts._piper_voice.synthesize_wav(
            text="hello world",
            wav_file=wf,
            set_wav_format=True,
        )

    # Load only STT model (avoid VAD download).
    asr = ASREngine(config)
    asr._configure_torch_runtime()
    asr._load_stt_model()
    assert asr._stt_model is not None

    text = asr.transcribe_file(str(wav_path)).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    assert "hello" in text or "world" in text
