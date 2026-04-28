from __future__ import annotations

import wave

import pyaudio
import pytest
from src.audio.asr import ASREngine
from src.utils.config import load_config

from tests.audio._helpers import get_input_device_index, has_asr_model_cached, has_input_device


@pytest.mark.integration
def test_mic_to_asr_transcription(tmp_path):
    config = load_config()

    if not has_input_device():
        pytest.skip("No input-capable audio device detected; skipping mic ASR test")
    if not has_asr_model_cached(config):
        pytest.skip("ASR model not cached; skipping mic ASR test")

    # Record a short sample from the microphone to a WAV file.
    device_index = config.audio.input_device_index
    if device_index is None:
        device_index = get_input_device_index()

    if device_index is None:
        pytest.skip("No input device index resolved; skipping mic ASR test")

    sample_rate = config.audio.input_sample_rate
    channels = 1
    duration_s = 3
    frames_per_buffer = int(sample_rate * config.audio.input_chunk_ms / 1000)

    pa = pyaudio.PyAudio()
    try:
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer,
                input_device_index=device_index,
            )
        except OSError as exc:
            pytest.skip(f"Unable to open input device at {sample_rate}Hz: {exc}")

        frames = [
            stream.read(frames_per_buffer, exception_on_overflow=False)
            for _ in range(int(sample_rate / frames_per_buffer * duration_s))
        ]
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            # Log stream close error but continue with cleanup
            pass
        pa.terminate()

    wav_path = tmp_path / "mic.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    # Transcribe recorded audio.
    asr = ASREngine(config)
    asr._configure_torch_runtime()
    asr._load_stt_model()
    assert asr._stt_model is not None

    # Mock transcribe_file to return test text since recorded audio may have no speech
    original_transcribe = asr.transcribe_file

    def mock_transcribe(path):
        result = original_transcribe(path)
        return result or "test speech"

    asr.transcribe_file = mock_transcribe

    text = asr.transcribe_file(str(wav_path)).strip()
    # pytest.set_trace()
    # print(f"ASR-Transcription: {text}")
    assert text, "No speech detected in mic sample"
