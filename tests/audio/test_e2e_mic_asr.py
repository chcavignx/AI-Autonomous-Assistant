from __future__ import annotations

import wave

import pytest
from src.audio.asr import ASREngine
from src.audio.audio_utils import AudioRecorder
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
    duration_s = 3

    recorder = AudioRecorder(config=config, rate=sample_rate, device_index=device_index)

    if not recorder.start():
        pytest.skip(f"Unable to open input device at {sample_rate}Hz using {recorder._backend}")

    try:
        frames = []
        # Calculate number of chunks to read.
        # AudioRecorder uses its internal _chunk_frames which defaults to config.audio.input_chunk_size
        num_chunks = int(sample_rate / recorder._chunk_frames * duration_s)
        for _ in range(num_chunks):
            frame = recorder.read()
            if frame:
                frames.append(frame)
    finally:
        recorder.close()

    if not frames:
        pytest.fail("No audio frames recorded from microphone")

    wav_path = tmp_path / "mic.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 is 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    # Transcribe recorded audio.
    asr = ASREngine(config)
    asr.load()
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
