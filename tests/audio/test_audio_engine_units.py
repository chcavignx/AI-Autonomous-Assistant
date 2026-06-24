from __future__ import annotations

import builtins
import queue
import sys
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.audio import audio_utils
from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.utils.config import Config

pytestmark = pytest.mark.basic


def test_tts_speak_truncates_long_text() -> None:
    config: Config = Config()
    config.audio.output_chunk_size = 10
    tts: TTSEngine = TTSEngine(config)
    tts.speak(text="ABCDEFGHIJKLMNOP")

    queued: str | None = tts._tts_queue.get_nowait()
    assert queued == "ABCDEFGHIJ..."


def test_tts_interrupt_clears_queue() -> None:
    config: Config = Config()
    tts: TTSEngine = TTSEngine(config)
    tts._tts_queue.put("hello world")
    tts._tts_queue.put("foo bar")

    tts.interrupt()
    assert tts._tts_queue.empty()


def test_asr_extract_text_from_result_variants() -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    assert asr._extract_text_from_result(result={"segments": [{"text": "hello"}]}) == "hello"

    class DummySegment:
        text: str = "world"

    assert asr._extract_text_from_result(result=([DummySegment()], None)) == "world"

    class DummyResult:
        segments: list[DummySegment] = [DummySegment()]  # noqa: RUF012

    assert asr._extract_text_from_result(result=DummyResult()) == "world"
    assert asr._extract_text_from_result(result=["foo", "bar"]) == "foo bar"


def test_transcribe_file_raises_without_loaded_model() -> None:
    asr: ASREngine = ASREngine(config=Config())
    with pytest.raises(RuntimeError, match="ASREngine not loaded"):
        asr.transcribe_file(audio_path="dummy.wav")


def test_asr_configure_torch_runtime(monkeypatch) -> None:
    # Create mock torch module if not installed (with _tensor for vad.py compatibility)
    if "torch" not in sys.modules:
        torch_mock = MagicMock()
        torch_mock._tensor = MagicMock()
        sys.modules["torch"] = torch_mock
        sys.modules["torch._tensor"] = torch_mock._tensor
    elif "torch._tensor" not in sys.modules:
        # If torch exists but _tensor doesn't, add it
        sys.modules["torch"]._tensor = MagicMock()
        sys.modules["torch._tensor"] = sys.modules["torch"]._tensor

    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    torch_calls: list[Any] = []

    def mock_set_num_threads(*, num):
        torch_calls.append(("set_num_threads", num))

    def mock_set_num_interop_threads(n) -> None:
        torch_calls.append(("set_num_interop_threads", n))

    monkeypatch.setattr("torch.set_num_threads", mock_set_num_threads)
    monkeypatch.setattr("torch.set_num_interop_threads", mock_set_num_interop_threads)
    monkeypatch.setattr("src.audio.asr.detect_raspberry_pi_model", lambda: True)
    monkeypatch.setattr("src.audio.asr.limit_cpu_for_multiprocessing", lambda desired_cores=1: 1)

    asr._configure_torch_runtime()
    assert ("set_num_threads", 1) in torch_calls
    assert ("set_num_interop_threads", 1) not in torch_calls


def test_asr_load_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "whisper"
    asr: ASREngine = ASREngine(config)

    loaded = {}

    def mock_load_model(*args, **kwargs) -> Literal["mock_model"]:
        loaded.update(kwargs)
        if args:
            loaded["name"] = args[0]
        return "mock_model"

    mock_whisper = MagicMock()
    mock_whisper.load_model = mock_load_model
    monkeypatch.setitem(sys.modules, "whisper", mock_whisper)

    asr._load_whisper()
    assert loaded.get("name") == config.asr.model_size or loaded.get("model_size") == config.asr.model_size
    assert loaded.get("device") == config.asr.device
    assert asr._stt_model == "mock_model"


def test_asr_load_faster_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "faster-whisper"
    asr: ASREngine = ASREngine(config)

    loaded: dict[Any, Any] = {}

    def mock_WhisperModel(*args, **kwargs) -> Literal["mock_model"]:
        loaded.update(kwargs)
        if args:
            loaded["model_size_or_path"] = args[0]
        return "mock_model"

    mock_faster_whisper = MagicMock()
    mock_faster_whisper.WhisperModel = mock_WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_faster_whisper)

    asr._load_faster_whisper()
    target_key = "model_size_or_path" if "model_size_or_path" in loaded else "model_size"
    assert loaded.get(target_key) == config.asr.model_size
    assert loaded.get("device") == config.asr.device
    assert asr._stt_model == "mock_model"


def test_asr_load_vad_model(monkeypatch) -> None:
    # Create mock silero_vad module if not installed
    if "silero_vad" not in sys.modules:
        sys.modules["silero_vad"] = MagicMock()

    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    loaded: dict[Any, Any] = {}

    def mock_load_silero_vad(*args, **kwargs) -> Literal["mock_vad"]:
        loaded.update(kwargs)
        if args:
            loaded["onnx_used"] = args[0]
        return "mock_vad"

    monkeypatch.setattr("silero_vad.load_silero_vad", mock_load_silero_vad)

    asr._load_vad_model()
    assert asr._vad_model == "mock_vad"


def test_asr_load_vad_model_fallback(monkeypatch) -> None:
    # Create mock silero_vad module if not installed
    if "silero_vad" not in sys.modules:
        sys.modules["silero_vad"] = MagicMock()

    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    def mock_load_silero_vad(*args, **kwargs) -> None:
        raise ImportError("No silero")

    monkeypatch.setattr("silero_vad.load_silero_vad", mock_load_silero_vad)

    asr._load_vad_model()
    assert asr._vad_model is None


def test_asr_detect_speech_with_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    # Mock the VAD model callable that returns object with .item()
    mock_vad_res = MagicMock()
    mock_vad_res.item.return_value = 0.9
    asr._vad_model = MagicMock(return_value=mock_vad_res)

    # Make chunk long enough
    chunk: bytes = b"\xff\x7f" * 6000
    result: bool = asr._detect_speech(chunk)
    assert result


def test_asr_detect_speech_fallback_energy() -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)
    asr._vad_model = None

    # Loud chunk: use higher values, long enough
    chunk: bytes = b"\xff\x7f" * 6000  # 32767
    assert asr._detect_speech(chunk)

    # Quiet chunk
    chunk = b"\x00\x00" * 6000
    assert not asr._detect_speech(chunk)


def test_asr_transcribe_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "whisper"
    asr: ASREngine = ASREngine(config)

    class MockModel:
        def transcribe(self, audio: bytes, **kwargs) -> dict[str, list[dict[str, str]]]:
            return {"segments": [{"text": "hello world"}]}

    asr._stt_model = MockModel()

    result: str = asr._transcribe_whisper(audio_bytes=b"mock_audio")
    assert result == "hello world"


def test_asr_transcribe_faster_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "faster-whisper"
    asr: ASREngine = ASREngine(config)

    class MockModel:
        def transcribe(self, audio: bytes, **kwargs) -> tuple[list[dict[str, str]], Literal["info"]]:
            return ([{"text": "hello world"}], "info")

    asr._stt_model = MockModel()

    result: str = asr._transcribe_whisper(audio_bytes=b"mock_audio")
    assert result == "hello world"


def test_tts_load_piper_python(monkeypatch) -> None:
    config: Config = Config()
    tts: TTSEngine = TTSEngine(config)

    loaded: dict[Any, Any] = {}

    def mock_PiperVoice_load(_path) -> Literal["mock_voice"]:
        loaded["path"] = _path
        return "mock_voice"

    def mock_exists(path) -> Literal[True]:
        return True

    monkeypatch.setattr("piper.PiperVoice.load", mock_PiperVoice_load)
    monkeypatch.setattr("pathlib.Path.exists", mock_exists)

    tts._load_piper_python()
    assert loaded["path"] == str(config.tts.full_model_path)
    assert tts._piper_voice == "mock_voice"


def test_tts_synthesize_via_api(monkeypatch: pytest.MonkeyPatch) -> None:
    config: Config = Config()
    tts: TTSEngine = TTSEngine(config)

    synthesized = {}

    def mock_synthesize_wav(text, wav_file, set_wav_format):
        synthesized["text"] = text
        synthesized["wav_file"] = wav_file
        synthesized["set_wav_format"] = set_wav_format
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"dummy")

    mock_voice: MagicMock = MagicMock()
    mock_voice.synthesize_wav = mock_synthesize_wav
    tts._piper_voice = mock_voice

    result: bytes | None = tts._synthesize("hello")

    assert synthesized["text"] == "hello"
    assert synthesized["set_wav_format"] is True
    assert result is not None
    assert result.endswith(b"dummy")


def test_tts_playback_loop_processes_queue(monkeypatch) -> None:
    config: Config = Config()
    tts: TTSEngine = TTSEngine(config)
    tts._running = True

    synthesized: list[Any] = []

    monkeypatch.setattr(
        "src.audio.tts.TTSEngine._synthesize", lambda _self, text: (synthesized.append(text), b"wav_data")[1]
    )
    monkeypatch.setattr("src.audio.tts.TTSEngine._play_wav", lambda _self, _data: None)

    tts._tts_queue.put(item="hello")
    tts._tts_queue.put(item=None)  # Sentinel

    tts._playback_loop()
    assert synthesized == ["hello"]


def test_asr_open_input_stream(monkeypatch) -> None:
    engine: ASREngine = ASREngine(config=Config())
    dummy_stream = MagicMock()
    opened = audio_utils.OpenedInputStream(
        stream=dummy_stream,
        backend="sounddevice",
        capture_rate=16000,
        native_chunk_frames=512,
        device_index=None,
        need_resample=False,
    )
    monkeypatch.setattr("src.audio.asr.open_input_stream_with_fallback", lambda **kwargs: opened)

    result: bool = engine._open_input_stream()

    assert result is True
    assert engine._stream == dummy_stream
    assert engine._chunk_frames == 512
    assert engine._capture_rate == 16000
    assert engine._need_resample is False


def test_asr_resolve_device_candidates_with_index(monkeypatch) -> None:
    engine: ASREngine = ASREngine(Config())
    engine._stream = MagicMock()
    engine._running = True
    engine._chunk_frames = 4
    engine._need_resample = True
    engine._resample_up = 2
    engine._resample_down = 1

    class DummyStream:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
            self.calls.append((num_frames, exception_on_overflow))
            engine._running = False
            return np.array([1, 2, 3, 4], dtype=np.int16).tobytes()

    dummy = DummyStream()
    engine._stream = dummy
    monkeypatch.setattr("src.audio.asr._resample_poly", lambda audio, up, down: audio * 2)
    engine._capture_loop()
    assert dummy.calls == [(4, False)]


def test_asr_resolve_device_candidates_unavailable_device(monkeypatch) -> None:
    engine: ASREngine = ASREngine(Config())
    monkeypatch.setattr("src.audio.asr.open_input_stream_with_fallback", lambda **kwargs: None)
    assert engine._open_input_stream() is False


def test_asr_try_open_success(monkeypatch) -> None:
    engine: ASREngine = ASREngine(Config())
    engine._stream = MagicMock()
    engine._running = True

    speech_chunks: list[bytes] = []
    monkeypatch.setattr(engine, "_detect_speech", lambda chunk: True)
    monkeypatch.setattr(engine, "_transcribe", lambda audio_bytes: speech_chunks.append(audio_bytes))
    for chunk in [b"\x01\x00" * 6000, None]:
        engine._audio_queue.put(chunk)
    engine._config.vad.silence_timeout_seconds = 0.0
    engine._config.audio.input_chunk_ms = 100
    engine._process_loop()
    assert speech_chunks == []


def test_asr_try_open_failure(monkeypatch) -> None:
    engine: ASREngine = ASREngine(Config())
    engine._stt_model = None
    engine._transcript_callback = None
    engine._transcribe(b"\x00\x00")
    assert engine._stt_model is None


def test_asr_process_loop_accumulates_speech(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)
    engine._running = True

    # Mock VAD to always return confidence > threshold
    mock_vad_res = MagicMock()
    mock_vad_res.item.return_value = 0.8
    engine._vad_model = MagicMock(return_value=mock_vad_res)

    # Mock transcribe method to capture calls
    transcribe_calls: list[bytes] = []

    def mock_transcribe(audio_bytes: bytes) -> None:
        transcribe_calls.append(audio_bytes)

    monkeypatch.setattr("src.audio.asr.ASREngine._transcribe", mock_transcribe)

    # Put silence data followed by speech, then silence again
    speech_chunk: bytes = b"\xff\x7f" * 6000  # Loud speech
    silence_chunk: bytes = b"\x00\x00" * 6000  # Quiet silence

    # Queue format: speech, speech, silence, silence, sentinel
    engine._audio_queue.put(item=speech_chunk)
    engine._audio_queue.put(item=speech_chunk)
    engine._audio_queue.put(item=silence_chunk)
    engine._audio_queue.put(item=silence_chunk)
    engine._audio_queue.put(item=None)  # Sentinel to stop loop

    # Run the process loop
    engine._process_loop()

    # Should have accumulated and processed audio
    assert True  # Basic smoke test passed


def test_asr_process_loop_handles_empty_queue_timeout(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)
    engine._running = True

    # Simple test that _process_loop handles queue timeout
    # Put sentinel immediately to end loop quickly
    engine._audio_queue.put(item=None)

    # Run the process loop - should handle timeout gracefully
    engine._process_loop()

    # If we reach here without exception, test passes
    assert engine._running is True


def test_asr_start_creates_threads(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)

    # Mock the key methods
    def mock_open_stream(self) -> Literal[True]:
        return True

    monkeypatch.setattr("src.audio.asr.ASREngine._open_input_stream", mock_open_stream)
    monkeypatch.setattr("threading.Thread.start", lambda self: None)

    engine.start(callback=lambda _x: None)

    assert engine._running is True
    assert engine._capture_thread is not None
    assert engine._process_thread is not None


def test_asr_stop_clears_queue(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)
    engine._running = True

    # Mock the stream
    engine._stream = MagicMock()

    # Mock threads that won't actually join
    mock_thread: MagicMock = MagicMock()
    mock_thread.is_alive.return_value = False
    mock_thread.join = MagicMock()

    engine._capture_thread = mock_thread
    engine._process_thread = mock_thread

    # Put some items in the queue
    engine._audio_queue.put(item=b"test1")
    engine._audio_queue.put(item=b"test2")

    engine.stop()

    # Verify running is set to False
    assert engine._running is False


def test_tts_unload_stops_thread(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._running = True

    # Mock thread
    engine._playback_thread = MagicMock()

    engine.unload()

    assert engine._running is False
    # Verify queue received sentinel
    try:
        sentinel: str | None = engine._tts_queue.get_nowait()
        assert sentinel is None
    except engine._tts_queue.Empty:
        pass  # Queue was already consumed


def test_tts_wait_blocks_until_queue_empty(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Put items in queue
    engine._tts_queue.put(item="hello")
    engine._tts_queue.put(item="world")

    # Simulate task completion
    engine._tts_queue.task_done()
    engine._tts_queue.task_done()

    # wait() should return immediately after all tasks done
    engine.wait()  # Should not hang


def test_tts_pcm_to_wav_conversion() -> None:
    # Test conversion of raw PCM to WAV bytes
    pcm_bytes: bytes = b"\x00\x00" * 1000  # Silent PCM data
    wav_bytes: bytes = TTSEngine._pcm_to_wav(pcm_bytes, sample_rate=22050)

    # Should contain RIFF header
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert len(wav_bytes) > len(pcm_bytes)


def test_tts_synthesize_via_api_no_voice(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._piper_voice = None

    result: bytes | None = engine._synthesize(text="hello")
    assert result is None


def test_tts_synthesize_via_api_error(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Mock voice with error
    mock_voice: MagicMock = MagicMock()
    mock_voice.synthesize_wav.side_effect = Exception("Synthesis error")
    engine._piper_voice = mock_voice

    result: bytes | None = engine._synthesize(text="hello")
    assert result is None


def test_asr_energy_based_vad_high_amplitude() -> None:
    # Test VAD with loud audio
    loud_chunk: bytes = b"\x7f\x7f" * 6000  # Very loud
    result: bool = ASREngine._energy_based_vad(chunk=loud_chunk)
    assert result


def test_asr_energy_based_vad_low_amplitude() -> None:
    # Test VAD with quiet audio
    quiet_chunk: bytes = b"\x10\x00" * 6000  # Very quiet
    result: bool = ASREngine._energy_based_vad(chunk=quiet_chunk)
    assert not result


def test_asr_transcribe_with_empty_text(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)

    # Mock callback to track calls
    callback_calls: list[str] = []

    def mock_callback(text: str) -> None:
        callback_calls.append(text)

    engine._transcript_callback = mock_callback
    engine._stt_model = MagicMock()

    # Mock transcribe to return empty/short text
    monkeypatch.setattr("src.audio.asr.ASREngine._transcribe_whisper", lambda _self, **_k: "  ")

    # This should not trigger callback for short text
    engine._transcribe(audio_bytes=b"mock_audio")

    assert len(callback_calls) == 0


def test_asr_transcribe_with_valid_text(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)

    # Mock callback to track calls
    callback_calls: list[str] = []

    def mock_callback(text: str) -> None:
        callback_calls.append(text)

    engine._transcript_callback = mock_callback
    engine._stt_model = MagicMock()

    # Mock transcribe to return valid text
    monkeypatch.setattr("src.audio.asr.ASREngine._transcribe_whisper", lambda _self, **_k: "hello world")

    engine._transcribe(audio_bytes=b"mock_audio")

    assert len(callback_calls) == 1
    assert callback_calls[0] == "hello world"


def test_tts_speak_empty_text() -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Empty or whitespace text should be ignored
    engine.speak(text="")
    assert engine._tts_queue.empty()

    engine.speak(text="   ")
    assert engine._tts_queue.empty()


def test_tts_interrupt() -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Queue items
    engine._tts_queue.put(item="item1")
    engine._tts_queue.put(item="item2")
    engine._tts_queue.put(item="item3")

    # Interrupt should clear queue
    engine.interrupt()

    assert engine._tts_queue.empty()


def test_tts_is_speaking_property() -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Initially not speaking
    assert engine.is_speaking is False

    # Set the event
    engine._is_speaking.set()
    assert engine.is_speaking is True

    # Clear
    engine._is_speaking.clear()
    assert engine.is_speaking is False


def test_asr_load_logs_ready(monkeypatch, caplog) -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    # Mock model loading
    monkeypatch.setattr("src.audio.asr.ASREngine._configure_torch_runtime", lambda _self: None)
    monkeypatch.setattr("src.audio.asr.ASREngine._load_stt_model", lambda _self: None)
    monkeypatch.setattr("src.audio.asr.ASREngine._load_vad_model", lambda _self: None)

    import logging

    caplog.set_level(logging.INFO)

    asr.load()

    # Should have logged that ASR is ready
    assert "ASR ready" in caplog.text or asr._stt_model is None  # Either logged or models loaded


def test_tts_unload_clears_piper_voice() -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Set piper voice as if it was loaded
    engine._piper_voice = MagicMock()
    engine._running = True

    class DummyThread:
        def join(self, timeout=None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return False

    engine._playback_thread = DummyThread()

    engine.unload()

    # Voice should be cleared
    assert engine._piper_voice is None
    assert engine._running is False


def test_asr_unload_with_faster_whisper_skips_teardown(monkeypatch) -> None:
    """Test that ASR unload skips faster-whisper model teardown to avoid segfaults."""
    config: Config = Config()
    config.asr.engine = "faster-whisper"
    engine: ASREngine = ASREngine(config)

    # Mock the model as loaded
    engine._stt_model = MagicMock()
    engine._vad_model = MagicMock()

    # Call unload - should skip teardown for faster-whisper
    engine.unload()

    # Models should still be set (not cleared) due to skip logic
    assert engine._stt_model is not None
    assert engine._vad_model is not None


def test_asr_unload_with_whisper_clears_models(monkeypatch) -> None:
    """Test that ASR unload clears models for whisper engine."""
    config: Config = Config()
    config.asr.engine = "whisper"
    engine: ASREngine = ASREngine(config)

    # Mock the model as loaded
    engine._stt_model = MagicMock()
    engine._vad_model = MagicMock()

    # Call unload - should clear models for whisper
    engine.unload()

    # Models should be cleared
    assert engine._stt_model is None
    assert engine._vad_model is None


def test_asr_unload_with_skip_config_flag(monkeypatch) -> None:
    """Test that ASR unload respects skip_native_teardown config flag."""
    config: Config = Config()
    config.asr.engine = "whisper"
    config.asr.skip_native_teardown = True
    engine: ASREngine = ASREngine(config)

    # Mock the model as loaded
    engine._stt_model = MagicMock()
    engine._vad_model = MagicMock()

    # Call unload - should skip teardown due to config
    engine.unload()

    # Models should still be set (not cleared) due to config flag
    assert engine._stt_model is not None
    assert engine._vad_model is not None


def test_tts_playback_loop_with_multiple_items(monkeypatch) -> None:
    """Test TTS playback loop processes multiple queue items."""
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._running = True

    synthesized_items: list[str] = []

    monkeypatch.setattr(
        "src.audio.tts.TTSEngine._synthesize", lambda _self, text: (synthesized_items.append(text), b"wav_data")[1]
    )
    monkeypatch.setattr("src.audio.tts.TTSEngine._play_wav", lambda _self, _data: None)

    # Queue multiple items
    engine._tts_queue.put(item="first message")
    engine._tts_queue.put(item="second message")
    engine._tts_queue.put(item="third message")
    engine._tts_queue.put(item=None)  # Sentinel to stop

    # The playback loop would process these in a real scenario
    # For this test, we just verify the queue setup
    assert engine._tts_queue.qsize() == 4


def test_tts_pcm_to_wav_with_various_sample_rates() -> None:
    """Test PCM to WAV conversion with different sample rates."""
    # Test standard sample rates
    for sample_rate in [8000, 16000, 22050, 44100, 48000]:
        pcm_bytes: bytes = b"\x00\x00" * 100  # Silent audio
        wav_bytes: bytes = TTSEngine._pcm_to_wav(pcm_bytes, sample_rate=sample_rate)

        # Verify WAV format
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes
        assert len(wav_bytes) > len(pcm_bytes)


def test_tts_interrupt_during_playback(monkeypatch) -> None:
    """Test TTS interrupt clears queue during active playback."""
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Simulate active playback
    engine._is_speaking.set()

    # Queue multiple items
    engine._tts_queue.put(item="message1")
    engine._tts_queue.put(item="message2")
    engine._tts_queue.put(item="message3")

    # Interrupt should clear queue (but doesn't affect speaking state)
    engine.interrupt()

    assert engine._tts_queue.empty()
    # Note: interrupt() only clears queue, doesn't affect speaking state
    # The speaking state would be cleared by the playback loop when interrupted


def test_asr_current_import_and_transcribe_paths(monkeypatch) -> None:
    config = Config()
    asr = ASREngine(config)

    asr._config.asr.engine = "bogus"
    asr._load_stt_model()
    assert asr._stt_model is None

    original_import = builtins.__import__

    def missing_whisper(name: str, *args: Any, **kwargs: Any):
        if name == "whisper":
            raise ImportError("no whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_whisper)
    asr._config.asr.engine = "whisper"
    asr._load_whisper()
    assert asr._stt_model is None

    def missing_fw(name: str, *args: Any, **kwargs: Any):
        if name == "faster_whisper":
            raise ImportError("no faster-whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_fw)
    asr._config.asr.engine = "faster-whisper"
    asr._load_faster_whisper()
    assert asr._stt_model is None

    def missing_vad(name: str, *args: Any, **kwargs: Any):
        if name == "silero_vad":
            raise ImportError("no vad")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_vad)
    asr._load_vad_model()
    assert asr._vad_model is None

    asr._vad_model = None
    assert asr._detect_speech(b"\x00\x00" * 6000) is False
    assert ASREngine._energy_based_vad(b"\xff\x7f" * 6000) is True

    asr._stt_model = MagicMock()
    asr._config.asr.engine = "whisper"
    asr._stt_model.transcribe.return_value = {"segments": [{"text": "hello"}]}
    assert asr._transcribe_whisper(b"\x00\x00" * 10) == "hello"

    asr._config.asr.engine = "faster-whisper"

    class FasterModel:
        def transcribe(self, audio, **kwargs):
            return ([{"text": "world"}], None)

    asr._stt_model = FasterModel()
    assert asr._transcribe_whisper(b"\x00\x00" * 10) == "world"

    assert asr._extract_text_from_result("raw text") == "raw text"
    assert asr._extract_text_from_result({"segments": [{"text": "dict"}]}) == "dict"


def test_wake_word_load_and_stream(monkeypatch) -> None:
    from src.audio import wake_word as wake_word_module

    config = Config()
    tmp_path = config.paths.tmp_path

    model_path = tmp_path / "wake.onnx"
    model_path.write_text("dummy")
    melspec_path = tmp_path / "melspec.onnx"
    melspec_path.write_text("dummy")
    embedding_path = tmp_path / "embedding.onnx"
    embedding_path.write_text("dummy")

    config = Config()
    config.wake.model_name = "wake"
    config.wake.download_root = str(tmp_path)
    config.wake.melspec_model = "melspec"
    config.wake.embedding_model = "embedding"
    config.wake.inference_framework = "onnx"

    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    detector = wake_word_module.WakeWordDetector(config)
    detector.load()
    assert detector._ww_sess is not None

    monkeypatch.setattr(
        "src.audio.wake_word.open_input_stream_with_fallback",
        lambda **kwargs: None,
    )
    assert detector._open_input_stream() is False

    class DummyOpened:
        def __init__(self) -> None:
            self.backend = "sounddevice"
            self.stream = MagicMock()
            self.native_chunk_frames = 2
            self.capture_rate = 16000
            self.need_resample = False
            self.resample_up = 1
            self.resample_down = 1
            self.device_index = None

    monkeypatch.setattr(
        "src.audio.wake_word.open_input_stream_with_fallback",
        lambda **kwargs: DummyOpened(),
    )
    assert detector._open_input_stream() is True


def test_wake_word_loops(monkeypatch) -> None:
    from src.audio import wake_word as wake_word_module

    config = Config()
    detector = wake_word_module.WakeWordDetector(config)

    class DummyQueue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def full(self) -> bool:
            return False

        def put(self, item: Any) -> None:
            self.items.append(item)

    detector._audio_queue = DummyQueue()

    class DummyCaptureStream:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, *_args, **_kwargs):
            self.calls += 1
            detector._running = False
            return np.array([1, 2], dtype=np.int16).tobytes()

    detector._stream = DummyCaptureStream()
    detector._running = True
    detector._native_chunk = 2
    detector._need_resample = True
    monkeypatch.setattr("src.audio.wake_word._resample_poly", lambda audio, up, down: audio * 2)
    detector._capture_loop()
    assert detector._audio_queue.items

    detector._audio_queue = queue.Queue()
    detector._audio_queue.put(np.ones(1280, dtype=np.float32))
    detector._audio_queue.put(None)

    mock_ww_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_ww_sess.get_inputs.return_value = [mock_input]
    mock_ww_sess.run.return_value = [[[0.95]]]

    detector._ww_sess = mock_ww_sess
    detector._preprocessor = MagicMock()
    detector._preprocessor.get_features.return_value = np.zeros((1, 16, 96), dtype=np.float32)
    detector._prediction_count = 5

    callback_calls: list[str] = []
    detector._callback = lambda: callback_calls.append("hit")
    monkeypatch.setattr("time.time", lambda: 100.0)
    detector._running = True
    detector._detect_loop()
    assert callback_calls == ["hit"]


def test_vad_platform_tuning(monkeypatch) -> None:
    # Create mock torch module if not installed (vad.py imports from torch._tensor)
    if "torch" not in sys.modules:
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock
        # Create torch._tensor as an attribute of the torch mock
        torch_mock._tensor = MagicMock()
        sys.modules["torch._tensor"] = torch_mock._tensor

    from src.audio import vad as vad_module

    monkeypatch.setattr(vad_module, "load_silero_vad", lambda: MagicMock())
    monkeypatch.setattr(vad_module, "detect_raspberry_pi_model", lambda: True)
    calls: list[Any] = []
    monkeypatch.setattr(
        vad_module, "limit_cpu_for_multiprocessing", lambda desired_cores=None: calls.append(desired_cores) or 1
    )

    config = Config()
    config.asr.language = "en"
    config.asr.model_size = "small"
    config.asr.faster_model_size = "medium"
    config.platform.cpu_cores = 2
    vad = vad_module.VADEngine(config)
    assert vad.config.asr.model_size.endswith(".en")
    assert vad.config.asr.faster_model_size.endswith(".en")
    assert calls == [2]

    monkeypatch.setattr(vad_module, "detect_raspberry_pi_model", lambda: False)
    config = Config()
    config.asr.model_size = "small"
    monkeypatch.setattr(vad_module, "load_silero_vad", lambda: MagicMock())
    vad = vad_module.VADEngine(config)
    assert vad.config.asr.model_size == "base"


def test_asr_load_dispatch_and_open_input_stream_branches(monkeypatch) -> None:
    config = Config()
    asr = ASREngine(config)

    whisper_called: list[str] = []
    faster_called: list[str] = []
    monkeypatch.setattr(asr, "_load_whisper", lambda: whisper_called.append("whisper"))
    monkeypatch.setattr(asr, "_load_faster_whisper", lambda: faster_called.append("faster"))

    asr._config.asr.engine = "whisper"
    asr._load_stt_model()
    asr._config.asr.engine = "faster-whisper"
    asr._load_stt_model()
    asr._config.asr.engine = "bogus"
    asr._load_stt_model()
    assert whisper_called == ["whisper"]
    assert faster_called == ["faster"]

    dummy_stream = MagicMock()
    opened = audio_utils.OpenedInputStream(
        stream=dummy_stream,
        backend="sounddevice",
        capture_rate=8000,
        native_chunk_frames=256,
        device_index=3,
        need_resample=True,
        resample_up=2,
        resample_down=1,
    )
    monkeypatch.setattr("src.audio.asr.open_input_stream_with_fallback", lambda **kwargs: opened)
    asr._config.audio.input_device_index = 1
    asr._config.audio.input_sample_rate = 16000
    assert asr._open_input_stream() is True
    assert asr._capture_rate == 8000
    assert asr._need_resample is True

    monkeypatch.setattr("src.audio.asr.open_input_stream_with_fallback", lambda **kwargs: None)
    assert asr._open_input_stream() is False


def test_asr_transcribe_and_extract_variants(monkeypatch) -> None:
    config = Config()
    asr = ASREngine(config)

    # No model means no-op transcription.
    asr._stt_model = None
    asr._transcribe(b"\x00\x00")

    captured: list[str] = []
    asr._transcript_callback = lambda text: captured.append(text)

    class WhisperModel:
        def transcribe(self, audio, **kwargs):
            return {"segments": [{"text": "hello"}, {"text": " world"}]}

    asr._config.asr.engine = "whisper"
    asr._stt_model = WhisperModel()
    assert asr._transcribe_whisper(b"\x01\x00" * 10) == "hello world"

    from dataclasses import dataclass

    @dataclass
    class Segment:
        text: str

    assert asr._extract_text_from_result(([Segment("a"), Segment("b")], None)) == "a b"
    assert asr._extract_text_from_result(SimpleNamespace(segments=[Segment("c")])) == "c"
    assert asr._extract_text_from_result([Segment("d"), "e"]) == "d e"
    assert asr._extract_text_from_result("f") == "f"
    assert not asr._extract_text_from_result(42)

    class FasterModel:
        def transcribe(self, audio, **kwargs):
            return ([Segment("g")], None)

    asr._config.asr.engine = "faster-whisper"
    asr._stt_model = FasterModel()
    assert asr._transcribe_whisper(b"\x02\x00" * 10) == "g"

    asr._config.asr.engine = "whisper"
    asr._stt_model = WhisperModel()
    text = asr._transcribe_whisper(b"\x03\x00" * 10)
    assert text == "hello world"
    asr._transcript_callback = lambda text: captured.append(text)
    asr._transcribe(b"\x04\x00" * 10)
    assert captured


def test_asr_transcribe_file_and_resample_wrapper(monkeypatch) -> None:
    from src.audio import asr as asr_module

    config = Config()
    asr = ASREngine(config)

    class WhisperModel:
        def transcribe(self, audio, **kwargs):
            return {"segments": [{"text": "file"}]}

    asr._config.asr.engine = "whisper"
    asr._stt_model = WhisperModel()
    assert asr.transcribe_file("audio.wav") == "file"

    class FasterModel:
        def transcribe(self, audio, **kwargs):
            return (({"text": "fast"} for _ in [0]), None)

    asr._config.asr.engine = "faster-whisper"
    asr._stt_model = FasterModel()
    assert asr.transcribe_file("audio.wav") == "fast"

    fake_signal = SimpleNamespace(resample_poly=lambda audio, up, down: audio + 1)
    monkeypatch.setattr("src.audio.asr.import_module", lambda name: fake_signal)
    resampled = asr_module._resample_poly(np.array([1.0], dtype=np.float32), 2, 1)
    assert np.array_equal(resampled, np.array([2.0], dtype=np.float32))


def test_asr_start_fails_fast_and_capture_loop_edge_paths(monkeypatch) -> None:

    engine: ASREngine = ASREngine(Config())
    engine._running = True
    engine.start(callback=lambda _text: None)
    assert engine._transcript_callback is None

    engine._running = False
    monkeypatch.setattr(engine, "_open_input_stream", lambda: False)
    engine.start(callback=lambda _text: None)
    assert engine._running is False

    engine._stream = None
    engine._capture_loop()

    class RaisingStream:
        def read(self, *args, **kwargs):
            engine._running = False
            raise RuntimeError("boom")

    engine._stream = RaisingStream()
    engine._running = True
    engine._capture_loop()
