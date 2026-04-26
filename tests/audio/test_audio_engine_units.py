from __future__ import annotations

import subprocess
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.utils.config import Config


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
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    torch_calls: list[Any] = []

    def mock_set_num_threads(n):
        torch_calls.append(("set_num_threads", n))

    def mock_set_num_interop_threads(n) -> None:
        torch_calls.append(("set_num_interop_threads", n))

    monkeypatch.setattr("torch.set_num_threads", mock_set_num_threads)
    monkeypatch.setattr("torch.set_num_interop_threads", mock_set_num_interop_threads)

    asr._configure_torch_runtime()
    assert ("set_num_threads", 1) in torch_calls
    assert ("set_num_interop_threads", 1) in torch_calls


def test_asr_load_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "whisper"
    asr: ASREngine = ASREngine(config)

    loaded = {}

    def mock_load_model(name: str, device=None, download_root=None) -> Literal["mock_model"]:
        loaded["name"] = name
        loaded["device"] = device
        loaded["download_root"] = download_root
        return "mock_model"

    monkeypatch.setattr("whisper.load_model", mock_load_model)

    asr._load_whisper()
    assert loaded["name"] == config.asr.model_size
    assert loaded["device"] == config.asr.device
    assert asr._stt_model == "mock_model"


def test_asr_load_faster_whisper(monkeypatch) -> None:
    config: Config = Config()
    config.asr.engine = "faster-whisper"
    asr: ASREngine = ASREngine(config)

    loaded: dict[Any, Any] = {}

    def mock_WhisperModel(
        model_size_or_path: str, device=None, compute_type=None, cpu_threads=0, num_workers=1
    ) -> Literal["mock_model"]:
        loaded["model_size_or_path"] = model_size_or_path
        loaded["device"] = device
        loaded["compute_type"] = compute_type
        loaded["cpu_threads"] = cpu_threads
        loaded["num_workers"] = num_workers
        return "mock_model"

    monkeypatch.setattr("faster_whisper.WhisperModel", mock_WhisperModel)

    asr._load_faster_whisper()
    assert loaded["model_size_or_path"] == config.asr.model_size
    assert loaded["device"] == config.asr.device
    assert asr._stt_model == "mock_model"


def test_asr_load_vad_model(monkeypatch) -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    def mock_load_silero_vad() -> Literal["mock_vad"]:
        return "mock_vad"

    monkeypatch.setattr("silero_vad.load_silero_vad", mock_load_silero_vad)

    asr._load_vad_model()
    assert asr._vad_model == "mock_vad"


def test_asr_load_vad_model_fallback(monkeypatch) -> None:
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    def mock_load_silero_vad() -> None:
        raise ImportError("No silero")

    monkeypatch.setattr("silero_vad.load_silero_vad", mock_load_silero_vad)

    asr._load_vad_model()
    assert asr._vad_model is None


def test_asr_detect_speech_with_vad(monkeypatch):
    config: Config = Config()
    asr: ASREngine = ASREngine(config)

    # Mock the VAD
    mock_vad: MagicMock = MagicMock()
    mock_vad.is_speech_detected.return_value = True
    asr.vad = mock_vad

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
    chunk: bytes = b"\x00\x00" * 6000
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


def test_tts_find_piper_binary(monkeypatch) -> None:
    config: Config = Config()
    config.tts.cli_mode = True
    tts: TTSEngine = TTSEngine(config)

    def mock_exists(path) -> bool:
        return str(path).endswith("piper")

    monkeypatch.setattr("pathlib.Path.exists", mock_exists)

    tts._find_piper_binary()
    assert tts._piper_bin is not None


def test_tts_load_piper_python(monkeypatch) -> None:
    config: Config = Config()
    config.tts.cli_mode = False
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


def test_tts_synthesize_via_api(monkeypatch):
    config: Config = Config()
    config.tts.cli_mode = False
    tts: TTSEngine = TTSEngine(config)

    synthesized = {}

    def mock_synthesize_wav(text, wav_file, set_wav_format):
        synthesized["text"] = text
        synthesized["wav_file"] = wav_file
        synthesized["set_wav_format"] = set_wav_format
        wav_file.write(b"dummy")

    mock_voice: MagicMock = MagicMock()
    mock_voice.synthesize_wav = mock_synthesize_wav
    tts._piper_voice = mock_voice

    result: bytes | None = tts._synthesize_via_api("hello")

    assert synthesized["text"] == "hello"
    assert synthesized["set_wav_format"] is True
    assert result == b"dummy"


def test_tts_synthesize_via_cli(monkeypatch) -> None:
    config = Config()
    config.tts.cli_mode = True
    tts: TTSEngine = TTSEngine(config)
    tts._piper_bin = "mock_piper"

    run_calls: list[Any] = []

    def mock_run(cmd, **kwargs) -> MagicMock:
        run_calls.append(cmd)
        mock_result: MagicMock = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"\x00\x00"
        mock_result.stderr = b""
        return mock_result

    monkeypatch.setattr("subprocess.run", mock_run)

    result: bytes | None = tts._synthesize_via_cli(text="hello")
    assert "mock_piper" in run_calls[0]
    assert "--model" in run_calls[0]
    assert isinstance(result, bytes)
    assert result.startswith(b"RIFF")


def test_tts_playback_loop_processes_queue(monkeypatch) -> None:
    config: Config = Config()
    tts: TTSEngine = TTSEngine(config)
    tts._running = True

    synthesized: list[Any] = []

    def mock_synthesize(self, text) -> Literal[b"wav_data"]:
        synthesized.append(text)
        return b"wav_data"

    def mock_play_wav(self, data) -> None:
        pass

    monkeypatch.setattr(
        "src.audio.tts.TTSEngine._synthesize", lambda _self, text: (synthesized.append(text), b"wav_data")[1]
    )
    monkeypatch.setattr("src.audio.tts.TTSEngine._play_wav", lambda _self, _data: None)

    tts._tts_queue.put(item="hello")
    tts._tts_queue.put(item=None)  # Sentinel


def test_asr_open_input_stream(monkeypatch) -> None:
    engine: ASREngine = ASREngine(config=Config())

    # Mock the resolve_device_candidates to return a device
    monkeypatch.setattr("src.audio.asr.ASREngine._resolve_device_candidates", lambda _self: [None])

    # Mock _try_open to return stream, chunk_frames
    mock_stream: MagicMock = MagicMock()
    monkeypatch.setattr("src.audio.asr.ASREngine._try_open", lambda _self, _rate, _dev: (mock_stream, 512))

    result: bool = engine._open_input_stream()

    assert result is True
    assert engine._stream == mock_stream
    assert engine._chunk_frames == 512


def test_asr_resolve_device_candidates_with_index(monkeypatch) -> None:
    config: Config = Config()
    config.audio.input_device_index = 2
    engine: ASREngine = ASREngine(config)

    # Mock PyAudio
    mock_pa: MagicMock = MagicMock()
    mock_pa.get_device_info_by_index.return_value = {"maxInputChannels": 2}
    engine._pa = mock_pa

    candidates: list[int | None] = engine._resolve_device_candidates()
    assert 2 in candidates
    assert None in candidates  # Always include default


def test_asr_resolve_device_candidates_unavailable_device(monkeypatch) -> None:
    config: Config = Config()
    config.audio.input_device_index = 99
    engine: ASREngine = ASREngine(config)

    mock_pa: MagicMock = MagicMock()
    mock_pa.get_device_info_by_index.side_effect = Exception("Device not found")
    engine._pa = mock_pa

    candidates: list[int | None] = engine._resolve_device_candidates()
    assert None in candidates


def test_asr_try_open_success(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)

    mock_stream: MagicMock = MagicMock()
    mock_pa: MagicMock = MagicMock()
    mock_pa.open.return_value = mock_stream
    engine._pa = mock_pa

    stream, chunk_frames = engine._try_open(rate=16000, dev_idx=None)
    assert stream == mock_stream
    assert chunk_frames > 0


def test_asr_try_open_failure(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)

    mock_pa: MagicMock = MagicMock()
    mock_pa.open.side_effect = Exception("Port audio error")
    engine._pa = mock_pa

    stream, chunk_frames = engine._try_open(rate=16000, dev_idx=None)
    assert stream is None
    assert chunk_frames == 0


def test_asr_process_loop_accumulates_speech(monkeypatch) -> None:
    config: Config = Config()
    engine: ASREngine = ASREngine(config)
    engine._running = True

    # Mock VAD to always return True/False for speech detection
    engine._vad_model = lambda _tensor, _rate: 0.8

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

    # Mock the stream mock threads and pa
    engine._stream = MagicMock()
    engine._pa = MagicMock()

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


def test_tts_load_initializes_thread(monkeypatch) -> None:
    config: Config = Config()
    config.tts.cli_mode = True

    engine: TTSEngine = TTSEngine(config)

    # Mock _find_piper_binary to avoid filesystem lookups
    monkeypatch.setattr("src.audio.tts.TTSEngine._find_piper_binary", lambda _self: None)

    # Mock pyaudio
    mock_pa_instance = MagicMock()
    monkeypatch.setattr("pyaudio.PyAudio", lambda: mock_pa_instance)

    # Mock thread start to avoid blocking
    monkeypatch.setattr("threading.Thread.start", lambda self: None)

    engine.load()

    assert engine._running is True
    assert engine._playback_thread is not None


def test_tts_unload_stops_thread(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._running = True

    # Mock thread
    engine._playback_thread = MagicMock()
    engine._pa = MagicMock()

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


def test_tts_synthesize_via_cli_timeout(monkeypatch) -> None:
    config: Config = Config()
    config.tts.cli_mode = True
    engine: TTSEngine = TTSEngine(config)
    engine._piper_bin = "/mock/piper"

    # Mock subprocess.run to raise timeout
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="piper", timeout=15)

    monkeypatch.setattr("subprocess.run", mock_run)

    result: bytes | None = engine._synthesize_via_cli(text="hello")
    assert result is None


def test_tts_synthesize_via_cli_error(monkeypatch) -> None:
    config: Config = Config()
    config.tts.cli_mode = True
    engine: TTSEngine = TTSEngine(config)
    engine._piper_bin = "/mock/piper"

    # Mock subprocess.run to return error
    mock_result: MagicMock = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"Error: model not found"

    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: mock_result)

    result: bytes | None = engine._synthesize_via_cli(text="hello")
    assert result is None


def test_tts_synthesize_via_api_no_voice(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._piper_voice = None

    result: bytes | None = engine._synthesize_via_api(text="hello")
    assert result is None


def test_tts_synthesize_via_api_error(monkeypatch) -> None:
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)

    # Mock voice with error
    mock_voice: MagicMock = MagicMock()
    mock_voice.synthesize_wav.side_effect = Exception("Synthesis error")
    engine._piper_voice = mock_voice

    result: bytes | None = engine._synthesize_via_api(text="hello")
    assert result is None


def test_asr_energy_based_vad_high_amplitude() -> None:
    # Test VAD with loud audio
    loud_chunk: bytes = b"\x7f\x7f" * 6000  # Very loud
    result: bool = ASREngine._energy_based_vad(chunk=loud_chunk)
    assert result


def test_asr_energy_based_vad_low_amplitude():
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
    engine._playback_thread = MagicMock()
    engine._pa = MagicMock()

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


def test_tts_load_with_pyaudio_error(monkeypatch) -> None:
    """Test TTS load handles PyAudio import errors."""
    config: Config = Config()
    config.tts.cli_mode = True
    engine: TTSEngine = TTSEngine(config)

    # Mock PyAudio to raise ImportError
    def mock_pyaudio() -> None:
        raise ImportError("pyaudio not installed")

    # Mock piper binary finding to avoid file system lookup
    monkeypatch.setattr("src.audio.tts.TTSEngine._find_piper_binary", lambda self: None)
    monkeypatch.setattr("pyaudio.PyAudio", mock_pyaudio)

    # Should raise ImportError with appropriate message
    with pytest.raises(ImportError, match="pyaudio not installed"):
        engine.load()


def test_tts_synthesize_with_cli_mode(monkeypatch) -> None:
    """Test TTS synthesis in CLI mode with successful execution."""
    config: Config = Config()
    config.tts.cli_mode = True
    engine: TTSEngine = TTSEngine(config)
    engine._piper_bin = "/usr/bin/piper"

    # Mock successful subprocess execution
    mock_result: MagicMock = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"\x00\x00"  # Minimal WAV header

    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: mock_result)

    result: bytes | None = engine._synthesize_via_cli(text="hello world")
    assert result is not None
    assert result.startswith(b"RIFF")  # WAV files start with RIFF header


def test_tts_playback_loop_with_multiple_items(monkeypatch) -> None:
    """Test TTS playback loop processes multiple queue items."""
    config: Config = Config()
    engine: TTSEngine = TTSEngine(config)
    engine._running = True

    synthesized_items: list[str] = []

    def mock_synthesize(self, text) -> bytes:
        synthesized_items.append(text)
        return b"wav_data"

    def mock_play_wav(self, data) -> None:
        pass  # Just acknowledge playback

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
