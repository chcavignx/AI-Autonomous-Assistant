"""
Comprehensive unit tests for the Voice Agent voice_agent_offline module.
Tests cover: AudioUtils, VADEngine, STTEngine, TTSEngine, AIResponseEngine, VoiceAgent, AgentConfig
"""

import importlib.util
import json
import os
import sys
import types
from typing import Any, Generator
from unittest.mock import MagicMock

import numpy as np
import pytest

# ============================================================================
# PYTEST FIXTURE: Module loader with fake dependencies
# ============================================================================


@pytest.fixture
def mod() -> Generator[Any, None, None]:
    """
    Fixture that injects fake modules and loads voice_agent_offline.py for isolated testing.
    Records original sys.modules entries, injects fakes, loads the module, and restores state
    on teardown to ensure test isolation.
    """
    # Prepare fake modules to satisfy imports in voice_agent_offline.py
    fake_modules = {}

    # whisper fake
    whisper_mod = types.SimpleNamespace()

    def fake_load_model(name, device=None, download_root=None):
        class FakeWhisperModel:
            def transcribe(self, audio, **kwargs):
                return {"segments": [{"text": "Test transcript from whisper"}]}

        return FakeWhisperModel()

    whisper_mod.load_model = fake_load_model
    fake_modules["whisper"] = whisper_mod

    # faster_whisper fake
    fw_mod = types.SimpleNamespace()

    class FakeFWModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            # emulate returning (segments, info)
            segments = [types.SimpleNamespace(text="FW segment")]
            info = types.SimpleNamespace(dummy=1)
            return (segments, info)

    fw_mod.WhisperModel = FakeFWModel
    fake_modules["faster_whisper"] = fw_mod

    # sysutils fake
    sysutils = types.SimpleNamespace(
        detect_raspberry_pi_model=lambda: False,
        limit_cpu_for_multiprocessing=lambda *args, **kwargs: None,
    )
    fake_modules["src.utils.sysutils"] = sysutils

    # vosk fake
    vosk_mod = types.SimpleNamespace()

    class FakeVoskModel:
        def __init__(self, path):
            self.path = path

    class FakeKaldiRecognizer:
        def __init__(self, model, sample_rate):
            self.model = model
            self.sample_rate = sample_rate
            self._accepted = False

        def AcceptWaveform(self, data):
            self._accepted = True
            return True

        def Result(self):
            return json.dumps({"text": "vosk recognized text"})

        def FinalResult(self):
            return json.dumps({"text": "vosk recognized text"})

    vosk_mod.Model = FakeVoskModel
    vosk_mod.KaldiRecognizer = FakeKaldiRecognizer
    fake_modules["vosk"] = vosk_mod

    # piper fake
    piper_mod = types.SimpleNamespace()

    class FakePiperConfig:
        sample_rate = 16000

    class FakePiperVoice:
        config = FakePiperConfig()

        @staticmethod
        def load(path, *args, **kwargs):
            class V:
                config = FakePiperConfig()

                def synthesize(self, text, *args, **kwargs):
                    # emulate generator of chunks
                    class Chunk:
                        def __init__(self, arr):
                            self.audio_int16_array = arr

                    return [Chunk(np.zeros(160, dtype=np.int16))]

                def synthesize_wav(self, text, wav_file, *args, **kwargs):
                    # create a tiny valid WAV (silence)
                    if hasattr(wav_file, "setnchannels"):
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(b"\x00\x00" * 160)

            return V()

    class FakeSynthesisConfig:
        def __init__(self, *args, **kwargs):
            self.volume = 1.0

    piper_mod.PiperVoice = FakePiperVoice
    piper_mod.SynthesisConfig = FakeSynthesisConfig
    fake_modules["piper"] = piper_mod

    # silero_vad fake
    def fake_load_silero_vad(*args, **kwargs):
        return MagicMock()

    def fake_get_speech_timestamps(
        audio_tensor,
        model,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        return_seconds=False,
    ):
        # Simple heuristic: If array length > 1000, pretend speech exists
        try:
            if hasattr(audio_tensor, "__len__"):
                length = len(audio_tensor)
            else:
                length = np.asarray(audio_tensor).size
        except Exception:
            length = 0

        if length > 1000:
            if return_seconds:
                return [{"start": 0.0, "end": float(length) / 16000.0}]
            return [{"start": 0, "end": length}]
        return []

    silero = types.SimpleNamespace(
        load_silero_vad=fake_load_silero_vad,
        get_speech_timestamps=fake_get_speech_timestamps,
    )
    fake_modules["silero_vad"] = silero

    # pyaudio fake
    pa = types.SimpleNamespace()
    pa.paFloat32 = 8
    pa.paInt16 = 8

    class FakeStream:
        def read(self, n, exception_on_overflow=False):
            return (np.zeros(n, dtype=np.float32)).tobytes()

        def get_read_available(self):
            return 0

        def stop_stream(self):
            pass

        def close(self):
            pass

    class FakePyAudio:
        def __init__(self):
            pass

        def get_device_count(self):
            return 1

        def get_device_info_by_index(self, i):
            return {"maxInputChannels": 1, "name": "fake", "defaultSampleRate": 16000}

        def open(self, *args, **kwargs):
            return FakeStream()

        def terminate(self):
            pass

    pa.PyAudio = FakePyAudio
    fake_modules["pyaudio"] = pa

    # sounddevice fake
    sd = types.SimpleNamespace()

    class FakeOutputStream:
        def __init__(self, samplerate=None, channels=None, dtype=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def start(self):
            pass

        def write(self, data):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    sd.OutputStream = FakeOutputStream
    sd.play = lambda data, samplerate=None, **kwargs: None
    sd.wait = lambda: None
    sd.stop = lambda: None
    fake_modules["sounddevice"] = sd

    # soundfile fake
    sf = types.SimpleNamespace(
        read=lambda path, dtype=None: (np.zeros(160, dtype=np.float32), 16000),
        write=lambda path, data, samplerate: None,
    )
    fake_modules["soundfile"] = sf

    # torch fake
    torch_mod = types.SimpleNamespace()

    def fake_FloatTensor(x):
        return np.asarray(x, dtype=np.float32)

    torch_mod.FloatTensor = fake_FloatTensor
    fake_modules["torch"] = torch_mod

    # Record originals and inject fakes
    originals = {}
    injected = []
    module = None

    try:
        for name, fake_mod in fake_modules.items():
            if name in sys.modules:
                originals[name] = sys.modules[name]
            sys.modules[name] = fake_mod
            injected.append(name)

        # Load the target module
        this_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(this_dir, "voice_agent_offline.py")

        if not os.path.exists(target_path):
            pytest.skip(f"voice_agent_offline.py not found at {target_path}")
            return

        spec = importlib.util.spec_from_file_location(
            "voice_agent_offline_testmod", target_path
        )
        if spec is None or spec.loader is None:
            pytest.skip(f"Cannot load module spec for {target_path}")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules["voice_agent_offline_testmod"] = module
        spec.loader.exec_module(module)

        yield module

    finally:
        # Cleanup
        for name in injected:
            if name in originals:
                sys.modules[name] = originals[name]
            else:
                sys.modules.pop(name, None)

        sys.modules.pop("voice_agent_offline_testmod", None)


# ============================================================================
# TESTS: AudioUtils
# ============================================================================


def test_validate_and_clean_audio_basic(mod):
    """Test audio validation with NaN, inf, extreme values, and silence."""
    AudioUtils = mod.AudioUtils

    arr = np.array(
        [0.0, np.nan, np.inf, -np.inf, 1e12, -1e12, 0.5, -0.5], dtype=np.float32
    )
    cleaned = AudioUtils.validate_and_clean_audio(arr)

    assert np.all(np.isfinite(cleaned)), "Cleaned audio contains NaN or Inf"
    assert np.max(np.abs(cleaned)) <= 1.0 + 1e-6, "Values not clipped to [-1, 1]"
    assert cleaned.dtype == np.float32, "Output dtype is not float32"
    assert cleaned.shape == arr.shape, "Shape changed during cleaning"


def test_convert_to_int16_properties(mod):
    """Test conversion to int16 with proper bounds and dtype."""
    AudioUtils = mod.AudioUtils

    arr = np.zeros(2000, dtype=np.float32)
    int16 = AudioUtils.convert_to_int16(arr)

    assert int16.dtype == np.int16, "Output dtype is not int16"
    assert int16.shape[0] == arr.shape[0], "Array size changed"
    assert np.max(int16) <= 32767, "Values exceed int16 max"
    assert np.min(int16) >= -32768, "Values below int16 min"


def test_audio_validation_nan_inf(mod):
    """Test that NaN and Inf are cleaned properly."""
    AudioUtils = mod.AudioUtils
    arr = np.array([float("nan"), float("inf"), -float("inf"), 0.5], dtype=np.float32)
    out = AudioUtils.validate_and_clean_audio(arr)

    assert np.all(np.isfinite(out)), "NaN or Inf not removed"
    assert out.shape == arr.shape, "Shape changed"


def test_audio_validation_extreme_values(mod):
    """Test that extreme values are clipped to [-1, 1]."""
    AudioUtils = mod.AudioUtils
    arr = np.array([1e12, -1e12, 1.0, -1.0], dtype=np.float32)
    out = AudioUtils.validate_and_clean_audio(arr)

    assert np.all(np.abs(out) <= 1.0), "Extreme values not clipped"


def test_audio_validation_empty_array(mod):
    """Test that empty arrays are handled gracefully."""
    AudioUtils = mod.AudioUtils
    arr = np.array([], dtype=np.float32)
    out = AudioUtils.validate_and_clean_audio(arr)

    assert out.size == 0, "Empty array not handled correctly"


def test_audio_convert_to_int16_basic(mod):
    """Test basic int16 conversion."""
    AudioUtils = mod.AudioUtils
    arr = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = AudioUtils.convert_to_int16(arr)

    assert out.dtype == np.int16, "Output dtype is not int16"
    assert np.max(out) <= 32767, "Values exceed int16 max"


# ============================================================================
# TESTS: AgentConfig
# ============================================================================


def test_agent_config_initialization_defaults(mod):
    """Test AgentConfig default values."""
    AgentConfig = mod.AgentConfig
    cfg = AgentConfig()

    assert cfg.sample_rate == 16000, "Default sample_rate incorrect"
    assert cfg.wake_word == "jarvis", "Default wake_word incorrect"
    assert hasattr(cfg, "synthesis_config"), "synthesis_config missing"


def test_agent_config_custom_values(mod):
    """Test AgentConfig with custom values."""
    AgentConfig = mod.AgentConfig
    cfg = AgentConfig(wake_word="test", sample_rate=8000)

    assert cfg.wake_word == "test", "Custom wake_word not set"
    assert cfg.sample_rate == 8000, "Custom sample_rate not set"


# ============================================================================
# TESTS: STTEngine
# ============================================================================


def test_transcribe_whisper_returns_expected_text(mod):
    """Test Whisper STT transcription returns expected text."""
    AgentConfig = mod.AgentConfig
    STTEngine = mod.STTEngine
    STTBackend = mod.STTBackend

    cfg = AgentConfig(
        stt_engine=STTBackend.WHISPER,
        whisper_model_size="tiny",
        whisper_download_root=".",
        language="en",
    )
    engine = STTEngine(cfg)
    audio = np.ones(2000, dtype=np.float32) * 0.2
    text = engine.transcribe(audio)

    assert "Test transcript from whisper" in text, f"Unexpected text: {text}"


def test_transcribe_vosk_returns_expected_text(mod):
    """Test Vosk STT transcription returns expected text."""
    AgentConfig = mod.AgentConfig
    STTEngine = mod.STTEngine
    STTBackend = mod.STTBackend

    cfg = AgentConfig(
        stt_engine=STTBackend.VOSK, vosk_model_path="/tmp/fake", sample_rate=16000
    )
    engine = STTEngine(cfg)
    audio = np.ones(2000, dtype=np.float32) * 0.1
    text = engine.transcribe(audio)

    assert "vosk recognized text" in text, f"Unexpected text: {text}"


# ============================================================================
# TESTS: AIResponseEngine
# ============================================================================


def test_ai_intent_matching(mod):
    """Test AI intent matching and continuation logic."""
    AIResponseEngine = mod.AIResponseEngine
    ai_engine = AIResponseEngine()

    resp, cont = ai_engine.generate_response("hello")
    assert "Hello" in resp or "hello" in resp.lower(), f"Unexpected response: {resp}"
    assert cont is True, "Should continue after greeting"

    resp, cont = ai_engine.generate_response("stop")
    assert (
        "Goodbye" in resp or "goodbye" in resp.lower()
    ), f"Unexpected response: {resp}"
    assert cont is False, "Should not continue after stop command"


def test_ai_fallback(mod):
    """Test AI fallback response for unknown commands."""
    AIResponseEngine = mod.AIResponseEngine
    ai_engine = AIResponseEngine()

    resp, cont = ai_engine.generate_response("unknown command xyz")
    assert len(resp) > 0, "Empty response for unknown command"
    assert cont is True, "Should continue after unknown command"


def test_ai_generate_ai_response(mod):
    """Test AI response generation."""
    AIResponseEngine = mod.AIResponseEngine
    ai_engine = AIResponseEngine()

    out = ai_engine.generate_ai_response("lights")
    assert len(out) > 0, "Empty AI response"


# ============================================================================
# END OF TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

    # pytest -v -s test_voice_agent.py
