"""Unit tests for src/utils/config.py."""

import pathlib
import sys

import pytest
import yaml

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils import config


def test_path_config_defaults() -> None:
    pc = config.PathConfig()
    assert pc.src == "src"
    assert pc.data == "data"
    assert pc.cache == ".cache"
    assert pc.models == "models"


def test_path_config_properties() -> None:
    pc = config.PathConfig()
    assert pc.src_path == config.ROOT_DIR / "src"
    assert pc.data_path == config.ROOT_DIR / "data"
    assert pc.cache_path == config.ROOT_DIR / ".cache"
    assert pc.models_path == config.ROOT_DIR / ".cache" / "models"
    assert pc.models_audio_path == config.ROOT_DIR / ".cache" / "audio" / "models"


def test_asr_config_defaults() -> None:
    asr = config.ASRConfig()
    assert asr.engine == "faster-whisper"
    assert asr.model_size == "tiny"
    assert asr.faster_model_size == "small"
    assert asr.language == "en"
    assert asr.translate is False
    assert asr.transformers is False
    assert asr.transformers_engine == "huggingface"
    assert asr.download_root is None
    assert asr.device == "cpu"
    assert asr.compute_type == "int8"
    assert asr.skip_native_teardown is False


def test_asr_config_download_path_default() -> None:
    asr = config.ASRConfig()
    expected = config.ROOT_DIR / ".cache" / "audio" / "models" / "faster-whisper"
    assert asr.download_path == expected


def test_asr_config_download_path_with_root() -> None:
    asr = config.ASRConfig(download_root="custom")
    expected = config.ROOT_DIR / "custom" / "faster-whisper"
    assert asr.download_path == expected


def test_asr_config_full_model_path() -> None:
    asr = config.ASRConfig()
    expected = asr.download_path / "en-tiny.onnx"
    assert asr.full_model_path == expected


def test_tts_config_defaults() -> None:
    import math

    tts = config.TTSConfig()
    assert tts.engine == "piper"
    assert tts.model_name == "jarvis-medium.onnx"
    assert tts.model_path is None
    assert tts.cli_mode is False
    assert tts.device == "cpu"
    assert math.isclose(tts.length_scale, 1.0)
    assert math.isclose(tts.noise_scale, 1.0)
    assert math.isclose(tts.noise_w_scale, 1.0)
    assert tts.normalize_audio is True
    assert math.isclose(tts.speed, 1.0)
    assert math.isclose(tts.volume, 0.5)


def test_tts_config_full_model_path_default() -> None:
    tts = config.TTSConfig()
    expected = config.ROOT_DIR / ".cache" / "audio" / "models" / "piper" / "jarvis-medium.onnx"
    assert tts.full_model_path == expected


def test_tts_config_full_model_path_with_path() -> None:
    tts = config.TTSConfig(model_path="custom/model.onnx")
    expected = config.ROOT_DIR / "custom" / "model.onnx"
    assert tts.full_model_path == expected


def test_wake_config_defaults() -> None:
    import math

    wake = config.WakeConfig()
    assert wake.wake_word == "hey_jarvis"
    assert wake.model_name == "hey_jarvis"
    assert wake.model_path is None
    assert wake.inference_framework == "onnx"
    assert math.isclose(wake.threshold, 0.4)
    assert math.isclose(wake.cooldown_seconds, 2.0)
    assert wake.download_root is None
    assert wake.noise_suppression is False
    assert math.isclose(wake.vad_threshold, 0.6)


def test_wake_config_download_path_default() -> None:
    wake = config.WakeConfig()
    expected = config.ROOT_DIR / ".cache" / "audio" / "models" / "wakeword"
    assert wake.download_path == expected


def test_wake_config_full_model_path() -> None:
    wake = config.WakeConfig(model_name="model")
    expected = wake.download_path / "model.onnx"
    assert wake.full_model_path == expected


def test_vad_config_defaults() -> None:
    import math

    vad = config.VADConfig()
    assert vad.min_speech_duration_ms == 100
    assert vad.min_silence_duration_ms == 500
    assert vad.silence_timeout_seconds == 1
    assert vad.max_recording_seconds == 15
    assert math.isclose(vad.threshold, 0.6)


def test_audio_config_defaults() -> None:
    import math

    audio = config.AudioConfig()
    assert audio.input_sample_rate == 22050
    assert audio.input_chunk_ms == 30
    assert audio.input_chunk_size == 500
    assert audio.input_device_index is None
    assert math.isclose(audio.volume, 0.5)
    assert audio.output_device_index is None
    assert audio.output_sample_rate == 22050
    assert audio.output_chunk_ms == 30
    assert audio.output_chunk_size == 500
    assert audio.output_device_name is None


def test_platform_config_defaults() -> None:
    platform = config.PlatformConfig()
    assert platform.cpu_cores == 2
    assert platform.pi is False


def test_platform_config_is_raspberry_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = config.PlatformConfig()
    monkeypatch.setattr(config, "detect_raspberry_pi_model", lambda: True)
    assert platform.is_raspberry_pi() is True
    assert platform.pi is True


def test_platform_config_cpu_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = config.PlatformConfig()
    monkeypatch.setattr(config, "limit_cpu_for_multiprocessing", lambda x: 4)
    assert platform.cpu_limit() == 4
    assert platform.cpu_cores == 4


def test_config_init() -> None:
    cfg = config.Config()
    assert isinstance(cfg.paths, config.PathConfig)
    assert isinstance(cfg.asr, config.ASRConfig)
    assert isinstance(cfg.tts, config.TTSConfig)
    assert isinstance(cfg.wake, config.WakeConfig)
    assert isinstance(cfg.vad, config.VADConfig)
    assert isinstance(cfg.audio, config.AudioConfig)
    assert isinstance(cfg.platform, config.PlatformConfig)


def test_config_properties() -> None:
    cfg = config.Config()
    assert cfg.cpu_cores == cfg.platform.cpu_cores
    assert cfg.sample_rate == cfg.vad.sample_rate
    assert cfg.min_speech_duration_ms == cfg.vad.min_speech_duration_ms
    assert cfg.min_silence_duration_ms == cfg.vad.min_silence_duration_ms
    assert cfg.stt_language == cfg.asr.language
    assert cfg.stt_model_size == cfg.asr.model_size
    assert cfg.faster_stt_model_size == cfg.asr.faster_model_size


def test_config_property_setters() -> None:
    cfg = config.Config()
    cfg.stt_model_size = "large"
    assert cfg.asr.model_size == "large"
    cfg.faster_stt_model_size = "medium"
    assert cfg.asr.faster_model_size == "medium"


def test_asr_config_download_path_with_engine_in_path() -> None:
    asr = config.ASRConfig(download_root="faster-whisper")
    expected = config.ROOT_DIR / "faster-whisper"
    assert asr.download_path == expected


def test_tts_config_full_model_path_file() -> None:
    tts = config.TTSConfig(model_path="custom/model.onnx")
    expected = config.ROOT_DIR / "custom" / "model.onnx"
    assert tts.full_model_path == expected


def test_wake_config_download_path_with_suffix() -> None:
    wake = config.WakeConfig(download_root="model.onnx")
    expected = config.ROOT_DIR / "model.onnx"
    assert wake.download_path == expected


def test_load_config_default() -> None:
    cfg = config.load_config()
    assert isinstance(cfg, config.Config)


def test_load_config_with_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock yaml.safe_load to return config dict
    mock_config_dict = {"asr": {"model_size": "large"}}
    monkeypatch.setattr(yaml, "safe_load", lambda f: mock_config_dict)
    # Mock exists to True
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    # Mock open
    from io import StringIO

    mock_file = StringIO()
    monkeypatch.setattr(pathlib.Path, "open", lambda self, mode, encoding: mock_file)
    cfg = config.load_config()
    assert cfg.asr.model_size == "large"


def test_load_config_nonexistent_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
    cfg = config.load_config()
    assert isinstance(cfg, config.Config)


def test_load_config_invalid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(self):
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)
    cfg = config.load_config(pathlib.Path("invalid"))
    assert isinstance(cfg, config.Config)


def test_load_config_yaml_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yaml, "safe_load", lambda f: None)
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    from io import StringIO

    mock_file = StringIO()
    monkeypatch.setattr(pathlib.Path, "open", lambda self, mode, encoding: mock_file)
    cfg = config.load_config()
    assert isinstance(cfg, config.Config)
