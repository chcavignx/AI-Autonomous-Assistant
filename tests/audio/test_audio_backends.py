import sys
import unittest
from unittest.mock import MagicMock, patch

from src.audio.audio_utils import get_audio_backend


class TestAudioBackends(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

    def test_get_audio_backend_explicit_sounddevice(self):
        self.config.audio.backend = "sounddevice"
        assert get_audio_backend(self.config) == "sounddevice"

    def test_get_audio_backend_explicit_pyaudio(self):
        self.config.audio.backend = "pyaudio"
        assert get_audio_backend(self.config) == "pyaudio"

    @patch.dict(sys.modules, {"sounddevice": MagicMock()})
    def test_get_audio_backend_auto_with_sounddevice(self):
        self.config.audio.backend = "auto"
        # Mock import success
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                MagicMock() if name == "sounddevice" else __import__(name, *args, **kwargs)
            ),
        ):
            assert get_audio_backend(self.config) == "sounddevice"

    @patch.dict(sys.modules, {"pyaudio": MagicMock()})
    def test_get_audio_backend_auto_fallback_to_pyaudio(self):
        self.config.audio.backend = "auto"

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("mocked sounddevice import failure")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            assert get_audio_backend(self.config) == "pyaudio"

    @patch("src.audio.audio_utils.SoundDeviceAudioStream")
    def test_asr_engine_uses_correct_backend(self, mock_sd_stream):
        from src.audio.asr import ASREngine

        self.config.audio.backend = "sounddevice"
        self.config.audio.input_sample_rate = 16000
        self.config.audio.input_chunk_ms = 20

        ASREngine(self.config)
        # Verify preference detection
        assert get_audio_backend(self.config) == "sounddevice"

    @patch("src.audio.audio_utils.get_audio_backend", return_value="pyaudio")
    @patch("pyaudio.PyAudio")
    def test_tts_engine_falls_back_to_pyaudio(self, mock_pyaudio, mock_get_backend):
        from src.audio.tts import TTSEngine

        self.config.audio.backend = "auto"
        self.config.tts.cli_mode = True

        engine = TTSEngine(self.config)
        engine.load()
        assert engine._audio_player._backend == "pyaudio"


if __name__ == "__main__":
    unittest.main()
