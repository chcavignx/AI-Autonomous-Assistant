import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from src.audio.audio_utils import get_audio_backend


@pytest.mark.basic
class TestAudioBackends(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.mock_sd = ModuleType("sounddevice")
        self.mock_sd.InputStream = MagicMock()
        self.mock_sd.OutputStream = MagicMock()
        self.mock_sd.play = MagicMock()
        self.mock_sd.wait = MagicMock()
        self.mock_sd.stop = MagicMock()
        self.mock_sd.query_devices = MagicMock(return_value=[{"name": "mock_device"}])
        self.mock_sd.default = MagicMock()
        self.mock_sd.PortAudioError = Exception
        self._patcher = patch.dict(sys.modules, {"sounddevice": self.mock_sd})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_get_audio_backend_is_exclusive(self) -> None:
        # Verify that sounddevice is the only returned and available backend
        assert get_audio_backend() == "sounddevice"

    def test_engines_default_to_sounddevice(self) -> None:
        from src.audio.asr import ASREngine
        from src.audio.tts import TTSEngine

        self.config.audio.input_sample_rate = 16000
        self.config.audio.input_chunk_ms = 20
        self.config.tts.cli_mode = True

        ASREngine(self.config)
        tts = TTSEngine(self.config)

        assert get_audio_backend() == "sounddevice"
        assert tts._audio_player.get_backend() == "sounddevice"


if __name__ == "__main__":
    unittest.main()
