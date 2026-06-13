import unittest
from unittest.mock import MagicMock

import pytest
from src.audio.audio_utils import get_audio_backend


@pytest.mark.basic
class TestAudioBackends(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

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
