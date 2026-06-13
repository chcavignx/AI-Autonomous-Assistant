from __future__ import annotations

import pytest
from src.audio.tts import TTSEngine
from src.utils.config import load_config

pytestmark = pytest.mark.basic


def test_speak_enqueues_text_without_loading() -> None:
    config = load_config()
    tts = TTSEngine(config)
    tts.speak("hello world")
    queued = tts._tts_queue.get_nowait()
    assert queued == "hello world"


def test_speak_ignores_empty_text() -> None:
    config = load_config()
    tts = TTSEngine(config)
    tts.speak("   ")
    assert tts._tts_queue.empty()


def test_interrupt_clears_queue() -> None:
    config = load_config()
    tts = TTSEngine(config)
    tts.speak("hello world")
    tts.interrupt()
    assert tts._tts_queue.empty()
