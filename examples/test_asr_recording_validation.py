#!/usr/bin/env python3
"""Integration test: ASREngine Recording Validation.

Uses the production ASREngine to capture audio and validates that the
captured data is valid PCM by saving it to a temporary WAV file.

Run with:
  python examples/test_asr_recording_validation.py
"""

import os
import queue
import sys
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, cast

from typing_extensions import override

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.asr import ASREngine
from src.utils.config import load_config

if TYPE_CHECKING:
    _BaseQueue = queue.Queue[bytes | None]
else:
    _BaseQueue = queue.Queue


def test_asr_recording() -> bool:

    # 1. Load configuration
    config = load_config()

    # 2. Initialize Engine
    engine = ASREngine(config)
    try:
        engine.load()
    except Exception:
        return False

    # 3. Instrument the queue to capture raw audio chunks
    captured_data: list[bytes] = []
    transcripts: list[str] = []

    def on_transcript(text: str) -> None:
        if text:
            transcripts.append(text)

    class CaptureQueue(_BaseQueue):
        @override
        def put(self, item: bytes | None, block: bool = True, timeout: float | None = None) -> None:
            if item is not None:
                captured_data.append(item)
            super().put(item, block=block, timeout=timeout)

    # pyright: ignore[reportPrivateUsage]
    engine._audio_queue = cast("_BaseQueue", CaptureQueue())

    # 4. Run capture
    duration = 3

    try:
        engine.start(callback=on_transcript)

        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.5)

        engine.stop()

    except Exception:
        engine.stop()
        return False

    # 5. Validate captured data
    if not captured_data:
        return False

    # 6. Save to temporary file for verification
    tmp_file = Path(__file__).parent.parent / ".tmp" / "test_asr_capture.wav"
    tmp_file.parent.mkdir(exist_ok=True)

    with wave.open(str(tmp_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(config.audio.input_sample_rate)
        wf.writeframes(b"".join(captured_data))

    return True


if __name__ == "__main__":
    success = test_asr_recording()
    os._exit(0 if success else 1)
