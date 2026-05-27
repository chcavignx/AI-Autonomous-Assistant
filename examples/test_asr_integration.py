#!/usr/bin/env python3
"""Integration test: ASR Engine Pipeline.

Validates the high-level ASREngine by ensuring it can initialize hardware,
capture audio chunks through its internal threads, and pass them to the pipeline.

Run with:
  python examples/test_asr_integration.py
"""

import os
import queue
import sys
import time
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


def test_asr_integration() -> bool:

    config = load_config()

    # Track data flow metrics
    chunks_received: list[int] = []
    transcripts: list[str] = []

    def on_transcript(text: str) -> None:
        if text:
            transcripts.append(text)

    # We use a custom Queue to verify the recording thread is feeding the engine
    class CountingQueue(_BaseQueue):
        @override
        def put(self, item: bytes | None, block: bool = True, timeout: float | None = None) -> None:
            if item is not None:
                chunks_received.append(len(item))
            super().put(item, block=block, timeout=timeout)

    try:
        engine = ASREngine(config)
        engine.load()
        # Inject our counting queue for verification
        # pyright: ignore[reportPrivateUsage]
        engine._audio_queue = cast("_BaseQueue", CountingQueue())
    except Exception:
        return False

    try:
        engine.start(callback=on_transcript)

        for _ in range(5):
            time.sleep(1)

        engine.stop()

        engine.unload()

        # Validation

        return len(chunks_received) != 0

    except Exception:
        return False


if __name__ == "__main__":
    success = test_asr_integration()
    # Explicit exit to prevent potential ALSA/PortAudio teardown segfaults
    os._exit(0 if success else 1)
