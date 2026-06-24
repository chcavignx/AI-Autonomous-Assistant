#!/usr/bin/env python3
"""simple_voice_agent.py.
======================
Minimal voice agent demonstrating ASR, TTS, and wake-word detection.

Features:
  - Wake word detection (Direct ONNX Runtime, no openwakeword library package)
  - Speech recognition (Whisper or Faster-Whisper)
  - Text-to-speech synthesis (Piper)
  - Simple intent-based response system
  - Non-blocking queue-based threading architecture

Usage:
  python examples/simple_voice_agent.py
"""

import datetime
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import final

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.audio.wake_word import WakeWordDetector
from src.utils import config as _config_module

app_name = 'voice_agent_offline'
logger = logging.getLogger(app_name)


_config_module.setup_python_path()
config = _config_module.load_config()


@final
class SimpleVoiceAgent:
    """Minimal voice agent orchestrating ASR, TTS, and wake word detection.

    Flow:
      1. Listen for wake word
      2. On wake, transcribe user speech
      3. Generate response
      4. Speak response
      5. Return to listening for wake word
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the agent with centralized config."""
        super().__init__()
        try:
            if config_path is not None:
                self.config = _config_module.load_config(config_path)
            else:
                self.config = _config_module.load_config()
        except Exception as e:
            logger.exception("Failed to load config: %s", e)
            raise

        # Initialize components
        self.asr = ASREngine(self.config)
        self.tts = TTSEngine(self.config)
        self.wake_detector = WakeWordDetector(self.config)

        # State
        self.is_running = False
        self.wake_word_active = False
        self._listener_lock = threading.Lock()
        self._asr_active = False
        self._wake_active = False

        logger.info("✓ Voice agent initialized")

    def load_models(self) -> None:
        """Load ASR, TTS, and wake word models."""
        logger.info("Loading models...")
        try:
            self.asr.load()
            self.tts.load()
            self.wake_detector.load()
            logger.info("✓ All models loaded")
        except Exception as e:
            logger.exception("Failed to load models: %s", e)
            raise

    def start(self) -> None:
        """Start listening for wake word."""
        self.is_running = True
        logger.info(f"🎤 Listening for wake word: '{self.config.wake.wake_word}'")

        self._ensure_wake_mode()

    def stop(self) -> None:
        """Stop all engines and cleanup."""
        logger.info("Stopping voice agent...")
        self.is_running = False

        self._stop_listeners()
        self.tts.unload()
        self.asr.unload()

        logger.info("✓ Voice agent stopped")

    def _on_wake_word_detected(self) -> None:
        """Callback when wake word is detected."""
        logger.info("🟣 Wake word detected!")
        self.wake_word_active = True

        # Stop listening during TTS playback to avoid hearing ourselves
        with self._listener_lock:
            if self._wake_active:
                self.wake_detector.stop()
                self._wake_active = False

        self.tts.speak("Yes? How can I help you?", blocking=True)
        self._ensure_asr_mode()

    def _on_transcript_received(self, transcript: str) -> None:
        """Callback when speech is transcribed."""
        transcript = transcript.strip()
        logger.info("🟡 Transcript received: '%s'", transcript)
        if not transcript:
            return

        logger.info("📝 You said: '%s'", transcript)

        if not self.wake_word_active:
            # Still listening for wake word
            return

        # Process the command
        response = self._generate_response(transcript)
        logger.info("🤖 Response: '%s'", response)

        # Stop listening during TTS playback to avoid hearing ourselves
        with self._listener_lock:
            if self._asr_active:
                self.asr.stop()
                self._asr_active = False

        self.tts.speak(response, blocking=True)

        # Check if we should continue listening in ASR mode or return to wake word mode
        should_continue = not any(exit_word in transcript.lower() for exit_word in ["stop", "exit", "quit"])

        if should_continue:
            logger.info("🟢 Continuing conversation, staying in ASR mode")
            self._ensure_asr_mode()
        else:
            self.wake_word_active = False
            logger.info("🟢 Returning to wake word mode")
            self._ensure_wake_mode()

    def _ensure_wake_mode(self) -> None:
        """Run wake-word listening without a parallel ASR capture stream."""
        if not self.is_running:
            return
        logger.info("\n🟢 Starting wake word listener, stopping ASR...")
        with self._listener_lock:
            if self._asr_active:
                self.asr.stop()
                self._asr_active = False
            if not self._wake_active:
                self.wake_detector.start(callback=self._on_wake_word_detected)
                self._wake_active = True
                logger.info(f"🎤 Listening for wake word (wake mode): '{self.config.wake.wake_word}'")

    def _ensure_asr_mode(self) -> None:
        """Run ASR listening without a parallel wake-word capture stream."""
        if not self.is_running:
            return
        logger.info("\n⏹️ Stopping wake word listener, starting ASR...")
        with self._listener_lock:
            if self._wake_active:
                self.wake_detector.stop()
                self._wake_active = False
            if not self._asr_active:
                self.asr.start(callback=self._on_transcript_received)
                self._asr_active = True
                logger.info("🎤 Listening for speech (ASR mode)")

    def _stop_listeners(self) -> None:
        """Stop audio listeners in a state-aware order."""
        with self._listener_lock:
            if self._wake_active:
                self.wake_detector.stop()
                self._wake_active = False
            if self._asr_active:
                self.asr.stop()
                self._asr_active = False

    def _generate_response(self, user_input: str) -> str:
        """Generate a simple response based on user input.

        Replace this with your AI model (OpenAI, local LLM, etc.)
        """
        user_input_lower = user_input.lower()

        # Simple keyword matching
        responses = {
            "hello": f"Hello! I'm {self.config.wake.wake_word}, your AI assistant. How can I help you?",
            "hi": f"Hi there! What can I do for you?",
            'time': 'The current time is ' + datetime.datetime.now().strftime("%I:%M %p"),  # Get the current time and format it
            'date': 'Today, the date is: ' + datetime.datetime.now().strftime("%d %B %Y"),  # Get the current date and format it
            "lights": f"I would control your lights if I had smart home integration.",
            "music": f"I would play music if I had access to your media system.",
            "stop": f"Goodbye! Returning to wake word detection.",
            "bye_bye": f"See you later! Going back to sleep mode.",
            "help": f"I can respond to simple commands like hello, hi, time, date, lights, music, stop and bye-bye."
        }

        # Match keywords
        for keyword, response in responses.items():
            if keyword in user_input_lower:
                logger.info("🤖 Response: '%s'", response)
                return response

        # Default response
        # Replace this with your AI model, e.g., using sentence_similarity with intent
        # Example:
        # intent, score, context = sentence_similarity(user_input, self.intents)
        # if score > 0.7:
        #     response = self._generate_response(intent, context)
        # else:
        #     response = "I'm not sure how to respond to that. Try again.

        response = f"You said: {user_input}. I'm still learning how to respond to that."
        logger.info("🤖 Response: '%s'", response)
        return response

    def run(self) -> None:
        """Main event loop (simplified since threading handles listening)."""

        def signal_handler(signalnum: int, frame: FrameType | None) -> None:
            del signalnum, frame
            logger.info("\n⏹️ Interrupted")
            self.stop()
            sys.exit(0)

        _ = signal.signal(signal.SIGINT, signal_handler)

        try:
            self.load_models()
            self.start()
            logger.info("Press Ctrl+C to exit\n")

            # Keep the main thread alive

            while self.is_running:
                time.sleep(1)

        except Exception as e:
            logger.exception("Error: %s", e)
            raise

        finally:
            self.stop()


def main() -> None:
    """Entry point."""
    agent = SimpleVoiceAgent()
    agent.run()


if __name__ == "__main__":
    main()
