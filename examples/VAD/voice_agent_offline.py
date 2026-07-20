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

import yaml
from pydantic import BaseModel, Field

from src.audio.asr import ASREngine
from src.audio.tts import TTSEngine
from src.audio.wake_word import WakeWordDetector
from src.llm import generate_llm_response
from src.utils import config as _config_module

def get_default_language() -> str:
    """Gets the default language for responses.

    Falls back to `config.asr.language` if defined, otherwise "en".
    """
    try:
        return config.asr.language
    except (NameError, AttributeError):
        return "en"

class ResponsesConfig(BaseModel):
    """Configuration for voice agent responses."""

    language: str = Field(default_factory=get_default_language)
    use_llm: bool = False
    files: dict[str, str] = {
        "en": "data/responses_en.yaml",
        "fr": "data/responses_fr.yaml",
    }
    llm_payload: dict[str, object] = {
        "model": "{model}",
        "prompt": "You are a helpful, concise voice assistant. The user said: '{user_input}'. Respond shortly in one or two sentences.",
        "stream": False,
    }

    @property
    def full_responses_path(self) -> Path:
        """Returns the full path to the responses file."""
        file_relative_path = self.files.get(self.language, f"data/responses_{self.language}.yaml")
        return (project_root / file_relative_path).resolve()


def load_responses_config(config_path: Path | None = None) -> ResponsesConfig:
    """Load the responses configuration from a YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "voice_agent_offline.yaml"

    if not config_path.exists():
        return ResponsesConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if config_dict and isinstance(config_dict, dict) and "responses" in config_dict:
        return ResponsesConfig.model_validate(config_dict["responses"])

    return ResponsesConfig()

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

    def __init__(self, config_path: Path | None = None, responses_config: ResponsesConfig | None = None) -> None:
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

        if responses_config is not None:
            self.responses_config = responses_config
        else:
            self.responses_config = load_responses_config()

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

        # Load voice response keyword table
        self._response_rules = []
        self._load_responses_table()

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
        response, should_continue = self._generate_response(transcript)
        logger.info("🤖 Response: '%s'", response)

        # Stop listening during TTS playback to avoid hearing ourselves
        with self._listener_lock:
            if self._asr_active:
                self.asr.stop()
                self._asr_active = False

        self.tts.speak(response, blocking=True)

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

    def _load_responses_table(self) -> None:
        """Load the responses table from YAML based on config."""
        try:
            responses_path = self.responses_config.full_responses_path
            logger.info("Loading responses table from %s", responses_path)
            if responses_path.exists():
                with open(responses_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and "responses" in data:
                        self._response_rules = data["responses"]
                        logger.info("✓ Loaded %d responses", len(self._response_rules))
                    else:
                        logger.warning("No 'responses' list found in %s", responses_path)
            else:
                logger.error("Responses configuration file not found at %s", responses_path)
        except Exception as e:
            logger.exception("Failed to load responses table: %s", e)

    def _generate_response(self, user_input: str) -> tuple[str, bool]:
        """Generate a response based on user input.

        Scans the loaded keyword rules for matching patterns.
        """
        user_input_lower = user_input.lower().strip()
        matched_rule = None

        # Find first matching rule
        for rule in self._response_rules:
            keywords = rule.get("keywords", [])
            # Empty keywords is considered fallback/default rule, skip in normal scan
            if not keywords:
                continue
            if any(keyword.lower() in user_input_lower for keyword in keywords):
                matched_rule = rule
                break

        # Fallback if no matching rule found
        if not matched_rule:
            for rule in self._response_rules:
                keywords = rule.get("keywords", [])
                if not keywords:
                    matched_rule = rule
                    break

        if not matched_rule:
            return f"You said: {user_input}. I'm still learning how to respond to that.", True

        # Resolve response based on type
        rule_type = matched_rule.get("type", "text")
        val = matched_rule.get("value", "")
        should_continue = not matched_rule.get("exit", False)

        if rule_type == "text" or (rule_type == "llm" and not self.responses_config.use_llm):
            if len([p for p in self.config.wake.wake_word.split("_") if p]) > 1:
                wake_word_last = self.config.wake.wake_word.split("_")[-1]
                response = val.format(wake_word=wake_word_last, user_input=user_input)
            else:
                response = val.format(wake_word=self.config.wake.wake_word, user_input=user_input)
        elif rule_type == "action":
            response = self._execute_action(val)
        elif rule_type == "llm":
            response = self._generate_llm_response(user_input, fallback_template=val)
        else:
            response = str(val)

        return response, should_continue

    def _execute_action(self, action_name: str) -> str:
        """Map and execute named action."""
        action_handlers = {
            "get_time": self._action_get_time,
            "get_date": self._action_get_date,
            "control_lights": self._action_control_lights,
            "play_music": self._action_play_music,
        }
        handler = action_handlers.get(action_name)
        if handler:
            try:
                return handler()
            except Exception as e:
                logger.exception("Error executing action %s: %s", action_name, e)
                return f"Sorry, there was an error executing action {action_name}."
        logger.warning("No handler found for action: %s", action_name)
        return f"I recognized the action {action_name}, but I don't know how to execute it yet."

    def _action_get_time(self) -> str:
        return "The current time is " + datetime.datetime.now().strftime("%I:%M %p")

    def _action_get_date(self) -> str:
        return "Today, the date is: " + datetime.datetime.now().strftime("%d %B %Y")

    def _action_control_lights(self) -> str:
        return "I would control your lights if I had smart home integration."

    def _action_play_music(self) -> str:
        return "I would play music if I had access to your media system."

    def _generate_llm_response(self, user_input: str, fallback_template: str = "") -> str:
        """Call LLM API to generate response, with fallback."""
        api_type = self.config.llm.api_type
        url = self.config.llm.url
        model = self.config.llm.model
        timeout = self.config.llm.timeout
        api_key = self.config.llm.api_key

        prompt_tmpl = self.responses_config.llm_payload.get("prompt", "The user said: '{user_input}'. Respond shortly in one or two sentences.")
        prompt = prompt_tmpl.format(model=model, user_input=user_input) if isinstance(prompt_tmpl, str) else str(prompt_tmpl)

        payload_override = {}
        for k, v in self.responses_config.llm_payload.items():
            if isinstance(v, str):
                payload_override[k] = v.format(model=model, user_input=user_input)
            else:
                payload_override[k] = v

        override = payload_override if api_type == "ollama" else None

        res = generate_llm_response(
            api_type=api_type,
            url=url,
            model=model,
            prompt=prompt,
            timeout=timeout,
            api_key=api_key,
            payload_override=override,
        )
        if res is not None:
            return res

        return fallback_template.format(wake_word=self.config.wake.wake_word, user_input=user_input)

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
