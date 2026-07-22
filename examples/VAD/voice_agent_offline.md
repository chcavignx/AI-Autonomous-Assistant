# `voice_agent_offline.py`

## Overview

This example shows a minimal offline voice agent built around four components:

- `WakeWordDetector` to listen for the wake word
- `ASREngine` to transcribe the user after wake-up
- `TTSEngine` to speak the response back
- **LLM Integration** to dynamically generate responses using an external or local provider like Ollama

The agent uses a simple state machine:

- Start in wake-word mode
- Switch to ASR mode when the wake word is detected
- Transcribe a single user utterance
- Generate a response (using intent matching or an LLM)
- Speak the response
- Return to wake-word mode

## What Changed in the Script

- The agent class is now `SimpleVoiceAgent`
- Configuration is loaded through `src.utils.config`
- Wake-word and ASR listening are kept mutually exclusive with a shared lock
- The response logic uses a rule-based configuration (`data/responses_en.yaml`) and optionally integrates with a local LLM for dynamic answering
- The main loop keeps the process alive until `Ctrl+C`

## Main Extension Point

The current place to customize behavior is:

- `_generate_response(user_input: str) -> tuple[str, bool]`

That method currently parses the configured responses YAML (e.g., `data/responses_en.yaml`) for:

- Simple keywords mapping to specific text
- Action triggers like `get_time` or `control_lights`
- An `llm` rule type that forwards queries to `_generate_llm_response()` if `use_llm` is true.

If none of those rules match or the LLM is unavailable, the agent falls back to an echo-style response.

## LLM Integration (Ollama)

The agent integrates with the `src.llm` library. When `use_llm` is enabled in your configuration:
- User transcripts are sent to the configured LLM API (e.g., a local Ollama instance).
- You can format the payload to include custom context (e.g., injecting the wake word).
- If the LLM times out or is unreachable, the agent falls back to static intent matching.
- **Ensure the Ollama server is running locally** and the desired model is pulled (e.g., `tinyllama`).

## Architecture Notes

- `start()` enables wake-word listening
- `_on_wake_word_detected()` stops wake listening and starts ASR
- `_on_transcript_received()` handles the spoken command and returns to wake-word mode
- `_ensure_wake_mode()` and `_ensure_asr_mode()` prevent both listeners from running at once
- `_stop_listeners()` shuts down active listeners in a safe order

## Usage

Run the example from the repository root:

```bash
python examples/VAD/voice_agent_offline.py
```

## Configuration

The script expects the project configuration to define the voice stack used by:

- wake word detection
- speech recognition
- text-to-speech
- llm API parameters

If you want to change models or runtime behavior, update the project config (`config.yaml`) rather than editing the listener flow directly.

## Customization Ideas

- Replace or extend the intent routing before speaking the response
- Extend the keyword map with command-style responses
- Swap ASR or TTS implementations through the existing engine interfaces
- Build custom prompts for the LLM payload

## Runtime Behavior

- On startup, the agent loads ASR, TTS, wake-word models, and initializes the LLM config
- A wake word triggers a spoken prompt: `Yes? How can I help you?`
- The next transcript is processed only while wake-word mode is active
- `Ctrl+C` cleanly stops the agent and unloads the engines
