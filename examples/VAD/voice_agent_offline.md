# `voice_agent_offline.py`

## Overview

This example shows a minimal offline voice agent built around three components:

- `WakeWordDetector` to listen for the wake word
- `ASREngine` to transcribe the user after wake-up
- `TTSEngine` to speak the response back

The agent uses a simple state machine:

- Start in wake-word mode
- Switch to ASR mode when the wake word is detected
- Transcribe a single user utterance
- Generate a response
- Speak the response
- Return to wake-word mode

## What Changed in the Script

- The agent class is now `SimpleVoiceAgent`
- Configuration is loaded through `src.utils.config`
- Wake-word and ASR listening are kept mutually exclusive with a shared lock
- The response logic is a small keyword-based handler in `_generate_response()`
- The main loop keeps the process alive until `Ctrl+C`

## Main Extension Point

The current place to customize behavior is:

- `_generate_response(user_input: str) -> str`

That method currently handles a few simple keywords:

- `hello`
- `hi`
- `time`
- `weather`
- `help`
- `thanks`
- `thank you`

If none of those keywords match, the agent falls back to an echo-style response.

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

If you want to change models or runtime behavior, update the project config rather than editing the listener flow directly.

## Customization Ideas

- Replace `_generate_response()` with an LLM-backed response generator
- Add intent routing before speaking the response
- Extend the keyword map with command-style responses
- Swap ASR or TTS implementations through the existing engine interfaces

## Runtime Behavior

- On startup, the agent loads ASR, TTS, and wake-word models
- A wake word triggers a spoken prompt: `Yes? How can I help you?`
- The next transcript is processed only while wake-word mode is active
- `Ctrl+C` cleanly stops the agent and unloads the engines
