# Offline Text-to-Speech (TTS)

The goal is to enable the assistant to generate voice from text to the user without an internet connection. As usual, extensive documentation and numerous examples are available; however, again, two methods are best suited for the hardware in use: eSpeak (via `pyttsx3`) and Piper.

## 1. eSpeak (via `pyttsx3`)

### Description of the eSpeak TTS Engine

eSpeak is a lightweight, open-source, and very simple-to-use TTS engine that works offline. It's widely available on most operating systems. Its main characteristic is its compact size, but the voice quality is quite robotic and synthetic, making it less suitable for applications that need a natural sound. The Python library **`pyttsx3`** acts as a wrapper that allows you to use eSpeak (or other engines like Sapi5 on Windows and NSSpeechSynthesizer on macOS) in a simple and portable way.

### Installation of the eSpeak TTS Engine

1. **Install dependencies**

```bash
pip3 install sounddevice
```

1. **Install the eSpeak engine**:

```bash
sudo apt-get install espeak-ng
```

If installation failed on your raspberry pi version OS, you have to follow the guide to build library from github repo.
For more details, visit the eSpeak-ng GitHub repository at: [https://github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng).

```bash
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh
./configure
make
sudo make install
```

1. **Install the `pyttsx3` Python library**:

For more details, visit the pyttsx3 GitHub repository at: [https://github.com/nateshmbhat/pyttsx3.git](https://github.com/nateshmbhat/pyttsx3.git)

```bash
pip install pyttsx3
```

### Code Example for eSpeak

This example python script [text2speech_espeak.py](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/examples/TTS/text2speech_espeak.py) uses `pyttsx3` to make eSpeak speak. It also shows how to use a specific voice model in French and English (JARVIS).

## 2. Piper

### Description of the Piper TTS Engine

Piper is a modern, local TTS engine developed by the **Rhasspy** community. It uses ONNX-based neural models to generate high-quality, much more natural-sounding speech than eSpeak or PicoTTS. It is particularly popular for its speed and efficiency, even on low-power devices like the Raspberry Pi. Once the voice models are downloaded, it works completely offline.

### Installation of the Piper TTS Engine

1. **Install the `piper-tts` Python library**:

See <https://github.com/OHF-Voice/piper1-gpl.git> for details

```bash
pip install piper-tts
```

1. **Download the voice models**: Piper works with external voice models. You need to download two files for each voice:

2.1 The model file (e.g., **`model-name.onnx`**)
2.2 The configuration file (**`model-name.onnx.json`**)

Those models are available on the **Piper repository on Hugging Face** [Link](https://huggingface.co/rhasspy/piper-voices/tree/main). For French and English GB voices, you can find:

- English (GB): [en_GB-alan-low.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en_GB-alan-low.onnx) and [en_GB-alan-low.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en_GB-alan-low.onnx.json)
- French: [fr_FR-gilles-low.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/fr_FR-gilles-low.onnx) and [fr_FR-gilles-low.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/fr_FR-gilles-low.onnx.json)

## 3. Final Choice and TTS Engine Building

The final choice for the TTS engine in this project is **Piper**, which is the default backend in the current configuration and optimized for local execution on Raspberry Pi 5.

The current TTS stack in `src/audio/tts.py` is built around a single offline path:

- Text input via the TTS engine API
- Voice synthesis with Piper (either Python API or CLI)
- Audio playback with PyAudio

The engine uses the `piper-tts` Python package for the default mode and supports alternative Piper CLI usage.

### Configuration Settings

Default TTS settings come from `src/utils/config.py`:

- `engine = piper`
- `model_name = en_US-hfc_female-medium.onnx`
- `cli_mode = false`
- `speed = 1.0`
- `volume = 0.5`(range 0.0 to 1.0)
- `output_sample_rate = 22050`(common for TTS models, but can be adjusted based on the model's requirements and playback capabilities)
- `output_chunk_size = 500`

### How The TTS Engine Works

1. `TTSEngine.speak()` queues text for synthesis.
2. A background playback thread consumes the queue.
3. Piper generates WAV audio either through the Python API or the CLI subprocess.
4. The audio is played back through PyAudio.

The implementation is non-blocking by default, so the assistant can keep listening while speech is queued.

### Supported Modes

#### Piper Python API

- Used when `cli_mode = false`
- Requires the `piper-tts` Python package
- Loads the voice model directly from the configured model path

#### Piper CLI

- Used when `cli_mode = true`
- Searches for a `piper` binary in common locations:
  - `~/.local/bin/piper`
  - `/usr/local/bin/piper`
  - `/usr/bin/piper`
- Useful when you prefer the standalone Piper binary

### Installation

Install the main dependencies used by the current TTS engine:

```bash
pip install piper-tts pyaudio
```

If you want to use the Piper CLI mode, also install the Piper binary from the official project.

### Example Usage

The current voice agent example uses this engine automatically:

```bash
python examples/VAD/voice_agent_offline.py
```

The relevant API surface is:

- `TTSEngine.load()`
- `TTSEngine.speak(text, blocking=False)`
- `TTSEngine.wait()`
- `TTSEngine.interrupt()`
- `TTSEngine.unload()`

### Notes For Custom Voices

- Replace the configured model name if you want a different Piper voice
- Put the `.onnx` file and its matching model data in the configured model directory
- If you change `output_sample_rate`, make sure your playback hardware can handle it cleanly

### What Is Not Part Of The Mainline Engine

The current mainline implementation does not use `pyttsx3` or eSpeak. Those older examples are separate demos and are not the TTS engine used by `src/audio/tts.py`.
