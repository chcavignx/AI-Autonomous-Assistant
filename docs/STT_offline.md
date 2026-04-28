# Offline Voice Recognition (USB Microphone)

The goal is to enable the user to issue voice commands to the assistant without an internet connection. Extensive documentation and numerous examples are available; however, two methods are best suited for the hardware in use: Vosk and OpenAI Whisper (which is more resource-intensive but also highly effective).

## Test Hardware

See [test usb devise](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/docs/audio_usb_test.md)

## Vosk

One of the best open-source solutions for offline voice recognition on the Raspberry Pi, compatible with multiple languages.

**Precision**
Achieves a recognition rate of 85% to 95% on short commands and limited vocabulary in a calm environment (accuracy decreases for similar objects and crowded scenes).

**Performance**
Processes commands almost instantly (latency < 1 second per command).

**Acceleration**
CUDA acceleration is available with NVIDIA GPUs. Note that the Hailo module is not supported.

**Installation and Test**

See [Vosk guide](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/examples/STT/vosk/STT_offline_vosk_rpi5.md)

## Whisper

**Precision**
Up to 90% precision on short sentences, although it is slower than Vosk.

**Performance (Speed)**
Latency ranges from 2 to 5 seconds depending on the sentence length and the selected model size.

**Acceleration**
Acceleration is possible using the Hailo module.

**Installation and Test**

See [Whisper guide](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/examples/STT/whisper/STT_offline_whisper_rpi5.md)

### Hugging Face Transformers & Distil-Whisper

An alternative implementation involves using the Hugging Face Transformers library along with the Distil-Whisper model. This approach offers several advantages:

- Reduced computational requirements compared to the full Whisper model.
- Faster inference times, making it more suitable for devices with limited resources.
- Comparable recognition accuracy on short commands.

This solution is ideal for users seeking a balance between performance and efficiency.

## Final Choice and ASR Library Building

The final choice for the ASR engine in this project is **Faster-Whisper**, which is the default backend in the current configuration and optimized for CPU inference on Raspberry Pi 5.

The current audio stack in `src/audio/asr.py` is built around a single offline path:

- Microphone capture with PyAudio
- Speech segmentation with Silero VAD
- Transcription with Faster-Whisper or OpenAI Whisper
- Callback-based delivery of the recognized text

If Silero VAD cannot be loaded, the engine falls back to a simple RMS energy check so transcription can still run.

### Configuration Settings

Default ASR settings come from `src/utils/config.py`:

- `engine = faster-whisper`
- `model_size = tiny`
- `language = en`
- `device = cpu`
- `compute_type = int8`
- `input_sample_rate = 22050`
- `input_chunk_ms = 30`

### How The ASR Engine Works

1. The microphone stream is opened with PyAudio.
2. If the device does not run at the configured sample rate, the engine resamples audio in software.
3. Silero VAD detects speech vs silence.
4. Speech chunks are buffered until silence marks the end of the utterance.
5. The buffered audio is transcribed and passed to the callback provided to `ASREngine.start()`.

### Supported Backends

#### Faster-Whisper

- Default backend in the current config
- Good fit for Raspberry Pi 5
- Uses `beam_size=3` and `vad_filter=True` in the current implementation

#### OpenAI Whisper

- Supported as an alternate backend
- Uses the `openai-whisper` Python package
- Good if you want the classic Whisper API path

### Installation

Install the runtime pieces used by the current ASR engine:

```bash
pip install pyaudio torch scipy silero-vad faster-whisper
```

Install OpenAI Whisper only if you want to switch the config to `engine: whisper`:

```bash
pip install openai-whisper
```

### Example Usage

The integrated example that exercises the live audio stack is:

```bash
python examples/VAD/voice_agent_offline.py
```

That example wires together wake-word detection, ASR, and TTS.

### Practical Notes

- `ASREngine` is offline once the model files are available locally
- The engine uses `Silero VAD` first and only falls back to energy-based detection if needed
- The current implementation is callback-driven, so the recognized text is delivered asynchronously
- The helper `transcribe_file()` is available for file-based testing and debugging
- On some ARM/PortAudio stacks, native model teardown can segfault at process exit. If you see this,
  set `asr.skip_native_teardown: true` in `config.yaml` to skip native teardown.

### Audio Test Hardware

Before validating ASR, it is useful to check the USB microphone and speaker path with the repository audio test script and device listing guide:

- `scripts/list_audio_devices.py`
- `scripts/tests/audio_test.sh`
- `docs/audio_usb_test.md`
