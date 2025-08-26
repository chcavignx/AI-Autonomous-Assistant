### Offline Voice Recognition (USB Microphone)

The goal is to enable the user to issue voice commands to the assistant without an internet connection. Extensive documentation and numerous examples are available; however, two methods are best suited for the hardware in use: Vosk and OpenAI Whisper (which is more resource-intensive but also highly effective).

## Test Hardware

See tests/audio_usb_test.md

## Vosk

One of the best open-source solutions for offline voice recognition on the Raspberry Pi, compatible with multiple languages.

**Precision**
Achieves a recognition rate of 85% to 95% on short commands and limited vocabulary in a calm environment [4].

**Performance**
Processes commands almost instantly (latency < 1 second per command).

**Acceleration**
CUDA acceleration is available with NVIDIA GPUs. Note that the Hailo module is not supported.

**Installation and Test**
See STT_offline_vosk_rpi5.md

## Whisper

**Precision**
Up to 90% precision on short sentences, although it is slower than Vosk.

**Performance (Speed)**
Latency ranges from 2 to 5 seconds depending on the sentence length and the selected model size.

**Acceleration**
Acceleration is possible using the Hailo module.

**Installation and Test**
See STT_offline_whisper_rpi5.md

### Hugging Face Transformers & Distil-Whisper

An alternative implementation involves using the Hugging Face Transformers library along with the Distil-Whisper model. This approach offers several advantages:
- Reduced computational requirements compared to the full Whisper model.
- Faster inference times, making it more suitable for devices with limited resources.
- Comparable recognition accuracy on short commands.

This solution is ideal for users seeking a balance between performance and efficiency.

