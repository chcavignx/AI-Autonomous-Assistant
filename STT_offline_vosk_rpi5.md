# Vosk Installation & Demo on Raspberry Pi 5 (CPU-only)

## Introduction

Install Vosk for local speech-to-text (STT) using CPU only on Raspberry Pi 5. This guide includes installation and a demo app with simple voice commands.
Main sources for the installation process and building the demo app:

<https://alphacephei.com/vosk/install>

## Installation Steps

1. Update system and install dependencies:

```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-pip git portaudio19-dev python3-pyaudio
```

2. Install python module for Vosk:

```bash
pip3 install vosk
```

Download models (from <https://alphacephei.com/vosk/models>) files to be able to use vosk offline with the following script:

```bash
chmod +x  vosk_models.sh
./vosk_models.sh
```

3. SpeechRecognition:

Follow installation from <https://github.com/Uberi/speech_recognition.git>

```bash
pip install SpeechRecognition[vosk]
```

## testing Vosk

You can transcribe a file with a simple vosk-transcriber command line tool:

```bash
vosk-transcriber -i test.mp4 -o test.txt
vosk-transcriber -i test.mp4 -t srt -o test.srt
vosk-transcriber -l fr -i test.m4a -t srt -o test.srt
vosk-transcriber --list-languages
```

To run python samples, you can clone the vosk-api and run python example after modifying the path of the local model, previously downloaded, the following commands:

```bash
git clone https://github.com/alphacep/vosk-api
cd vosk-api/python/example
python3 ./test_simple.py jfk.flac
```

When using your own audio file make sure it has the correct format - PCM 16khz 16bit mono. Otherwise, if you have ffmpeg installed, you can use test_ffmpeg.py, which does the conversion for you.

Find more examples such as using a microphone, decoding with a fixed small vocabulary or speaker identification setup in the python/example subfolder

## Demo App
