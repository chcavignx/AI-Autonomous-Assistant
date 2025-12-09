# Offline Voice Synthetiser, text-to-speech (TTS)

The goal is to enable the assitant to generate voice from text to the user without an internet connection. As usual, extensive documentation and numerous examples are available; however, again, two methods are best suited for the hardware in use: eSpeak (via `pyttsx3`) and Piper.

## 1\. eSpeak (via `pyttsx3`)

### Description of the eSpeak TTS Engine

eSpeak is a lightweight, open-source, and very simple-to-use TTS engine that works offline. It's widely available on most operating systems. Its main characteristic is its compact size, but the voice quality is quite robotic and synthetic, making it less suitable for applications that need a natural sound. The Python library **`pyttsx3`** acts as a wrapper that allows you to use eSpeak (or other engines like Sapi5 on Windows and NSSpeechSynthesizer on macOS) in a simple and portable way.

### Installation of the eSpeak TTS Engine

1.**Install dependencies**

```bash
pip3 install sounddevice
```

2.**Install the eSpeak engine**:

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

3.**Install the `pyttsx3` Python library**:

For more details, visit the pyttsx3 GitHub repository at: [https://github.com/nateshmbhat/pyttsx3.git](https://github.com/nateshmbhat/pyttsx3.git)

```bash
pip install pyttsx3
```

### Code Example for eSpeak

This example python scryptv [text2speech_espeak.py](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/examples/TTS/text2speech_espeak.py) uses `pyttsx3` to make eSpeak speak. It also shows how to use a specific voice model in French and English (JARVIS).

## 2\. Piper

### Description of the Piper TTS Engine

Piper is a modern, local TTS engine developed by the **Rhasspy** community. It uses ONNX-based neural models to generate high-quality, much more natural-sounding speech than eSpeak or PicoTTS. It is particularly popular for its speed and efficiency, even on low-power devices like the Raspberry Pi. Once the voice models are downloaded, it works completely offline.

### Installation of the Piper TTS Engine

1.**Install the `piper-tts` Python library**:

See <https://github.com/OHF-Voice/piper1-gpl.git> for details

```bash
pip install piper-tts
```

2.**Download the voice models**: Piper works with external voice models. You need to download two files for each voice:

2.1 The model file (e.g., **`model-name.onnx`**)
2.2 The configuration file (**`model-name.onnx.json`**)

Those models are available on the **Piper repository on Hugging Face**. For French and English GB voices, you can use models like these (names may change):

**French**: `fr_FR-gilles-low.onnx` and `fr_FR-gilles-low.onnx.json`
**English (GB)**: `en_GB-vctk-medium.onnx` and `en_GB-vctk-medium.onnx.json`

For JARVIS voice you can find it here: <https://huggingface.co/jgkawell/jarvis/tree/main/en/en_GB/jarvis/medium>

### Code Example for Piper

This example python script [text2speech_piper.py](https://github.com/chcavignx/AI-Autonomous-Assistant/blob/main/examples/TTS/text2speech_piper.py) uses `pyttsx3` to make eSpeak speak. It also shows how to use a specific voice model in French and English (JARVIS).

To execute the test script, install the following module:

```bash
pip install piper-tts soundfile sounddevice
```

## Comparison: eSpeak vs. Piper

| Characteristic | eSpeak (via pyttsx3) | Piper |
| :--- | :--- | :--- |
| **Voice Quality** | **Low**, synthetic and robotic voice. | **High**, natural and fluid voice. |
| **Technology** | Based on phonetic rules. | Neural models (neural networks). |
| **Performance** | Very fast and lightweight. | Very fast, optimized for low-power devices. |
| **Installation** | Easy if the eSpeak engine is already installed. | Requires installing the library and downloading voice models. |
| **Size** | Very compact. | Size depends on the downloaded voice model, but is still reasonable. |
| **Flexibility** | Limited number of voices and languages by the engine. | Wide choice of voice models and languages. |
| **Requirements** | Low, works on most systems. | Low, but neural models can require more resources for the initial generation. |
