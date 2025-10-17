# Key Extensibility Features

## 1. **Clean Architecture**

- Separated concerns: VAD, STT, TTS, AI Response, and orchestration
- Configuration-driven with `AgentConfig` dataclass
- Easy to swap STT engines (Whisper/Vosk) via enum

## 2. **AI Integration Points**

- `AIResponseEngine.generate_ai_response()` - **Main extension point for AI models**
- Ready for OpenAI API, local LLMs, HuggingFace models
- Context support for conversation memory

## 3. **Clear Function Separation**

- `_process_voice_command()` - Command processing logic
- `_generate_ai_response()` - AI response generation
- `_execute_command()` - Command execution
- `speak_response()` - TTS output

## 4. **Extension Examples**

**Add OpenAI integration:**

```python
def generate_ai_response(self, user_input: str, context: Dict = None) -> str:
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content
```

**Add local LLM:**

```python
def generate_ai_response(self, user_input: str, context: Dict = None) -> str:
    from transformers import pipeline
    generator = pipeline('text-generation', model='microsoft/DialoGPT-medium')
    response = generator(user_input, max_length=100)
    return response[0]['generated_text']
```

This architecture makes it extremely easy to integrate any AI model while maintaining clean separation of voice processing and AI logic.

## Installation Steps

### 1. Update your system and install dependencies

```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-pip git portaudio19-dev python3-pyaudio alsa-utils
```

### 2. Install the Whisper Python module for offline use

```bash
pip3 install git+https://github.com/openai/whisper.git
pip3 install blobfile
```

Download the models, vocabulary, and encoder files to use Whisper offline with the following script:

```bash
python whisper_objects.py
```

The vocabulary, encoder, and model files will be stored in `($HOME_USER_DIR)/.cache/whisper`.

### 3. Install the faster whisper Python module for offline use

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper silero-vad pyaudio
```

Models will be downloaded automatically on first use in ./models directory, or you can create it manually:

```bash
mkdir -p ./models
```
Then the first time you run the script you must be online to download the models.

Or use the following script to download the models:

```bash
python faster_whisper_objects.py
```

The models will be saved to the `($HOME_USER_DIR)/.cache/huggingface` directory.
Then you can indicate the path when you load the models (download_root= `($HOME_USER_DIR)/.cache/huggingface`).

### 4. Installing Piper TTS

```bash
pip install piper-tts soundfile sounddevice
```

Download the voice models from https://huggingface.co/rhasspy/piper-voices

Or if you want to use the JARVIS voice:
[jarvis-medium.onnx](https://huggingface.co/jgkawell/jarvis/blob/main/en/en_GB/jarvis/medium/jarvis-medium.onnx) and [jarvis-medium.onnx.json](https://huggingface.co/jgkawell/jarvis/blob/main/en/en_GB/jarvis/medium/jarvis-medium.onnx.json)

You can find it here: https://huggingface.co/jgkawell/jarvis/tree/main/en/en_GB/jarvis/medium


### 5. Demo Application

The demo application is available at https://github.com/chcavignx/AI-Autonomous-Assistant/tree/main/demo/voice_agent_offline.py

```bash
python voice_agent_offline.py
```

The wake word is set to **"Thanos"** to simplify recognition.

### Conclusion

After evaluating the performance of the different models on the Raspberry Pi configuration used, the faster_whisper model emerges as the best choice, as predicted by the model comparison.

The faster_whisper model is more accurate and faster to process, while the Whisper model is slightly more accurate but slower to process.

The voice agent is able to recognize and respond to voice commands from the user, using the faster_whisper model for speech-to-text conversion and the Piper TTS model for text-to-speech.
