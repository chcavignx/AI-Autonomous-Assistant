# Whisper Installation & Demo on Raspberry Pi 5 (CPU-only)

## Introduction

Install OpenAI Whisper for local speech-to-text (STT) using CPU only on Raspberry Pi 5. This guide includes installation and a demo app with simple and advanced voice commands.
Main sources for the installation process and building the demo app:
<https://github.com/openai/whisper.git>
<https://github.com/openai/whisper/discussions/1463#discussion-5324136>
<https://github.com/Uberi/speech_recognition.git>
<https://github.com/davabase/whisper_real_time.git>
<https://github.com/Nerdy-Things/openai-whisper-raspberry-pi.git>
<https://github.com/graykode/gpt-2-Pytorch/tree/master/GPT2>
<https://huggingface.co/openai/whisper-large-v2#long-form-transcription>

## Installation Steps

1. Update system and install dependencies:

```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-pip git portaudio19-dev python3-pyaudio
```

2. Install python module for Whisper offline Use:

```bash
pip3 install git+https://github.com/openai/whisper.git
pip3 install blobfile
```

Alternatively, you can clone it:

```bash
git clone https://github.com/openai/whisper.git
cd whisper
pip3 install -e .
```
and Install modules to execute tests into whisper repository

```bash
pip3 install jiwer scipy pytest
```

Load models, vocabulary and encoder files to be able to use whisper offline with the following script:

```bash
chmod +x  whisper_objects.sh
./whisper_objects.sh
```

The vocabulary, encoder and models files will be store in ($HOME_USER_DIR)/.cache/whisper

3. SpeechRecognition:

Follow installation from <https://github.com/Uberi/speech_recognition.git>
Mainly:
Update file links in your local copy of openai_public.py which will be installed in your python folder e.g. /lib/python3.11/site-packages/tiktoken_ext/openai_public.py to point to where you downloaded the files.
Remove the URL "<https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/>" and replace it with your local copy, e.g. "($HOME_USER_DIR)/.cache/whisper/vocab.bpe" and "($HOME_USER_DIR).cache/whisper/encoder.json"

```bash
def gpt2():
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file="/$HOME/.cache/whisper/vocab.bpe",
        encoder_json_file="/$HOME/.cache/whisper/encoder.json",
    )
```

```bash
pip3 install SpeechRecognition[whisper-local]
```

## Demo python App
```bash
python3 test_whisper.py
```

## Alternative

### Integrating Whisper with Hugging Face Transformers

Whisper models are fully supported in the Hugging Face Transformers ecosystem. To enable model execution via Transformers, first install the Transformers library:

```bash
pip install transformers
```

For our application, we also recommend installing the Datasets library to retrieve sample audio datasets from the Hugging Face Hub and Accelerate to optimize model loading and inference times:

```bash
pip install datasets accelerate
```

These libraries not only streamline the setup process but also enhance performance and flexibility when deploying Whisper models in diverse environments.
To enable the use of models and datasets locally, firstly install the huggingface_hub package with the following command:

```bash
pip3 install --upgrade huggingface_hub
```

Secondly load models and data sets from:
<https://huggingface.co/openai/whisper-large-v3-turbo>
<https://huggingface.co/openai/whisper-tiny>
<https://huggingface.co/distil-whisper/distil-large-v3>
<https://huggingface.co/distil-whisper/distil-large-v3.5>
<https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy>
<https://huggingface.co/datasets/distil-whisper/librispeech_long>


with the following python script that will store the model and data set in ($HOME_USER_DIR)/.cache/huggingface directory
(following : https://huggingface.co/docs/huggingface_hub/guides/download)

```bash
python3 load_huggingface_objects.py
```

## Demo python App
```bash
python3 test_whisper_w_huggingface.py
```
