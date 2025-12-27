#!/usr/bin/env python3
"""
Example of text to speech using Piper
"""

import os
import wave

import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

# Paths to the model and config files for French and English voices
DATA_DIR = "../../data/"
TEST_FILE_NAME = "test.wav"

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=True,  # use raw audio from voice
    speaker_id=1,  # None, # default speaker (multi-speaker voices only)
)


# Service to create unique filenames from base name, suffix and counter if needed
def setup_output_filename(base_path, base_name, suffix, counter=1):
    """Generates a unique filename by appending a counter if needed."""
    counter = 1
    file_name, file_extension = os.path.splitext(base_name)
    output_file = os.path.join(base_path, f"{file_name}_{suffix}{file_extension}")

    while os.path.exists(output_file):
        output_file = os.path.join(
            base_path, f"{file_name}_{suffix}_{counter}{file_extension}"
        )
        counter += 1

    return output_file


# Function to synthesize text to speech and save as a WAV file
def synthesize_voice_and_save(model_path, text, output_file):
    """Synthesizes text to audio, saves it and plays."""
    # Create a Piper object
    voice = PiperVoice.load(model_path)

    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(
            text=text, wav_file=wav_file, set_wav_format=True, syn_config=syn_config
        )

    print(f"Audio file saved to: {output_file}")
    # Lecture du fichier généré
    # data, fs = sf.read(output_file, dtype='int16')
    # sd.play(data, fs)
    # sd.wait()
    # #sd.sleep(100)  # Pause to ensure playback completes
    # sd.stop()
    # print("Synthesis complete.")


def synthesize_voice(model_path, text):
    """Synthesizes text to audio and plays it."""
    # Create a Piper object
    voice = PiperVoice.load(model_path)
    stream = sd.OutputStream(
        samplerate=voice.config.sample_rate, channels=1, dtype="int16"
    )
    stream.start()
    for audio_bytes in voice.synthesize(text):
        stream.write(audio_bytes.audio_int16_array)

    stream.stop()
    stream.close()
    print("Synthesis complete.")


# Example for the French voice
model_fr = os.path.join(DATA_DIR, "fr_FR-gilles-low.onnx")
# TEXT_FR = "Ceci est un test avec voix française utilisant le moteur Piper."
TEXT_FR = " Je m'appelle Gilles et je suis ravi de vous rencontrer."
TEXT_FR += " Je suis très heureux de pouvoir parler avec vous aujourd'hui."
TEXT_FR += " J'espère que vous apprécierez cette démonstration."
file_fr = setup_output_filename(
    DATA_DIR, TEST_FILE_NAME, model_fr.split("/")[-1].replace(".onnx", "")
)
synthesize_voice_and_save(model_fr, TEXT_FR, file_fr)
synthesize_voice(model_fr, TEXT_FR)
# Example for the English (GB) voice
model_en = os.path.join(DATA_DIR, "jarvis-medium.onnx")
# TEXT_EN = "This is a test in British English using the Piper engine."
TEXT_EN = " My name is Jarvis and I am delighted to meet you."
TEXT_EN += " I am very happy to be able to speak with you today."
TEXT_EN += " I hope you will enjoy this demonstration."
file_en = setup_output_filename(
    DATA_DIR, TEST_FILE_NAME, model_en.split("/")[-1].replace(".onnx", "")
)
synthesize_voice_and_save(model_en, TEXT_EN, file_en)
synthesize_voice(model_en, TEXT_EN)
