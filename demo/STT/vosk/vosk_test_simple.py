#!/usr/bin/env python3
import wave
import sys
import os
from vosk import Model, KaldiRecognizer, SetLogLevel

# You can set log level to -1 to disable debug messages
SetLogLevel(0)
# Check if the audio file is provided as a command line argument
if len(sys.argv) != 2:
    print("Usage: python vosk_test_simple.py <audio_file.wav>")
    sys.exit(1)
# Open the audio file
wf = wave.open(sys.argv[1], "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)
# Initialize the Vosk model
# You can initialize the model with a specific language or use the default model
# Uncomment the following line to use a specific language model from online repository
#model = Model(lang="en-us")


model_name="vosk-model-small-en-us-0.15"
# You can also init model by name or with a folder when you have downloaded it
# Uncomment the following line to use a specific model from HuggingFace or local directory
# model = Model(model_name=model_name)

# If you have already downloaded the model, you can load it like this:
vosk_local_dir = ".cache/vosk"
cache_dir = os.path.join(os.path.expanduser("~"), vosk_local_dir)
local_dir = os.path.join(cache_dir, model_name)
if not os.path.exists(local_dir):
    print(f"Model {model_name} not found. Please download it from https://alphacephei.com/vosk/models")
    sys.exit(1) 
# Load the model from the local directory
print(f"Loading model from {model_name}")
model = Model(local_dir)
# Initialize the Kaldi recognizer with the model and sample rate
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)
# Read the audio file in chunks and process it
print("Starting transcription...")
# You can also use rec.AcceptWaveform(data) to process the audio in chunks
# or rec.PartialResult() to get partial results
# or rec.Result() to get final results
# or rec.FinalResult() to get final results with timestamps
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())
# Print the final result
print(rec.FinalResult())
