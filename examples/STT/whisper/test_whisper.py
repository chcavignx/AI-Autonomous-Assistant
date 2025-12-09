#!/usr/bin/env python3
import gc
import os
import time

import whisper
from sysutils import (
    detect_raspberry_pi_model,
    limit_cpu_for_multiprocessing,
    print_time_usage,
)

# Paths to the model and config files for French and English voices
DATA_DIR = "../../../data/"
# data for english
TEST_FILE_NAME = "jfk.flac"
ENGLISH = True  # Set to True if the audio is in English,
# False for French
TRANSLATE = False  # Set to True to translate to English,
# False to transcribe in original language

# datas for french
# TEST_FILE_NAME = "jfk_fr.flac"
# ENGLISH = False  # Set to True if the audio is in English, False for French
# TRANSLATE = True  # Set to True to translate to English, False to transcribe in original language

audio_file = os.path.join(DATA_DIR, TEST_FILE_NAME)
print("=== Script initialization ===")
start_time = time.time()
print_time_usage("Init", start_time)

# --- Optional parameters ---
CORES_TO_USE = 2  # Limit to 2 cores
# limit_cpu_for_multiprocessing(CORES_TO_USE)
if detect_raspberry_pi_model():
    limit_cpu_for_multiprocessing(CORES_TO_USE)
    MODEL_ID = f"tiny{'.en' if ENGLISH else ''}"  # "tiny" (Recommended model for low resources)
else:
    limit_cpu_for_multiprocessing()  # Use all available cores
    MODEL_ID = "medium"  # "large-v3", "medium", "small", "large-v3", "base", "tiny"
print(f"Selected model: {MODEL_ID}")
print_time_usage("After model load", start_time)
# --- Whisper Transcription ---
model = whisper.load_model(MODEL_ID)
# download_root = "~/.cache/whisper" # Optional, default is ~/.cache/whisper
# device = "cpu"  or "cuda" if you have a GPU and the right setup
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model(MODEL_ID, download_root=download_root, device=device)
try:
    start_time = time.time()
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        fp16=False,
        language="en" if ENGLISH else "fr",
        task="translate" if TRANSLATE else "transcribe",
    )
    print("Transcription:", result["text"])
    print_time_usage("After transcription", start_time)
except RuntimeError as e:
    print(f"Error during transcription: {e}, trying a smaller model")

# Force cleanup
gc.collect()
print("=== End of script ===")
