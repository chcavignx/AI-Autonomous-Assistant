#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol, cast

import whisper

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config
from src.utils.sysutils import (
    detect_raspberry_pi_model,
    limit_cpu_for_multiprocessing,
    print_time_usage,
)

MODEL_DIR = str(config.paths.models_path / "whisper")
# Paths to the model and config files for French and English voices
DATA_DIR = str(config.paths.data_path)
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
start_time = time.time()
print_time_usage("Init", start_time)


class _WhisperModelLike(Protocol):
    def transcribe(self, audio: str, **kwargs: Any) -> dict[str, Any]: ...


# --- Optional parameters ---
CORES_TO_USE = 2  # Limit to 2 cores
# limit_cpu_for_multiprocessing(CORES_TO_USE)
model_id = "medium"
if detect_raspberry_pi_model():
    _ = limit_cpu_for_multiprocessing(CORES_TO_USE)
    model_id = f"tiny{'.en' if ENGLISH else ''}"  # "tiny" (Recommended model for low resources)
else:
    _ = limit_cpu_for_multiprocessing()  # Use all available cores
    model_id = "medium"  # "large-v3", "medium", "small", "large-v3", "base", "tiny"
print_time_usage("After model load", start_time)
# --- Whisper Transcription ---
model = cast("_WhisperModelLike", whisper.load_model(model_id, download_root=MODEL_DIR))
# download_root = "~/.cache/whisper" # Optional, default is ~/.cache/whisper
# device = "cpu"  or "cuda" if you have a GPU and the right setup
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model(MODEL_ID, download_root=download_root, device=device)
try:
    start_time = time.time()
    _ = model.transcribe(
        audio_file,
        word_timestamps=True,
        fp16=False,
        language="en" if ENGLISH else "fr",
        task="translate" if TRANSLATE else "transcribe",
    )
    print_time_usage("After transcription", start_time)
except RuntimeError:
    pass

# Force cleanup
_ = gc.collect()
