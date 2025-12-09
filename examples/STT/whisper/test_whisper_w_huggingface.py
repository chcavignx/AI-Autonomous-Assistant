#!/usr/bin/env python3
"""
This script demonstrates how to run automatic speech recognition (ASR)
using Hugging Face's Transformers and Datasets libraries,
with optimizations for low-resource devices such as the Raspberry Pi.
It loads a Whisper model from local cache, processes an audio file,
and outputs the transcription. The script includes system resource
monitoring, device-specific optimizations, and error handling.
"""

import gc
import os
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.utils.sysutils import (
    detect_raspberry_pi_model,
    limit_cpu_for_multiprocessing,
    print_time_usage,
)

# Paths to the model and config files for French and English voices
DATA_DIR = "../../../data/"
TEST_FILE_NAME = "jfk.flac"


def main():
    """Main execution function"""
    print("=== Script initialization ===")
    # Raspberry Pi optimizations
    cores_to_use = 2  # Limit to 2 cores

    if detect_raspberry_pi_model():
        os.environ["PYTORCH_JIT"] = "0"
        limit_cpu_for_multiprocessing(cores_to_use)
        torch.set_float32_matmul_precision("high")  # For Pi 5
        torch.backends.cuda.matmul.allow_tf32 = True  # For Pi 5
        torch.set_num_threads(cores_to_use)  # Adjust based on your Pi's CPU cores
        # model recommended for low resources
        model_id = "openai/whisper-tiny"
        # model_id = "distil-whisper/distil-large-v3"
        # dataset_name = "distil-whisper/librispeech_long"
    else:
        limit_cpu_for_multiprocessing()  # Use all available cores
        # For more powerfull devices, you can use a larger model
        model_id = "openai/whisper-large-v3-turbo"

    print(f"Selected model: {model_id}")

    audio_file = os.path.join(DATA_DIR, TEST_FILE_NAME)

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
    # ----------------------
    # Model loading
    # ----------------------
    print(">>> Loading local model…")
    start_time = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=torch.float16,  # Use float16
        local_files_only=True,  # Use only local cached files
        low_cpu_mem_usage=(
            True if detect_raspberry_pi_model() else False
        ),  # Critical for Pi
        use_safetensors=True,
    )
    # print_sys_usage("After model load")
    print_time_usage("After model load", start_time)

    print(">>> Loading processor…")
    start_time = time.time()
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,  # Use only local cached files
    )
    # print_sys_usage("After processor load")
    print_time_usage("After processor load", start_time)

    # ----------------------
    # Loading Dataset
    # ----------------------
    print(">>> Loading local dataset…")
    start_time = time.time()

    # print_sys_usage("After dataset load")
    print_time_usage("After dataset load", start_time)

    # ----------------------
    # Device configuration
    # ----------------------
    start_time = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16  # if torch.cuda.is_available() else torch.float32
    print(f">>> Device used: {device}, dtype: {torch_dtype}")
    # print_sys_usage("Device config")
    print_time_usage("Device config", start_time)

    # ----------------------
    # Move model to device
    # ----------------------
    print(">>> Moving model to device…")
    start_time = time.time()
    model.to(device)
    # print_sys_usage("After model.to(device)")
    print_time_usage("After model.to(device)", start_time)

    # Force cleanup
    gc.collect()

    # ----------------------
    # ASR Pipeline
    # ----------------------
    print(">>> Creating ASR pipeline…")
    start_time = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=-1,
        processor=processor,
        model_kwargs={
            "low_cpu_mem_usage": True if detect_raspberry_pi_model() else False
        },
    )
    # print_sys_usage("After pipeline creation")
    print_time_usage("After pipeline creation", start_time)

    # ----------------------
    # Generation parameters
    # ----------------------
    generate_kwargs = {
        "language": "english",
        "task": "transcribe",
    }

    print(">>> Ready to analyze audio: jfk.flac")
    # print_sys_usage("Before transcription")
    print_time_usage("Before transcription", time.time())
    result = None  # Initialize with a default value

    # Process audio (replace with your file)
    try:
        start_time = time.time()
        result = pipe(audio_file, generate_kwargs=generate_kwargs)
        print_time_usage("After transcription", start_time)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        print("Try reducing chunk_length_s or using a smaller model")

    # ----------------------
    # Results
    # ----------------------
    if result is not None:
        print(f">>> Transcription result: {result['text']}")
        # print(result)  # Uncomment to display all
    else:
        print("Transcription failed. No result available.")

    # print(result)  # Uncomment to display all

    # Force cleanup
    gc.collect()

    print("=== End of script ===")


if __name__ == "__main__":
    main()
