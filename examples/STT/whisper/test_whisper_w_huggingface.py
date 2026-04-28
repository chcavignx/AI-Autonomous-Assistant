#!/usr/bin/env python3
"""This script demonstrates how to run automatic speech recognition (ASR)
using Hugging Face's Transformers and Datasets libraries,
with optimizations for low-resource devices such as the Raspberry Pi.
It loads a Whisper model from local cache, processes an audio file,
and outputs the transcription. The script includes system resource
monitoring, device-specific optimizations, and error handling.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_python import PreTrainedTokenizer
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config
from src.utils.sysutils import (
    detect_raspberry_pi_model,
    limit_cpu_for_multiprocessing,
    print_time_usage,
)

# Paths to the model and config files for French and English voices
MODEL_DIR = str(config.paths.models_path / "whisper")
DATA_DIR = str(config.paths.data_path)
TEST_FILE_NAME = "jfk.flac"


class _ModelLike(Protocol):
    def to(self, device: str) -> object: ...


class _ProcessorLike(Protocol):
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    feature_extractor: PreTrainedFeatureExtractor


class _PipelineLike(Protocol):
    def __call__(self, audio: str, *, generate_kwargs: dict[str, str]) -> object: ...


class _ModelFromPretrained(Protocol):
    def __call__(self, pretrained_model_name_or_path: str, **kwargs: object) -> PreTrainedModel: ...


class _ProcessorFromPretrained(Protocol):
    def __call__(self, pretrained_model_name_or_path: str, **kwargs: object) -> _ProcessorLike: ...


def main() -> None:
    """Main execution function."""
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

    audio_file = os.path.join(DATA_DIR, TEST_FILE_NAME)

    cache_dir = os.path.join(MODEL_DIR, "huggingface")
    # ----------------------
    # Model loading
    # ----------------------
    start_time = time.time()
    model_from_pretrained = cast(_ModelFromPretrained, AutoModelForSpeechSeq2Seq.from_pretrained)
    model = cast(
        _ModelLike,
        model_from_pretrained(
            pretrained_model_name_or_path=model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,  # Use float16
            local_files_only=True,  # Use only local cached files
            low_cpu_mem_usage=bool(detect_raspberry_pi_model()),  # Critical for Pi
            use_safetensors=True,
        ),
    )
    # print_sys_usage("After model load")
    print_time_usage("After model load", start_time)

    start_time = time.time()
    processor_from_pretrained = cast(_ProcessorFromPretrained, AutoProcessor.from_pretrained)
    processor = processor_from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,  # Use only local cached files
    )
    # print_sys_usage("After processor load")
    print_time_usage("After processor load", start_time)

    # ----------------------
    # Loading Dataset
    # ----------------------
    start_time = time.time()

    # print_sys_usage("After dataset load")
    print_time_usage("After dataset load", start_time)

    # ----------------------
    # Device configuration
    # ----------------------
    start_time = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16  # if torch.cuda.is_available() else torch.float32
    # print_sys_usage("Device config")
    print_time_usage("Device config", start_time)

    # ----------------------
    # Move model to device
    # ----------------------
    start_time = time.time()
    model.to(device)
    # print_sys_usage("After model.to(device)")
    print_time_usage("After model.to(device)", start_time)

    # Force cleanup
    gc.collect()

    # ----------------------
    # ASR Pipeline
    # ----------------------
    start_time = time.time()
    pipe = cast(
        _PipelineLike,
        pipeline(
            "automatic-speech-recognition",
            model=cast(PreTrainedModel, model),
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=-1,
            model_kwargs={"low_cpu_mem_usage": bool(detect_raspberry_pi_model())},
        ),
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

    # print_sys_usage("Before transcription")
    print_time_usage("Before transcription", time.time())
    result = None  # Initialize with a default value

    # Process audio (replace with your file)
    try:
        start_time = time.time()
        result = pipe(audio_file, generate_kwargs=generate_kwargs)
        print_time_usage("After transcription", start_time)
    except (RuntimeError, ValueError):
        pass

    # ----------------------
    # Results
    # ----------------------
    if result is not None:
        pass
        # print(result)  # Uncomment to display all

    # print(result)  # Uncomment to display all

    # Force cleanup
    gc.collect()


if __name__ == "__main__":
    main()
