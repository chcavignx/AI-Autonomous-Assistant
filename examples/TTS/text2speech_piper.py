#!/usr/bin/env python3
"""Example of text to speech using Piper."""

import os
import pathlib
import sys
import wave
from collections.abc import Iterable
from os import PathLike
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from piper import PiperVoice, SynthesisConfig

# Ensure repo root is accessible
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.audio.audio_utils import create_output_stream
from src.utils.config import config

# Paths to the model and config files for French and English voices
MODEL_DIR = str(config.paths.models_path / "piper")
DATA_DIR = str(config.paths.data_path)
TEST_FILE_NAME = "test.wav"

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=True,  # use raw audio from voice
    speaker_id=1,  # None, # default speaker (multi-speaker voices only)
)


class _AudioChunkLike(Protocol):
    audio_int16_array: NDArray[np.int16]


class _PiperVoiceLike(Protocol):
    class _ConfigLike(Protocol):
        sample_rate: int

    config: _ConfigLike

    @staticmethod
    def load(model_path: str | PathLike[str]) -> "_PiperVoiceLike": ...

    def synthesize_wav(
        self,
        *,
        text: str,
        wav_file: wave.Wave_write,
        set_wav_format: bool,
        syn_config: SynthesisConfig,
    ) -> object: ...

    def synthesize(self, text: str) -> Iterable[_AudioChunkLike]: ...


# Service to create unique filenames from base name, suffix and counter if needed
def setup_output_filename(
    base_path: str | PathLike[str],
    base_name: str,
    suffix: str,
    counter: int = 1,
) -> str:
    """Generates a unique filename by appending a counter if needed."""
    counter = 1
    resolved_base_path = os.fspath(base_path)
    file_name, file_extension = os.path.splitext(base_name)
    output_file = os.path.join(resolved_base_path, f"{file_name}_{suffix}{file_extension}")

    while pathlib.Path(output_file).exists():
        output_file = os.path.join(resolved_base_path, f"{file_name}_{suffix}_{counter}{file_extension}")
        counter += 1

    return output_file


# Function to synthesize text to speech and save as a WAV file
def synthesize_voice_and_save(
    model_path: str | PathLike[str],
    text: str,
    output_file: str | PathLike[str],
) -> None:
    """Synthesizes text to audio, saves it and plays."""
    # Create a Piper object
    voice = cast("_PiperVoiceLike", cast(object, PiperVoice.load(os.fspath(model_path))))

    stream = create_output_stream(
        rate=voice.config.sample_rate,
        chunk_frames=512,
    )

    if stream.start():
        for audio_chunk in voice.synthesize(text):
            stream.write(audio_chunk.audio_int16_array)
        stream.stop()

    stream.close()


def main() -> None:
    """Main function to demonstrate text-to-speech using Piper."""
    # Example for the French voice
    model_fr = os.path.join(MODEL_DIR, "fr_FR-gilles-low.onnx")
    # text_fr = "Ceci est un test avec voix française utilisant le moteur Piper."
    text_fr = " Je m'appelle Gilles et je suis ravi de vous rencontrer."
    text_fr += " Je suis très heureux de pouvoir parler avec vous aujourd'hui."
    text_fr += " J'espère que vous apprécierez cette démonstration."
    file_fr = setup_output_filename(DATA_DIR, TEST_FILE_NAME, model_fr.split("/")[-1].replace(".onnx", ""))
    synthesize_voice_and_save(model_fr, text_fr, file_fr)
    # Example for the English (GB) voice
    model_en = os.path.join(MODEL_DIR, "jarvis-medium.onnx")
    # text_en = "This is a test in British English using the Piper engine."
    text_en = " My name is Jarvis and I am delighted to meet you."
    text_en += " I am very happy to be able to speak with you today."
    text_en += " I hope you will enjoy this demonstration."
    file_en = setup_output_filename(DATA_DIR, TEST_FILE_NAME, model_en.split("/")[-1].replace(".onnx", ""))
    synthesize_voice_and_save(model_en, text_en, file_en)


if __name__ == "__main__":
    main()
