#! /usr/bin/env python3
"""
Extensible Voice Agent for Raspberry Pi 5
Date: 2025-09-25

Features:
- Modular architecture for easy AI integration
- Wake word detection with JARVIS
- Configurable STT engines (Whisper/Vosk)
- Piper TTS integration
- Intent recognition with extensible response system
- Clean separation of voice processing, AI logic, and TTS
"""

import datetime
import gc
import json
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Audio processing utilities
import numpy as np
import pyaudio
import torch

# STT Dependencies
# import sys; sys.path.insert(0, "/home/cca/whisper");
import whisper
from faster_whisper import WhisperModel
from sysutils import detect_raspberry_pi_model, limit_cpu_for_multiprocessing

try:
    from vosk import KaldiRecognizer
    from vosk import Model as VoskModel
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
import wave

import sounddevice as sd
import soundfile as sf

# TTS Dependencies
from piper import PiperVoice, SynthesisConfig

# VAD Dependencies
from silero_vad import get_speech_timestamps, load_silero_vad

# Suppress specific runtime warnings from faster_whisper
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor"
)
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor"
)


# ==================== CONFIGURATION CLASSES ====================
DATA_DIR = "../data/"
local_dir = ".cache"
cache_dir = os.path.join(os.path.expanduser("~"), local_dir)
vosk_cache_dir = os.path.join(cache_dir, "vosk")
default_vosk_model_path = os.path.join(vosk_cache_dir, "vosk-model-small-en-us-0.15")
defaultpiper_model_path: str = os.path.join(
    DATA_DIR, "jarvis-medium.onnx"
)  # "en_US-amy-medium.onnx"
whisper_openai_models = [
    "tiny",
    "base",
    "small",
    "medium",
]
faster_whisper_models = whisper_openai_models


class STTBackend(Enum):
    WHISPER = "whisper"
    VOSK = "vosk"
    FASTER_WHISPER = "faster_whisper"


# ==================== CONFIGURATION ====================
# Central configuration for the voice agent
# Modify these settings as needed
# Note: Ensure the Vosk model path is correct if using Vosk
# Ensure the Piper model path is correct for TTS
# Download Piper voices from https://huggingface.co/rhasspy/piper-voices
# Example: "en_US-amy-medium.onnx", "jarvis-medium.onnx"
# Ensure you have the required dependencies installed:
# pip install whisper vosk piper sounddevice soundfile silero-vad pyaudio
@dataclass
class AgentConfig:
    """Central configuration for the voice agent"""

    # Wake word settings
    wake_word: str = "jarvis"

    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024
    # Optional Parameters depending on system
    cpu_cores: Optional[int] = (
        4  # # Limit to 2 cores for Raspberry Pi, None to use all available cores
    )
    gpu: Optional[str] = None  # GPU model (if available)

    # STT settings
    stt_engine: STTBackend = STTBackend.WHISPER

    whisper_model_size: str = "base"  # tiny, base, small, medium
    whisper_download_root: str = os.path.join(os.path.expanduser("~"), ".cache/whisper")
    translate: bool = (
        False  # Set to True to translate to English, False to transcribe in original language
    )
    language: str = "en"  # if english else "fr"  # Language code for Whisper
    faster_whisper_model_size: str = "base"  # medium.en, small, large_v3
    faster_whisper_download_root: str = os.path.join(
        os.path.expanduser("~"), ".cache/huggingface"
    )
    vosk_model_path: str = default_vosk_model_path

    # TTS settings
    piper_model_path: str = defaultpiper_model_path
    synthesis_config: SynthesisConfig = field(
        default_factory=lambda: SynthesisConfig(
            volume=0.5,  # half as loud
            length_scale=1.0,  # twice as slow
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=True,  # use raw audio from voice
            speaker_id=1,  # None, # default speaker (multi-speaker voices only)
        )
    )
    # tts_to_file: bool = False  # If True, save TTS to file instead of playing directly
    # tts_to_sd: bool = False  # If True, use SoundDevice for TTS playback
    tts_to_cli: bool = (
        False  # If True, use Piper CLI for TTS synthesis else use SoundDevice for TTS playback with Python API
    )
    # VAD settings
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    silence_timeout_seconds: int = 3
    max_recording_seconds: int = 15

    # Low energy audio handling
    display_low_energy_warning: bool = True


# ==================== AUDIO UTILITIES ====================
# Audio processing utilities
# Note: These utilities help with audio validation, conversion, and playback
# They ensure robust handling of audio data to prevent issues during STT and TTS processing
# They can be extended as needed for additional audio formats or processing
# Example: Adding noise floor, normalizing audio, etc.
class AudioUtils:
    """Utility functions for audio processing"""

    @staticmethod
    def play_audio_file(file_path: str) -> None:
        """Play audio file using system audio player"""
        try:
            player = "aplay" if sys.platform != "darwin" else "afplay"
            subprocess.run([player, file_path], capture_output=True, check=False)
        except (subprocess.SubprocessError, OSError) as e:
            print(f"🔊 Audio playback error: {e}")

    @staticmethod
    def play_audio_file_to_sd(file_path: str) -> None:
        """Play audio file using SoundDevice"""
        audio_data = None
        sample_rate = None
        try:
            audio_data, sample_rate = sf.read(file_path, dtype="float32")
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
        except (sf.LibsndfileError, ValueError, sd.PortAudioError) as e:
            print(f"🔊 Audio playback error: {e}")
        finally:
            try:
                sd.stop()
            except (sd.PortAudioError, ValueError):
                pass
            # Only delete variables if they were assigned to avoid UnboundLocalError
            if "audio_data" in locals() and audio_data is not None:
                del audio_data
            if "sample_rate" in locals() and sample_rate is not None:
                del sample_rate

    @staticmethod
    def play_audio_stream(
        audio_stream: np.ndarray,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        """Play audio stream using SoundDevice"""
        stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=dtype)
        stream.start()
        try:
            for audio_bytes in audio_stream:
                stream.write(
                    audio_bytes.audio_int16_array
                )  # Assuming audio_stream yields audio chunks

        except (sd.PortAudioError, ValueError) as e:
            print(f"🔊 Audio stream playback error: {e}")
        finally:
            stream.stop()
            stream.close()
            del audio_stream

    @staticmethod
    def validate_and_clean_audio(audio_data: np.ndarray) -> np.ndarray:
        """Enhanced validation and cleaning of audio data to prevent Whisper numerical issues"""
        # Convert to numpy array if not already
        audio_array = np.array(audio_data, dtype=np.float32)

        # Return empty array if input is empty or invalid
        if audio_array.size == 0:
            return np.array([], dtype=np.float32)

        # Check for and handle invalid values (NaN, inf) - this fixes the Whisper matmul error
        has_invalid = False
        if np.any(np.isnan(audio_array)):
            print("⚠️ Detected NaN values in audio, replacing with zeros")
            audio_array = np.nan_to_num(audio_array, nan=0.0)
            has_invalid = True

        if np.any(np.isinf(audio_array)):
            print("⚠️ Detected infinite values in audio, clipping")
            audio_array = np.nan_to_num(audio_array, posinf=1.0, neginf=-1.0)
            has_invalid = True

        # Additional check for extreme values that could cause numerical issues
        extreme_threshold = 1e10
        if np.any(np.abs(audio_array) > extreme_threshold):
            print("⚠️ Detected extreme values in audio, clipping to [-1, 1]")
            audio_array = np.clip(audio_array, -1.0, 1.0)
            has_invalid = True

        # Normalize audio to prevent extreme values
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            # Normalize to [-0.95, 0.95] to prevent clipping and provide headroom
            if max_val > 0.95:  # Only normalize if needed
                audio_array = audio_array * (0.95 / max_val)
                has_invalid = True

        # Handle complete silence or near-silence
        noise_floor = 1e-8
        silence_threshold = 1e-6
        rms_energy = np.sqrt(np.mean(audio_array**2))

        if rms_energy < silence_threshold:
            # For very quiet audio, add minimal dither noise to prevent Whisper issues
            dither = np.random.normal(0, noise_floor, audio_array.shape)
            audio_array += dither
            has_invalid = True

        # Final validation: ensure no remaining invalid values
        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.95, neginf=-0.95)

        # Ensure proper data type and finite values
        audio_array = audio_array.astype(np.float32)

        if has_invalid:
            print(
                f"⚠️ Audio validation completed, RMS energy: {np.sqrt(np.mean(audio_array**2)):.6f}"
            )

        return audio_array

    @staticmethod
    def convert_to_int16(audio_float: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 with validation"""
        # First validate and clean the audio
        clean_audio = AudioUtils.validate_and_clean_audio(audio_float)
        return (clean_audio * 32767).astype(np.int16)


# ==================== VAD ENGINE ====================
# Voice Activity Detection using Silero VAD
# Note: This class wraps the Silero VAD model for detecting speech segments
# It provides methods to check for speech presence and extract speech segments
# It can be extended to support other VAD models or configurations as needed
# Example: Adding energy thresholding, multi-channel support, etc.
class VADEngine:
    """Voice Activity Detection using Silero VAD"""

    def __init__(self, config: AgentConfig):
        """Initialize VAD engine with configuration"""
        self.config = config
        self.model = load_silero_vad()

        if detect_raspberry_pi_model():
            limit_cpu_for_multiprocessing(self.config.cpu_cores)
            if self.config.whisper_model_size in whisper_openai_models:
                self.config.whisper_model_size = f"{self.config.whisper_model_size}{'.en' if self.config.language == 'en' else ''}"  # tiny # Recommended model for low resources
            if self.config.faster_whisper_model_size in faster_whisper_models:
                self.config.faster_whisper_model_size = f"{self.config.faster_whisper_model_size}{'.en' if self.config.language == 'en' else ''}"  # tiny # Recommended model for low resources
        else:
            limit_cpu_for_multiprocessing()  # Use all available cores
            # self.config.whisper_model_size = "base" # "large-v3" # "medium" # "small" # "large-v3" # "medium" # "small" # "base" # "tiny" # Recommended model for low resources

    # audio_data: np.ndarray
    def is_speech_detected(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains speech"""
        if len(audio_data) < self.config.sample_rate // 4:  # Need at least 250ms
            return False
        try:
            audio_tensor = torch.FloatTensor(audio_data)
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
            )
            return bool(timestamps)
        except (RuntimeError, ValueError) as e:
            print(f"❌ VAD error: {e}")
            return False

    def get_speech_segments(self, audio_data: np.ndarray) -> List[Dict]:
        """Get detailed speech segments with timestamps"""
        try:
            audio_tensor = torch.FloatTensor(audio_data)
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                return_seconds=True,
            )
            return timestamps
        except (RuntimeError, ValueError) as e:
            print(f"❌ VAD segmentation error: {e}")
            return []


# ==================== STT ENGINE ====================
# Speech-to-Text processing with multiple backend support
# Note: This class abstracts the STT processing using either Whisper or Vosk
# It provides methods to transcribe audio data to text
# It can be extended to support additional STT engines or configurations as needed
# Example: Adding custom language models, noise robustness, etc.
class STTEngine:
    """Speech-to-Text processing with multiple backend support"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected STT model"""
        try:
            if self.config.stt_engine == STTBackend.WHISPER:
                print(f"🧠 Loading Whisper {self.config.whisper_model_size} model...")
                self.model = whisper.load_model(
                    self.config.whisper_model_size,
                    device="cpu",
                    download_root=self.config.whisper_download_root,
                )
                print("✅ Whisper model loaded")

            elif self.config.stt_engine == STTBackend.VOSK:
                if VoskModel is None:
                    raise ImportError("Vosk not installed. pip install vosk")
                print(f"🧠 Loading Vosk model from {self.config.vosk_model_path}...")
                self.model = VoskModel(self.config.vosk_model_path)
                print("✅ Vosk model loaded")

            elif self.config.stt_engine == STTBackend.FASTER_WHISPER:
                print(
                    f"🧠 Loading Faster Whisper {self.config.faster_whisper_model_size} model..."
                )
                self.model = WhisperModel(
                    self.config.faster_whisper_model_size,
                    device="cpu",
                    compute_type="int8",  # Use int8 for lower memory usage, float16 if you have a compatible GPU
                    num_workers=4,
                    download_root=self.config.faster_whisper_download_root,
                )
                print("✅ Faster Whisper model loaded")

        except Exception as e:
            print(f"❌ STT model initialization error: {e}")
            raise

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text"""
        try:
            if self.config.stt_engine in (
                STTBackend.WHISPER,
                STTBackend.FASTER_WHISPER,
            ):
                return self._transcribe_whisper(audio_data)
            elif self.config.stt_engine == STTBackend.VOSK:
                return self._transcribe_vosk(audio_data)
            else:
                raise ValueError(f"Invalid STT engine: {self.config.stt_engine}")
        except (RuntimeError, ValueError) as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def _transcribe_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using Whisper with improved error handling and validation"""
        try:
            # Early validation: check for completely invalid input
            if audio_data.size == 0:
                return ""

            # Validate and clean audio data before processing
            clean_audio = AudioUtils.validate_and_clean_audio(audio_data)

            # Skip processing if validation resulted in empty or very short audio
            if (
                clean_audio.size == 0 or len(clean_audio) < 1600
            ):  # Less than 0.1 seconds at 16kHz
                return ""

            # Check if audio has sufficient energy
            rms_energy = np.sqrt(np.mean(clean_audio**2))
            if rms_energy < 1e-4:  # Very quiet audio
                return ""

            # Additional validation: ensure audio contains finite values only
            if not np.all(np.isfinite(clean_audio)):
                print("⚠️ Audio contains non-finite values, re-cleaning")
                clean_audio = AudioUtils.validate_and_clean_audio(clean_audio)

            # Transcribe with error handling and support for different Whisper versions
            if self.config.stt_engine == STTBackend.FASTER_WHISPER:
                segments, info = self.model.transcribe(
                    clean_audio,
                    multilingual=False,
                    language=self.config.language,
                    beam_size=1,  # Fast inference
                    best_of=1,  # Faster inference
                    condition_on_previous_text=False,
                    vad_filter=False,  # We already did VAD
                    suppress_blank=True,  # Suppress blank outputs
                    no_speech_threshold=0.6,  # Higher threshold for cleaner results
                    word_timestamps=True,
                    temperature=0.0,  # Deterministic output
                )
                result = segments  # segments is a list of Segment objects
            else:
                # Standard Whisper
                result = self.model.transcribe(
                    clean_audio,
                    word_timestamps=True,
                    fp16=False,
                    language=self.config.language,
                    task="translate" if self.config.translate else "transcribe",
                )
            # Extract text from result
            # Handle different return formats from different Whisper versions
            if isinstance(result, dict):
                # Standard whisper returns a dict with 'segments' key
                segments = result.get("segments", [])
            elif isinstance(result, tuple):
                # Some versions return (segments, info) tuple
                segments = result[0] if len(result) > 0 else []
            else:
                # Assume it's directly the segments or result object
                segments = (
                    getattr(result, "segments", result)
                    if hasattr(result, "segments")
                    else result
                )

            # Debug: log result type for troubleshooting
            # print(f"🔍 Debug: result type: {type(result)}, segments type: {type(segments)}")

            # Extract and clean text
            text_segments = []
            if segments:
                for segment in segments:
                    # Handle different segment formats
                    text = (
                        segment.get("text", "")
                        if isinstance(segment, dict)
                        else (
                            getattr(segment, "text", "")
                            if hasattr(segment, "text")
                            else segment if isinstance(segment, str) else ""
                        )
                    )

                    if text and text.strip():
                        text_segments.append(text.strip())

            result = " ".join(text_segments).strip()
            return result

        except (RuntimeError, ValueError) as e:
            print(f"⚠️ Whisper transcription error: {e}")
            return ""

    def _transcribe_vosk(self, audio_data: np.ndarray) -> str:
        """Transcribe using Vosk"""
        audio_int16 = AudioUtils.convert_to_int16(audio_data)
        recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
        recognizer.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(recognizer.Result())
        return result.get("text", "").strip()


# ==================== TTS ENGINE ====================
# Text-to-Speech using Piper
# Note: This class abstracts the TTS processing using Piper
# It provides methods to convert text to speech and play or save it
# It can be extended to support additional TTS engines or configurations as needed
# Example: Adding SSML support, multi-speaker voices, etc.
class TTSEngine:
    """Text-to-Speech using Piper"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._validate_piper_model()

    def _validate_piper_model(self):
        """Validate Piper model exists"""
        if not os.path.exists(self.config.piper_model_path):
            print(f"⚠️ Piper model not found: {self.config.piper_model_path}")
            print("Download from: https://huggingface.co/rhasspy/piper-voices")
        # Create a Piper object
        self.voice = PiperVoice.load(self.config.piper_model_path)

    def _synthesize_voice_to_wav(self, text, output_file):
        """Synthesizes text to audio, saves it and plays."""
        if not self.voice:
            print("⚠️ Piper voice not initialized.")
            return False
        if self.config.tts_to_cli:
            # Use Piper CLI for synthesis
            try:
                cmd = [
                    "piper",
                    "--model",
                    self.config.piper_model_path,
                    "--text",
                    text,
                    "--output_file",
                    output_file,
                ]
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"❌ Piper CLI error: {stderr.decode()}")
                    success = False
                success = True
            except (subprocess.SubprocessError, IOError) as e:
                print(f"❌ Piper CLI error: {e}")
                success = False
        else:
            # Direct synthesis using Piper library
            with wave.open(output_file, "wb") as wav_file:
                self.voice.synthesize_wav(
                    text=text,
                    wav_file=wav_file,
                    set_wav_format=True,
                    syn_config=self.config.synthesis_config,
                )
                print(f"Audio file saved to: {output_file}")
            success = True
        return success

    def _synthesize_voice(self, text):
        """Synthesizes text to audio and plays it."""
        if not self.voice:
            print("⚠️ Piper voice not initialized.")
            return
        return self.voice.synthesize(text)

    def speak(self, text: str) -> bool:
        """Convert text to speech and play it"""
        if not text.strip():
            return False

        print(f"🔊 Speaking: {text}")
        if self.config.tts_to_cli:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    self._synthesize_voice_to_wav(text, tmp_file.name)
                    AudioUtils.play_audio_file(tmp_file.name)
                    os.unlink(tmp_file.name)
                    return True
            except (subprocess.SubprocessError, IOError) as e:
                print(f"❌ TTS to file error: {e}")
                return False
        else:
            try:
                audio_stream = self._synthesize_voice(text)
                AudioUtils.play_audio_stream(
                    audio_stream, sample_rate=self.voice.config.sample_rate
                )
                return True
            except (sd.PortAudioError, IOError, ValueError) as e:
                print(f"❌ TTS error: {e}")
                return False


# ==================== AI RESPONSE ENGINE ====================
# Extensible AI response engine
# Note: This class is designed for easy integration with various AI models
# It currently uses simple intent matching but can be replaced with advanced AI models
# Example: OpenAI API, local LLMs, HuggingFace transformers, etc.
class AIResponseEngine:
    """
    Extensible AI response engine - replace this with your AI model
    Currently uses simple intent matching, but designed for easy AI integration
    """

    def __init__(self, config: AgentConfig = None):
        # Allow optional config injection; create default if not provided
        self.config = config if config is not None else AgentConfig()

        # Precompute dynamic time/date strings to keep lines short
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        date_str = datetime.datetime.now().strftime("%d %B %Y")

        # Simple intent-response mapping (replace with AI model)
        self.intents = {
            "hello": (
                f"Hello! I'm {self.config.wake_word}, your AI assistant. "
                "How can I help you?"
            ),
            "hi": "Hi there! What can I do for you?",
            "weather": (
                "I'm running offline and don't have access to weather data right now."
            ),
            "time": f"The current time is {time_str}",
            "date": f"Today, the date is: {date_str}",
            "lights": ("I would control your lights if I had smart home integration."),
            "music": "I would play music if I had access to your media system.",
            "stop": "Goodbye! Returning to wake word detection.",
            "exit": "See you later! Going back to sleep mode.",
            "help": (
                "I can respond to simple commands like hello, time, weather, "
                "lights, music, and stop."
            ),
        }

    def generate_response(self, user_input: str) -> Tuple[str, bool]:
        """
        Generate AI response to user input

        Args:
            user_input: Transcribed user speech

        Returns:
            Tuple of (response_text, should_continue_listening)
        """
        user_input = user_input.lower().strip()
        if not user_input:
            return "I didn't catch that. Could you please repeat?", True
        print(f"🤖 User input: '{user_input}'")
        # Intent matching (replace with your AI model)
        response = self._match_intent(user_input)
        # Check for exit commands
        exit_words = ["stop", "exit", "quit"]
        should_continue = not any(word in user_input for word in exit_words)
        if not should_continue:
            print("🟢 Exiting command mode, returning to wake word detection...")
        return response, should_continue

    def _match_intent(self, user_input: str) -> str:
        """Simple intent matching - replace with AI model inference"""
        # Check for exact matches first
        for intent, response in self.intents.items():
            if intent in user_input:
                return response

        # Fallback response
        return (
            f"I heard you say '{user_input}'. "
            "I'm still learning how to respond to that."
        )

    # TODO: Replace this method with your AI model integration
    def generate_ai_response(self, user_input: str, context: Dict = None) -> str:
        """
        EXTENSION POINT: Integrate your AI model here

        Examples:
        - OpenAI API calls
        - Local LLM inference (Llama, Mistral, etc.)
        - HuggingFace transformers
        - Custom trained models

        Args:
            user_input: User's spoken text
            context: Optional conversation context

        Returns:
            AI generated response text
        """
        # Placeholder for AI integration
        # return openai_client.chat.completions.create(...)
        # return local_llm.generate(user_input, context)
        # return huggingface_model(user_input)

        return self.generate_response(user_input)


# ==================== MAIN VOICE AGENT ====================
# Main voice agent orchestrating all components
# Note: This class integrates VAD, STT, TTS, and AI response engines
# It manages the main loop for listening, processing, and responding to voice commands
# It can be extended with additional features such as logging, multi-threading, etc.
# Example: Adding command history, user profiles, etc.


class VoiceAgent:
    """Main voice agent orchestrating all components"""

    def __init__(self, config: AgentConfig):
        self.config = config
        # Initialize components
        print("🚀 Initializing Voice Agent...")
        self.vad_engine = VADEngine(config)
        self.stt_engine = STTEngine(config)
        self.tts_engine = TTSEngine(config)
        self.ai_engine = AIResponseEngine(self.config)

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # State management
        self.wake_word_detected = False
        self.is_listening = True

        print("✅ Voice Agent initialized!")

    def start(self):
        """Start the voice agent"""
        print("🎤 Voice Agent starting...")
        print(f"🎯 Wake word: '{self.config.wake_word}'")
        print(f"🧠 STT Engine: {self.config.stt_engine.value}")
        print(f"🔊 TTS Model: {self.config.piper_model_path}")

        self._setup_audio_stream()
        self._main_loop()

    def stop(self):
        """Stop the voice agent"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🛑 Voice Agent stopped")

    def _setup_audio_stream(self):
        """Setup audio input stream with device detection"""
        # Find a suitable input device
        device_index = None
        try:
            device_count = self.audio.get_device_count()
            print(f"🎤 Found {device_count} audio devices")

            # Look for devices with input channels
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                max_input_channels = device_info.get("maxInputChannels", 0)
                device_name = device_info.get("name", "")

                if max_input_channels > 0:
                    # Prefer PulseAudio or default devices on Linux
                    name_l = device_name.lower()
                    if "pulse" in name_l or "default" in name_l:
                        device_index = i
                        print(
                            "🎤 Using audio device %d: %s (channels: %d)"
                            % (i, device_name, max_input_channels)
                        )
                        break
                    elif device_index is None:
                        # Use first available input device as fallback
                        device_index = i
                        print(
                            "🎤 Fallback audio device %d: %s (channels: %d)"
                            % (i, device_name, max_input_channels)
                        )

        except (OSError, ValueError) as e:
            print(f"⚠️ Error detecting audio devices: {e}")
            device_index = None

        # Try to open audio stream
        try:
            # Use detected device or None for default
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                input_device_index=device_index,
            )
            if device_index is not None:
                print(f"✅ Audio stream opened successfully with device {device_index}")
            else:
                print("✅ Audio stream opened successfully with default device")
        except (OSError, ValueError) as e:
            print(f"❌ Failed to open audio stream: {e}")
            if device_index is not None:
                print("🔄 Retrying with system default device...")
                try:
                    self.stream = self.audio.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.config.sample_rate,
                        input=True,
                        frames_per_buffer=self.config.chunk_size,
                        input_device_index=None,  # Use system default
                    )
                    print("✅ Audio stream opened successfully with system default")
                except (OSError, ValueError) as e2:
                    print(f"❌ Failed to open audio stream with system default: {e2}")
                    raise
            else:
                raise

    def _main_loop(self):
        """Main processing loop"""
        audio_buffer = []
        print("🟢 Ready! Say '{}' to activate...".format(self.config.wake_word.upper()))

        try:
            while self.is_listening:
                '''"👂 Listening for new audio chunk..."'''
                # Read audio chunk
                chunk_size = self.config.chunk_size
                data = self.stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                audio_buffer.extend(audio_chunk)

                # Process every second of audio
                if len(audio_buffer) >= self.config.sample_rate:
                    # print("🔊 Processing audio buffer...")
                    self._process_audio_buffer(audio_buffer)
                    audio_buffer.clear()
                    self.stream.get_read_available()  # Clear any remaining buffer
                    # Force cleanup
                    # print("🧹 Cleaning up memory...")
                    gc.collect()

        except KeyboardInterrupt:
            print("\n⏹️ Stopping agent...")
        finally:
            self.stop()

    def _process_audio_buffer(self, audio_buffer: List[float]):
        """Process accumulated audio buffer with validation"""
        try:
            # Skip processing if validation resulted in empty or very short audio
            if not audio_buffer or len(audio_buffer) < 1600:
                print("⚠️ Skip: short or empty audio buffer")
                return

            # Convert to numpy array with initial validation
            raw_array = np.array(audio_buffer, dtype=np.float32)

            # # Check for completely invalid audio data early
            # if raw_array is completely invalid: skip processing
            # (original: size==0 or all NaN or all Inf)

            # Validate and clean audio buffer
            audio_array = AudioUtils.validate_and_clean_audio(raw_array)

            # Additional energy check to avoid processing pure noise/silence
            rms_energy = np.sqrt(np.mean(audio_array**2))
            if rms_energy < 1e-5:  # Very low energy threshold
                if not self.config.display_low_energy_warning:
                    return
                print("⚠️ Skip: very low energy audio")
                return

            # Check for speech activity
            if self.vad_engine.is_speech_detected(audio_array):
                transcript = self.stt_engine.transcribe(audio_array)

                if transcript and len(transcript.strip()) > 0:
                    print(f"📝 Transcribed: '{transcript}'")
                    self._handle_transcript(transcript)
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ Audio buffer processing error: {e}")

    def _handle_transcript(self, transcript: str):
        """Handle transcribed speech"""
        transcript_lower = transcript.lower()

        if not self.wake_word_detected:
            # Check for wake word
            if self.config.wake_word in transcript_lower:
                self.wake_word_detected = True
                print(f"🟣 Wake word '{self.config.wake_word}' detected!")
                self.tts_engine.speak("Yes? How can I help you?")
        else:
            # Process command
            self._process_voice_command(transcript)

    def _process_voice_command(self, command: str):
        """Process voice command after wake word detection"""
        print(f"🤖 Processing command: '{command}'")
        # self.wake_word_detected = False  # Reset wake word state
        # Generate AI response
        response_text = self._generate_ai_response(command)
        if not response_text:
            response_text = "I'm sorry, I didn't catch that. Could you please repeat?"
        # Execute command (speak response)
        self._execute_command(response_text, command)

    def _generate_ai_response(self, user_input: str) -> str:
        """Generate response using AI engine"""
        response, should_continue = self.ai_engine.generate_response(user_input)
        if not should_continue:
            self.wake_word_detected = False
            print("🟢 Returning to wake word detection...")

        return response

    def _execute_command(self, response: str, original_command: str):
        """Execute the command by speaking the response"""
        success = self.speak_response(response)

        if success:
            print(f"✅ Command executed: '{original_command}'")
        else:
            print(f"❌ Command execution failed: '{original_command}'")

    def speak_response(self, text: str) -> bool:
        """Speak the response text"""
        return self.tts_engine.speak(text)


# ==================== ENTRY POINT ====================
# Main entry point
# Note: This section initializes the configuration and starts the voice agent
# Modify the configuration as needed before running
# Example: Changing wake word, STT engine, TTS model, etc.


def main():
    """Main entry point"""
    # Configuration
    config = AgentConfig(
        wake_word="thanos",  # Change to your desired wake word
        stt_engine=STTBackend.FASTER_WHISPER,  # or STTBackend.WHISPER
        vosk_model_path=os.path.join(vosk_cache_dir, "vosk-model-small-en-us-0.15"),
        language="en",
        # tiny for speed, base for accuracy
        whisper_model_size="tiny",
        # use 'medium.en' for better accuracy, 'base' for speed
        faster_whisper_model_size="tiny",
        piper_model_path=os.path.join(DATA_DIR, "jarvis-medium.onnx"),
        tts_to_cli=True,
    )

    # Create and start agent
    agent = VoiceAgent(config)

    try:
        agent.start()
    except Exception as e:
        print(f"❌ Agent error: {e}")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
