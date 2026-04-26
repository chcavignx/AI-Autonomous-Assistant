from __future__ import annotations

from pathlib import Path


def has_tts_model(config) -> bool:
    """Return True if the configured TTS model file exists.

    Returns:
        bool: True if the TTS model file exists, False otherwise.

    """
    try:
        model_path = Path(config.tts.full_model_path)
    except (AttributeError, TypeError):
        return False
    return model_path.exists()


def has_asr_model_cached(config) -> bool:
    """Return True if an ASR model appears to be cached locally.

    This avoids auto-downloading by only checking for an existing cache.

    Returns:
        bool: True if the ASR model is cached, False otherwise.

    """
    from faster_whisper.utils import download_model

    try:
        # This will raise if the model is not cached and local_files_only is True.
        download_model(
            config.asr.model_size,
            cache_dir=str(config.asr.download_path),
            local_files_only=True,
        )
        return True
    except (RuntimeError, OSError):
        return False


def get_input_device_index() -> int | None:
    """Return the first PyAudio input device index, if available.

    Returns:
        int | None: The index of the first available input device, or None.

    """
    try:
        import pyaudio
    except ImportError:
        return None

    pa = pyaudio.PyAudio()
    try:
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0):
                return int(info.get("index", index))
    finally:
        pa.terminate()
    return None


def has_input_device() -> bool:
    return get_input_device_index() is not None
