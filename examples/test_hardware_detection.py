#!/usr/bin/env python3
"""Integration test: Audio Hardware Detection.

Simple first test to validate audio hardware is accessible.
Useful for deployment verification and CI/CD health checks.

Run with:
  python examples/test_hardware_detection.py
"""

import sys
from pathlib import Path

# Ensure repo root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.audio_utils import get_audio_backend, list_audio_devices


def test_audio_hardware_available() -> bool | None:
    """Check if any audio hardware (input or output) is available."""
    backend = get_audio_backend()

    try:
        devices = list_audio_devices(backend)
        device_count = len(devices)

        if device_count == 0:
            return False

        # List all devices
        input_devices = [d for d in devices if d.is_input]
        output_devices = [d for d in devices if d.is_output]

        for _dev in input_devices:
            pass

        for _dev in output_devices:
            pass

        has_input: bool = len(input_devices) > 0
        has_output: bool = len(output_devices) > 0

        return has_input or has_output

    except Exception:
        return False


if __name__ == "__main__":
    success = test_audio_hardware_available()
    sys.exit(0 if success else 1)
