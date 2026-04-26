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

import pyaudio


def test_audio_hardware_available():
    """Check if any audio hardware (input or output) is available."""
    print("\n" + "=" * 60)
    print("TEST: Audio Hardware Detection")
    print("=" * 60)

    pa = pyaudio.PyAudio()
    try:
        device_count = pa.get_device_count()
        print(f"✓ Total audio devices detected: {device_count}")

        if device_count == 0:
            print("✗ FAILED: No audio devices found")
            return False

        # List all devices
        input_devices: list[tuple[int, str, int]] = []
        output_devices: list[tuple[int, str, int]] = []

        for idx in range(device_count):
            info = pa.get_device_info_by_index(idx)
            max_input = info.get("maxInputChannels", 0)
            max_output = info.get("maxOutputChannels", 0)
            name = info.get("name", "Unknown")

            if max_input and max_input > 0:
                input_devices.append((idx, name, max_input))
            if max_output and max_output > 0:
                output_devices.append((idx, name, max_output))

        print(f"\nInput devices ({len(input_devices)}):")
        for idx, name, channels in input_devices:
            print(f"  [{idx}] {name} ({channels} channels)")

        print(f"\nOutput devices ({len(output_devices)}):")
        for idx, name, channels in output_devices:
            print(f"  [{idx}] {name} ({channels} channels)")

        has_input: bool = len(input_devices) > 0
        has_output: bool = len(output_devices) > 0

        print(f"\n✓ Input capability: {'YES' if has_input else 'NO'}")
        print(f"✓ Output capability: {'YES' if has_output else 'NO'}")

        if not (has_input or has_output):
            print("\n✗ FAILED: No input or output devices available")
            return False

        print("\n✓ PASSED: Audio hardware detected")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    finally:
        pa.terminate()


if __name__ == "__main__":
    success = test_audio_hardware_available()
    sys.exit(0 if success else 1)
