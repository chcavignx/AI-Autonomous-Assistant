#!/usr/bin/env python3
"""Integration Test Runner: Audio Library Hardware Tests.

Runs all audio hardware integration tests in sequence with clear reporting.

Usage:
  python examples/audio/run_all_audio_tests.py      # Run all tests
  python examples/audio/run_all_audio_tests.py --verbose  # Detailed output
"""

import subprocess
import sys
from pathlib import Path

TESTS = [
    ("Hardware Detection", "test_hardware_detection.py"),
    ("Stream Open/Close", "test_stream_open_close.py"),
    ("Audio Playback", "test_playback.py"),
    ("Audio Recording", "test_recording.py"),
    ("Recorder Standalone", "test_recorder_standalone.py"),
    ("ASR with TTS", "test_asr_with_tts.py"),
    ("ASR Recording Validation", "test_asr_recording_validation.py"),
    ("Wake Word Standalone", "test_wake_word_standalone.py"),
    ("VAD Standalone Flow", "test_vad_standalone.py"),
    ("TTS Lifecycle & Utils", "test_tts_lifecycle_and_utils.py"),
]


def run_test(test_name: str, script: str) -> bool:
    """Run a single test and return success status."""
    script_path: Path = Path(__file__).parent / script

    if not script_path.exists():
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent.parent.parent,
            timeout=60,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def main() -> int:
    """Run all integration tests."""
    print("==================================================")
    print("Running Audio Library Hardware Integration Tests")
    print("==================================================")
    results: list[tuple[str, bool]] = []
    for test_name, script in TESTS:
        print(f"Running {test_name} ({script})... ", end="", flush=True)
        success = run_test(test_name, script)
        if success:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
        results.append((test_name, success))

    # Summary
    print("==================================================")
    print("INTEGRATION TESTS SUMMARY")
    print("==================================================")
    passed: int = sum(1 for _, success in results if success)
    total: int = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  - {test_name}: {status}")
    print("--------------------------------------------------")
    print(f"Overall: {passed}/{total} tests passed")
    print("==================================================")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
