#!/usr/bin/env python3
"""Integration Test Runner: Audio Library Hardware Tests.

Runs all audio hardware integration tests in sequence with clear reporting.

Usage:
  python examples/run_all_audio_tests.py      # Run all tests
  python examples/run_all_audio_tests.py --verbose  # Detailed output
"""

import sys
import subprocess
from pathlib import Path


TESTS = [
    ("Hardware Detection", "test_hardware_detection.py"),
    ("Stream Open/Close", "test_stream_open_close.py"),
    ("Audio Playback", "test_playback.py"),
    ("Audio Recording", "test_recording.py"),
]


def run_test(test_name: str, script: str) -> bool:
    """Run a single test and return success status."""
    script_path: Path = Path(__file__).parent / script

    if not script_path.exists():
        print(f"  ✗ Script not found: {script}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Running: {test_name}")
    print('=' * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {test_name} exceeded 30 seconds")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("AUDIO LIBRARY INTEGRATION TESTS")
    print("=" * 60)
    print(f"Running {len(TESTS)} tests...\n")

    results: list[tuple[str, bool]] = []
    for test_name, script in TESTS:
        success = run_test(test_name, script)
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed: int = sum(1 for _, success in results if success)
    total: int = len(results)

    for test_name, success in results:
        status: str = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
