"""Hailo device architecture detection utilities."""

from __future__ import annotations

import os

HAILO_ARCH_KEY = "HAILO_ARCH"


def detect_hailo_arch() -> str | None:
    """Detect connected Hailo architecture (e.g. 'hailo8', 'hailo8l')."""
    env_arch = os.getenv(HAILO_ARCH_KEY)
    if env_arch:
        return env_arch.strip().lower()

    try:
        from hailo_platform import Device  # type: ignore[import-not-found]

        devices = Device.scan()
        if devices:
            return "hailo8l" if "8l" in str(devices[0]).lower() else "hailo8"
    except Exception:
        pass
    return None
