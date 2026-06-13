#!/usr/bin/env python3
"""List audio devices visible to the local runtime.

Helpful on Raspberry Pi OS Bookworm where PipeWire is often active and ALSA,
PortAudio/PyAudio, and Pulse/PipeWire can expose slightly different views.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _print_subprocess_output(label: str, cmd: list[str]) -> None:
    code, stdout, stderr = _run(cmd)
    print(f"\n[{label}]")
    if stdout:
        print(stdout)
    if stderr:
        print(f"(stderr): {stderr}")
    if not stdout and not stderr:
        print("  (no output)")


def _list_sounddevice_devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices: list[dict[str, Any]] = []
    for index, info in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": info["name"],
                "host_api": info["hostapi"],
                "max_input_channels": info["max_input_channels"],
                "max_output_channels": info["max_output_channels"],
                "default_sample_rate": info["default_samplerate"],
            }
        )

    return devices


def _print_sounddevice_devices() -> None:
    _print_section("SoundDevice")
    devices = _list_sounddevice_devices()
    if not devices:
        print("  sounddevice not available or no devices found.")
        return

    inputs = [d for d in devices if d["max_input_channels"]]
    outputs = [d for d in devices if d["max_output_channels"]]

    for device in devices:
        role_parts = []
        if device["max_input_channels"]:
            role_parts.append("input")
        if device["max_output_channels"]:
            role_parts.append("output")
        role = "/".join(role_parts) if role_parts else "none"
        print(
            f"  [{device['index']:2d}] {device['name']!r:40s}"
            f"  {role:14s}"
            f"  in={device['max_input_channels']} out={device['max_output_channels']}"
            f"  {device['default_sample_rate']:.0f} Hz"
        )

    print(f"\n  Input devices  : {len(inputs)}")
    print(f"  Output devices : {len(outputs)}")


def _print_system_audio() -> None:
    _print_section("ALSA")
    for tool in ("arecord", "aplay"):
        path = shutil.which(tool)
        if not path:
            print(f"  {tool}: not found")
            continue
        _print_subprocess_output(f"{tool} -l", [tool, "-l"])


def _print_pipewire_audio() -> None:
    _print_section("PipeWire / Pulse")
    commands = [
        ["pw-cli", "ls", "Node"],
        ["wpctl", "status"],
        ["pactl", "list", "short", "sources"],
        ["pactl", "list", "short", "sinks"],
    ]
    for cmd in commands:
        if not shutil.which(cmd[0]):
            print(f"  {cmd[0]}: not found")
            continue
        _print_subprocess_output(" ".join(cmd), cmd)


def _package_installed(name: str) -> bool:
    code, _, _ = _run(["dpkg-query", "-W", "-f=${Status}", name])
    return code == 0


def _read_text(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _print_diagnostics() -> None:
    _print_section("Diagnostics")

    for pkg in ("pipewire-alsa", "pipewire-pulse", "pulseaudio"):
        installed = _package_installed(pkg)
        print(f"  {pkg}: {'installed' if installed else 'not installed'}")

    asound_conf = _read_text("/etc/asound.conf")
    if asound_conf is None:
        print("\n  /etc/asound.conf: not found")
    else:
        print(f"\n  /etc/asound.conf: {' '.join(asound_conf.split())}")


def _print_config_hint() -> None:
    _print_section("Config Hint")
    print(
        "  To select a specific device, set 'input_device_name' or\n"
        "  'input_device_index' in config.yaml under the 'audio:' section.\n"
        "  Example:\n"
        "    audio:\n"
        "      input_device_name: 'USB ENC Audio Device'"
    )


def main() -> int:
    _print_sounddevice_devices()
    _print_system_audio()
    _print_pipewire_audio()
    _print_diagnostics()
    _print_config_hint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
