"""Unit tests for src/utils/sysutils.py."""

import os
import pathlib
import sys
import time
from dataclasses import dataclass

import pytest

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils import sysutils

pytestmark = pytest.mark.basic


@dataclass
class DummyVmem:
    """Minimal virtual-memory object used to stub psutil."""

    used: int
    total: int


def test_print_sys_usage_calls_psutil_with_expected_access_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | float, ...]] = []

    class DummyPsutil:
        @staticmethod
        def cpu_percent(interval: float = 0.5) -> float:
            calls.append(("cpu_percent", interval))
            return 12.3456

        @staticmethod
        def virtual_memory() -> DummyVmem:
            calls.append(("virtual_memory",))
            return DummyVmem(used=int(1.5 * 1024**3), total=8 * 1024**3)

    monkeypatch.setattr(sysutils, "psutil", DummyPsutil)

    result = sysutils.print_sys_usage("STEP-A")

    assert result is None
    assert calls == [
        ("cpu_percent", 0.5),
        ("virtual_memory",),
    ]


def test_print_time_usage_computes_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the real `time` module — sysutils holds a reference to the same object.
    monkeypatch.setattr(time, "time", lambda: 101.234)

    result = sysutils.print_time_usage("LOAD", start_time=100.0)

    assert result is None


def test_detect_cpu_count_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert sysutils.detect_cpu_count() == 6


def test_detect_cpu_count_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert sysutils.detect_cpu_count() == 1


def test_limit_cpu_for_multiprocessing_default_to_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 4)

    assert sysutils.limit_cpu_for_multiprocessing() == 4


def test_limit_cpu_for_multiprocessing_caps_to_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 8)

    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=16) == 8


def test_limit_cpu_for_multiprocessing_floors_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 8)

    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=0) == 8
    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=-3) == 1


def test_detect_raspberry_pi_model_true(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_text(self: pathlib.Path, encoding: str | None = None) -> str:
        assert self == pathlib.Path("/proc/device-tree/model")
        assert encoding == "utf-8"
        return "Raspberry Pi 5 Model B\n"

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is True


def test_detect_raspberry_pi_model_other(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_text(self: pathlib.Path, encoding: str | None = None) -> str:
        assert self == pathlib.Path("/proc/device-tree/model")
        assert encoding == "utf-8"
        return "Raspberry Pi 4 Model B\n"

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is False


def test_detect_raspberry_pi_model_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_text(_self: pathlib.Path, encoding: str | None = None) -> str:
        raise OSError("no such file")

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is False


def test_get_cpu_usage_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class DummyPsutil:
        @staticmethod
        def cpu_percent(interval: float = 0.0) -> float:
            calls.append(interval)
            return 45.6

    monkeypatch.setattr(sysutils, "psutil", DummyPsutil)
    assert sysutils.get_cpu_usage_percent(interval=0.1) == 45.6
    assert calls == [0.1]


def test_get_cpu_temperature_c_thermal_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_text(self: pathlib.Path, encoding: str | None = None) -> str:
        assert self == pathlib.Path("/sys/class/thermal/thermal_zone0/temp")
        return "48123\n"

    def fake_exists(self: pathlib.Path) -> bool:
        return True

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.get_cpu_temperature_c() == 48.123
