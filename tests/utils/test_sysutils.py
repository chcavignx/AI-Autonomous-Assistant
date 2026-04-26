"""Unit tests for src/utils/sysutils.py."""

import pathlib
import sys
from dataclasses import dataclass

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from src.utils import sysutils


@dataclass
class DummyVmem:
    """Minimal virtual-memory object used to stub psutil."""

    used: int
    total: int


def test_print_sys_usage_calls_psutil_with_expected_access_pattern(monkeypatch):
    calls = []

    class DummyPsutil:
        @staticmethod
        def cpu_percent(interval=0.5):
            calls.append(("cpu_percent", interval))
            return 12.3456

        @staticmethod
        def virtual_memory():
            calls.append(("virtual_memory",))
            return DummyVmem(used=int(1.5 * 1024**3), total=8 * 1024**3)

    monkeypatch.setattr(sysutils, "psutil", DummyPsutil)

    result = sysutils.print_sys_usage("STEP-A")

    assert result is None
    assert calls == [
        ("cpu_percent", 0.5),
        ("virtual_memory",),
    ]


def test_print_time_usage_computes_elapsed_time(monkeypatch):
    monkeypatch.setattr(sysutils.time, "time", lambda: 101.234)

    result = sysutils.print_time_usage("LOAD", start_time=100.0)

    assert result is None


def test_detect_cpu_count_when_available(monkeypatch):
    monkeypatch.setattr(sysutils.os, "cpu_count", lambda: 6)

    assert sysutils.detect_cpu_count() == 6


def test_detect_cpu_count_when_none(monkeypatch):
    monkeypatch.setattr(sysutils.os, "cpu_count", lambda: None)

    assert sysutils.detect_cpu_count() == 1


def test_limit_cpu_for_multiprocessing_default_to_detected(monkeypatch):
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 4)

    assert sysutils.limit_cpu_for_multiprocessing() == 4


def test_limit_cpu_for_multiprocessing_caps_to_cpu_count(monkeypatch):
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 8)

    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=16) == 8


def test_limit_cpu_for_multiprocessing_floors_to_one(monkeypatch):
    monkeypatch.setattr(sysutils, "detect_cpu_count", lambda: 8)

    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=0) == 8
    assert sysutils.limit_cpu_for_multiprocessing(desired_cores=-3) == 1


def test_detect_raspberry_pi_model_true(monkeypatch):
    def fake_read_text(self, encoding=None):
        assert self == pathlib.Path("/proc/device-tree/model")
        assert encoding == "utf-8"
        return "Raspberry Pi 5 Model B\n"

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is True


def test_detect_raspberry_pi_model_other(monkeypatch):
    def fake_read_text(self, encoding=None):
        assert self == pathlib.Path("/proc/device-tree/model")
        assert encoding == "utf-8"
        return "Raspberry Pi 4 Model B\n"

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is False


def test_detect_raspberry_pi_model_oserror(monkeypatch):
    def fake_read_text(self, encoding=None):
        raise OSError("no such file")

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)

    assert sysutils.detect_raspberry_pi_model() is False
