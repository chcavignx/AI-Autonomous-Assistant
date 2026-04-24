"""Utility functions for system monitoring and resource management."""

import logging
import os
import pathlib
import time
from typing import TYPE_CHECKING, cast

import psutil

if TYPE_CHECKING:
    from psutil._ntuples import svmem

logger = logging.getLogger(name=__name__)


# Utility function to display CPU/RAM usage
def print_sys_usage(_step: str) -> None:
    """Print system usage statistics."""
    cpu: float = psutil.cpu_percent(interval=0.5)
    vm: svmem = psutil.virtual_memory()
    used_gb = cast("float", vm.used) / 1024**3
    total_gb = cast("float", vm.total) / 1024**3
    logger.info("%s: CPU=%.1f%% RAM=%.2f/%.2f GB", _step, cpu, used_gb, total_gb)


# Utility function to print elapsed time for each operation
def print_time_usage(_step: str, start_time: float) -> None:
    """Print elapsed time for each operation."""
    elapsed: float = time.time() - start_time
    logger.info("%s: %.3fs", _step, elapsed)


def detect_cpu_count() -> int:
    """Detect the number of available CPU cores.

    Returns:
        The detected CPU core count, or `1` when detection fails.

    """
    cpu_count: int | None = os.cpu_count()
    return cpu_count or 1


def limit_cpu_for_multiprocessing(desired_cores: int | None = None) -> int:
    """Limit the number of CPU cores used for multiprocessing.

    Returns:
        A core count clamped between `1` and the detected CPU count.

    """
    # On some platforms, you can also use multiprocessing or
    # set num_threads in whisper (if available)
    cpu_count: int = detect_cpu_count()
    n_cores: int = desired_cores or cpu_count
    return max(1, min(n_cores, cpu_count))


def detect_raspberry_pi_model() -> bool:
    """Detect if the system is running on a Raspberry Pi 5.

    Returns:
        `True` when the procfs hardware model contains ``Raspberry Pi 5``.

    """
    # Read hardware model from procfs (Linux/Raspberry Pi)
    try:
        model: str = pathlib.Path("/proc/device-tree/model").read_text(encoding="utf-8").strip()
        if "Raspberry Pi 5" in model:
            return True
    except OSError:
        pass
    return False
