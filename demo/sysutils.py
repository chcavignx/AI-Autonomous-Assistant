#!/usr/bin/env python3
import os
import time

import psutil


# Utility function to display CPU/RAM usage
def print_sys_usage(step: str):
    cpu = psutil.cpu_percent(interval=0.5)  # % CPU
    ram_used = psutil.virtual_memory().used / (1024**3)  # in GB
    ram_total = psutil.virtual_memory().total / (1024**3)
    print(f"[{step}] CPU: {cpu:.1f}% | RAM: {ram_used:.2f} GB / {ram_total:.2f} GB")


# Utility function to print elapsed time for each operation
def print_time_usage(step: str, start_time: float):
    elapsed = time.time() - start_time
    print(f"[{step}] Elapsed time: {elapsed:.2f} seconds")


def detect_cpu_count():
    cpu_count = os.cpu_count()
    print(f"Detected CPU count: {cpu_count if cpu_count else 'unknown'}")
    return cpu_count or 1


def limit_cpu_for_multiprocessing(desired_cores=None):
    # On some platforms, you can also use multiprocessing or
    # set num_threads in whisper (if available)
    cpu_count = detect_cpu_count()
    n_cores = desired_cores if desired_cores else cpu_count
    n_cores = max(1, min(n_cores, cpu_count))
    print(f"Limiting usage to {n_cores} cores.")
    return n_cores


def detect_raspberry_pi_model():
    # Read hardware model from procfs (Linux/Raspberry Pi)
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            model = f.read().strip()
        print(f"Detected model: {model}")
        if "Raspberry Pi 5" in model:
            print("Raspberry Pi 5 detected!")
            return True
    except OSError:
        pass
    print("Raspberry Pi 5 not detected.")
    return False
