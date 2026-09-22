#!/usr/bin/env python3
"""Compare Phase 1 (CPU) vs Phase 2 (IMX500) benchmark results from CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_bench_csv(path: Path) -> dict[str, np.ndarray]:
    """Load benchmark metrics from CSV supporting both per-frame and aggregate column schemas."""
    latencies: list[float] = []
    cpus: list[float] = []
    temps: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat_val = row.get("latency_ms") or row.get("lat_mean_ms") or row.get("latency") or 0.0
            cpu_val = row.get("cpu_percent") or row.get("cpu_mean") or row.get("cpu") or 0.0
            temp_val = row.get("temp_c") or row.get("temp_mean") or row.get("temp") or 0.0
            latencies.append(float(lat_val))
            cpus.append(float(cpu_val))
            temps.append(float(temp_val))

    return {
        "latency_ms": np.array(latencies, dtype=np.float32) if latencies else np.array([0.0], dtype=np.float32),
        "cpu_percent": np.array(cpus, dtype=np.float32) if cpus else np.array([0.0], dtype=np.float32),
        "temp_c": np.array(temps, dtype=np.float32) if temps else np.array([0.0], dtype=np.float32),
    }


def summarize(name: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
    lat = data["latency_ms"]
    cpu = data["cpu_percent"]
    temp = data["temp_c"]
    mean_lat = float(np.mean(lat))
    return {
        "name": name,
        "lat_mean": mean_lat,
        "lat_median": float(np.median(lat)),
        "lat_p95": float(np.percentile(lat, 95)),
        "fps_mean": 1000.0 / mean_lat if mean_lat > 0 else 0.0,
        "cpu_mean": float(np.mean(cpu)),
        "temp_mean": float(np.mean(temp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 1 vs Phase 2 benchmark CSV results.")
    parser.add_argument(
        "--phase1",
        type=str,
        default="results/bench_face_phase1_cpu.csv",
        help="Path to Phase 1 CSV file.",
    )
    parser.add_argument(
        "--phase2",
        type=str,
        default="results/bench_face_phase2_imx500.csv",
        help="Path to Phase 2 CSV file.",
    )
    args = parser.parse_args()

    phase1_path = Path(args.phase1)
    phase2_path = Path(args.phase2)

    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 CSV not found: {phase1_path}. Run bench_face_phase1_cpu.py first.")
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 CSV not found: {phase2_path}. Run bench_face_phase2_imx500.py first.")

    d1 = load_bench_csv(phase1_path)
    d2 = load_bench_csv(phase2_path)

    s1 = summarize("Phase 1 – InsightFace CPU", d1)
    s2 = summarize("Phase 2 – IMX500 + ArcFace", d2)

    header = "| Pipeline | FPS | Lat mean (ms) | Lat median (ms) | Lat p95 (ms) | CPU% | Temp °C |"
    sep = "|----------|-----|---------------|-----------------|--------------|------|---------|"
    print(header)
    print(sep)
    for s in (s1, s2):
        print(
            f"| {s['name']:<27} "
            + f"| {s['fps_mean']:>5.1f} "
            + f"| {s['lat_mean']:>13.1f} "
            + f"| {s['lat_median']:>15.1f} "
            + f"| {s['lat_p95']:>12.1f} "
            + f"| {s['cpu_mean']:>4.1f} "
            + f"| {s['temp_mean']:>7.1f} |"
        )


if __name__ == "__main__":
    main()
