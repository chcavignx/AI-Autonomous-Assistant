#!/usr/bin/env python3
"""Master vision benchmark — all backends, all metrics, structured output.

Face detectors : cascade | insightface | hailo | imx500
Object engines : yolo_cpu | libreyolo_cpu | yolo_ncnn | yolo_hailo | yolo_imx500

Outputs (in results/):
  bench_vision_<TIMESTAMP>.csv  — raw per-detector row
  bench_vision_<TIMESTAMP>.md   — Markdown table + ASCII bar charts

Usage examples:
  python bench_vision_all.py                                 # all face detectors, 10 s each
  python bench_vision_all.py --face cascade,insightface --duration 5
  python bench_vision_all.py --face all --object yolo_cpu,yolo_ncnn --duration 8
  python bench_vision_all.py --image /path/to/face.jpg      # static image mode (reproducible)
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from time import perf_counter

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.metrics import ThroughputMeter
from src.utils.sysutils import get_cpu_temperature_c, get_cpu_usage_percent
from src.vision.face_detector import CascadeFaceDetector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FACE_DETECTORS = ["cascade", "insightface", "hailo", "imx500"]
OBJECT_ENGINES = ["yolo_cpu", "libreyolo_cpu", "yolo_ncnn", "yolo_hailo", "yolo_imx500"]
RESULTS_DIR = Path("results")

# ─────────────────────────────────────────────────────────────────────────────
# Frame source
# ─────────────────────────────────────────────────────────────────────────────


class FrameSource:
    """Unified source: camera or static image or synthetic fallback."""

    def __init__(self, image_path: str | None = None, camera_index: int = 0) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._static: np.ndarray | None = None

        if image_path and Path(image_path).exists():
            img = cv2.imread(image_path)
            if img is not None:
                self._static = img
                print(f"[FrameSource] Static image: {image_path}")
                return

        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            self._cap = cap
            print(f"[FrameSource] Camera #{camera_index}")
        else:
            cap.release()
            self._static = _synthetic_frame()
            print("[FrameSource] Synthetic fallback frame (no camera).")

    def read(self) -> np.ndarray:
        if self._static is not None:
            return self._static.copy()
        assert self._cap is not None
        ret, frame = self._cap.read()
        return frame if ret else _synthetic_frame()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def _synthetic_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a minimal synthetic BGR frame with a face-like placeholder."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy), 80, (200, 200, 200), -1)
    cv2.circle(img, (cx - 30, cy - 20), 10, (0, 0, 0), -1)
    cv2.circle(img, (cx + 30, cy - 20), 10, (0, 0, 0), -1)
    cv2.ellipse(img, (cx, cy + 30), (30, 15), 0, 0, 180, (0, 0, 255), 3)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Face detector runners
# ─────────────────────────────────────────────────────────────────────────────


def _detect_face_status(proc: Any, requested_backend: str) -> str:
    """Inspect proc to determine whether the requested backend actually ran or fell back."""
    if requested_backend == "cascade":
        return "OK"

    active_det = getattr(proc, "detector", None)
    if active_det is None and hasattr(proc, "face_recognizer"):
        active_det = getattr(proc.face_recognizer, "detector", None)

    if requested_backend == "hailo":
        if hasattr(active_det, "use_hailo") and not active_det.use_hailo:
            return "FALLBACK (cascade)"
        if isinstance(active_det, CascadeFaceDetector):
            return "FALLBACK (cascade)"
    elif requested_backend == "insightface":
        if isinstance(active_det, CascadeFaceDetector):
            return "FALLBACK (cascade)"
    elif requested_backend == "imx500":
        if isinstance(active_det, CascadeFaceDetector) or getattr(active_det, "imx500", None) is None:
            return "FALLBACK (cascade)"

    return "OK"


def _run_face_benchmark(
    detector_type: str,
    source: FrameSource,
    duration_sec: float,
) -> dict:
    """Benchmark a single face detector backend."""
    from src.vision.face_insight_frame import FaceInsightFrame

    cfg = load_config()
    try:
        proc = FaceInsightFrame(cfg=cfg, detector_type=detector_type)
    except Exception as e:
        print(f"  ✗ [{detector_type}] init failed: {e}")
        return _null_result()

    status = _detect_face_status(proc, detector_type)
    if status != "OK":
        print(f"  ⚠ [{detector_type}] hardware/model unavailable — active mode is {status}")

    # Warmup (5 frames)
    for _ in range(5):
        try:
            proc.process_frame(source.read(), draw=False)
        except Exception:
            pass

    meter = ThroughputMeter()
    latencies: list[float] = []
    cpus: list[float] = []
    temps: list[float] = []

    t_end = perf_counter() + duration_sec
    while perf_counter() < t_end:
        frame = source.read()
        t0 = perf_counter()
        try:
            proc.process_frame(frame, draw=False)
        except Exception:
            pass
        latencies.append((perf_counter() - t0) * 1000.0)
        meter.tick()
        cpus.append(get_cpu_usage_percent())
        temps.append(get_cpu_temperature_c() or 0.0)
        if source._static is not None:
            time.sleep(0.005)

    _stop_detector(proc)
    return _aggregate(meter, latencies, cpus, temps, status=status)


# ─────────────────────────────────────────────────────────────────────────────
# Object engine runners
# ─────────────────────────────────────────────────────────────────────────────


def _detect_object_status(detector: Any, requested_engine: str) -> str:
    """Inspect detector to determine whether requested object engine actually ran or fell back."""
    if requested_engine == "yolo_hailo":
        if hasattr(detector, "use_hailo") and not detector.use_hailo:
            return "FALLBACK (cpu)"
    elif requested_engine == "yolo_imx500":
        if getattr(detector, "imx500", None) is None:
            return "FALLBACK (stub)"
    return "OK"


def _run_object_benchmark(
    engine: str,
    source: FrameSource,
    duration_sec: float,
) -> dict:
    """Benchmark a single object detection engine backend."""
    detector = None
    cfg = load_config()
    try:
        if engine == "yolo_cpu":
            from src.vision.yolo_cpu import YoloCpuDetector

            detector = YoloCpuDetector(cfg)
        elif engine == "libreyolo_cpu":
            from src.vision.yolo_cpu import YoloCpuDetector

            model_cand = cfg.paths.models_vision_path / "yolo" / "LibreYOLOXn.onnx"
            model_p = str(model_cand) if model_cand.exists() else "LibreYOLOXn.onnx"
            detector = YoloCpuDetector(cfg, model_path=model_p)
        elif engine == "yolo_ncnn":
            from src.vision.yolo_cpu import Yolo26NcnnDetector

            detector = Yolo26NcnnDetector(cfg)
        elif engine == "yolo_hailo":
            from src.vision.yolo_hailo import YoloHailoDetector

            detector = YoloHailoDetector(cfg)
        elif engine == "yolo_imx500":
            from src.vision.yolo_imx500 import Imx500Config, Imx500Detector

            imx_cfg = Imx500Config()
            detector = Imx500Detector(imx_cfg)
        else:
            print(f"  ✗ Unknown engine: {engine}")
            return _null_result()
    except Exception as e:
        print(f"  ✗ [{engine}] init failed: {e}")
        return _null_result()

    status = _detect_object_status(detector, engine)
    if status != "OK":
        print(f"  ⚠ [{engine}] hardware/runtime unavailable — active mode is {status}")

    for _ in range(5):
        try:
            detector.detect(source.read())
        except Exception:
            pass

    meter = ThroughputMeter()
    latencies: list[float] = []
    cpus: list[float] = []
    temps: list[float] = []

    t_end = perf_counter() + duration_sec
    while perf_counter() < t_end:
        frame = source.read()
        t0 = perf_counter()
        try:
            detector.detect(frame)
        except Exception:
            pass
        latencies.append((perf_counter() - t0) * 1000.0)
        meter.tick()
        cpus.append(get_cpu_usage_percent())
        temps.append(get_cpu_temperature_c() or 0.0)
        if source._static is not None:
            time.sleep(0.005)

    if hasattr(detector, "stop"):
        try:
            detector.stop()
        except Exception:
            pass

    return _aggregate(meter, latencies, cpus, temps, status=status)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _null_result() -> dict:
    return {
        "fps": 0.0,
        "lat_mean_ms": 0.0,
        "lat_p50_ms": 0.0,
        "lat_p95_ms": 0.0,
        "cpu_mean": 0.0,
        "temp_mean": 0.0,
        "status": "FAILED",
    }


def _aggregate(
    meter: ThroughputMeter,
    latencies: list[float],
    cpus: list[float],
    temps: list[float],
    status: str = "OK",
) -> dict:
    lat = np.array(latencies, dtype=np.float32) if latencies else np.array([0.0], dtype=np.float32)
    return {
        "fps": meter.fps(),
        "lat_mean_ms": float(np.mean(lat)),
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p95_ms": float(np.percentile(lat, 95)),
        "cpu_mean": float(np.mean(cpus)) if cpus else 0.0,
        "temp_mean": float(np.mean(temps)) if temps else 0.0,
        "status": status,
    }


def _stop_detector(proc: Any) -> None:
    if hasattr(proc, "stop"):
        try:
            proc.stop()
        except Exception:
            pass
    elif hasattr(proc, "detector") and proc.detector is not None:
        if hasattr(proc.detector, "stop"):
            try:
                proc.detector.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "category",
    "backend",
    "fps",
    "lat_mean_ms",
    "lat_p50_ms",
    "lat_p95_ms",
    "cpu_mean",
    "temp_mean",
    "status",
]


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  → {path}")


def _ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return " " * width
    filled = int(round(min(value / max_value, 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def _fps_bar_section(rows: list[dict], width: int = 30) -> str:
    lines = ["### FPS (higher is better)\n", "```"]
    active_rows = [r for r in rows if r["status"] != "FAILED"]
    max_fps = max((r["fps"] for r in active_rows), default=1.0)
    for r in rows:
        is_ok = r["status"] == "OK"
        is_fallback = "FALLBACK" in r["status"]
        if is_ok or is_fallback:
            bar = _ascii_bar(r["fps"], max_fps, width)
            fps_str = f"{r['fps']:6.1f}"
            suffix = " (FB)" if is_fallback else ""
            lines.append(f"{r['backend']:<20} {bar} {fps_str} FPS{suffix}")
        else:
            lines.append(f"{r['backend']:<20} {' ' * width}   N/A  FPS")
    lines.append("```\n")
    return "\n".join(lines)


def _latency_bar_section(rows: list[dict], width: int = 30) -> str:
    lines = ["### Mean Latency ms (lower is better)\n", "```"]
    active_rows = [r for r in rows if r["status"] != "FAILED"]
    max_lat = max((r["lat_mean_ms"] for r in active_rows), default=1.0)
    for r in rows:
        is_ok = r["status"] == "OK"
        is_fallback = "FALLBACK" in r["status"]
        if is_ok or is_fallback:
            bar = _ascii_bar(r["lat_mean_ms"], max_lat, width)
            lat_str = f"{r['lat_mean_ms']:8.1f} ms"
            suffix = " (FB)" if is_fallback else ""
            lines.append(f"{r['backend']:<20} {bar} {lat_str}{suffix}")
        else:
            lines.append(f"{r['backend']:<20} {' ' * width}    N/A")
    lines.append("```\n")
    return "\n".join(lines)


def save_markdown(rows: list[dict], path: Path, duration: float, ts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    face_rows = [r for r in rows if r["category"] == "face"]
    obj_rows = [r for r in rows if r["category"] == "object"]

    def table(section_rows: list[dict]) -> str:
        lines = [
            "| Backend | FPS | Lat mean (ms) | Lat p50 (ms) | Lat p95 (ms) | CPU% | Temp °C | Status |",
            "|---------|-----|---------------|--------------|--------------|------|---------|--------|",
        ]
        for r in section_rows:
            has_data = r["status"] != "FAILED"
            fps = f"{r['fps']:.1f}" if has_data else "—"
            lm = f"{r['lat_mean_ms']:.1f}" if has_data else "—"
            lp = f"{r['lat_p50_ms']:.1f}" if has_data else "—"
            l95 = f"{r['lat_p95_ms']:.1f}" if has_data else "—"
            cpu = f"{r['cpu_mean']:.1f}" if has_data else "—"
            tmp = f"{r['temp_mean']:.1f}" if has_data else "—"
            lines.append(f"| {r['backend']} | {fps} | {lm} | {lp} | {l95} | {cpu} | {tmp} | {r['status']} |")
        return "\n".join(lines)

    md_lines = [
        "# Vision Benchmark Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Duration per backend:** {duration:.0f} s  ",
        "",
        "---",
        "",
        "## Face Detection Backends",
        "",
        table(face_rows),
        "",
        _fps_bar_section(face_rows),
        _latency_bar_section(face_rows),
    ]

    if obj_rows:
        md_lines += [
            "## Object Detection Engines",
            "",
            table(obj_rows),
            "",
            _fps_bar_section(obj_rows),
            _latency_bar_section(obj_rows),
        ]

    md_lines += [
        "---",
        "",
        "> Generated by `bench_vision_all.py`",
    ]

    with path.open("w") as f:
        f.write("\n".join(md_lines))
    print(f"MD   → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Master vision benchmark — face & object detection, all backends.")
    parser.add_argument(
        "--face",
        default="all",
        help=f"Face detectors to benchmark (comma-separated, 'all', or 'none'). "
        + f"Choices: {', '.join(FACE_DETECTORS)}. Default: all.",
    )
    parser.add_argument(
        "--object",
        default="none",
        help=f"Object engines to benchmark (comma-separated or 'none' or 'all'). "
        + f"Choices: {', '.join(OBJECT_ENGINES)}. Default: none.",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration in seconds per backend (default: 10)."
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a static input image (reproducible/offline mode)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    args = parser.parse_args()

    face_list = [] if args.face == "none" else (
        FACE_DETECTORS if args.face == "all" else [f.strip() for f in args.face.split(",")]
    )
    obj_list = [] if args.object == "none" else (
        OBJECT_ENGINES if args.object == "all" else [e.strip() for e in args.object.split(",")]
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"bench_vision_{ts}"

    source = FrameSource(image_path=args.image, camera_index=args.camera)

    rows: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"  VISION BENCHMARK  —  {ts}")
    print(f"  Face: {face_list or 'none'}")
    print(f"  Object: {obj_list or 'none'}")
    print(f"  Duration/backend: {args.duration} s")
    print(f"{'=' * 60}\n")

    # Face detectors
    for det in face_list:
        print(f"[face] {det}...")
        res = _run_face_benchmark(det, source, args.duration)
        rows.append({"category": "face", "backend": det, **res})
        if res["status"] != "FAILED":
            flag = f" [{res['status']}]" if res["status"] != "OK" else ""
            print(
                f"  ✓  FPS={res['fps']:.1f}  lat={res['lat_mean_ms']:.1f} ms  "
                + f"p95={res['lat_p95_ms']:.1f} ms  cpu={res['cpu_mean']:.1f}%  "
                + f"T={res['temp_mean']:.1f}°C{flag}"
            )
        else:
            print("  ✗  skipped")

    # Object engines
    for eng in obj_list:
        print(f"\n[object] {eng}...")
        res = _run_object_benchmark(eng, source, args.duration)
        rows.append({"category": "object", "backend": eng, **res})
        if res["status"] != "FAILED":
            flag = f" [{res['status']}]" if res["status"] != "OK" else ""
            print(
                f"  ✓  FPS={res['fps']:.1f}  lat={res['lat_mean_ms']:.1f} ms  "
                + f"p95={res['lat_p95_ms']:.1f} ms  cpu={res['cpu_mean']:.1f}%  "
                + f"T={res['temp_mean']:.1f}°C{flag}"
            )
        else:
            print("  ✗  skipped")

    source.release()

    # Write outputs
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    save_csv(rows, RESULTS_DIR / f"{stem}.csv")
    save_markdown(rows, RESULTS_DIR / f"{stem}.md", args.duration, ts)

    # Print summary table to stdout
    print()
    print(f"{'Backend':<20} {'FPS':>6} {'Lat mean':>10} {'Lat p95':>9} {'CPU%':>6} {'Status':<18}")
    print("-" * 72)
    for r in rows:
        tag = f"[{r['category']}]"
        if r["status"] != "FAILED":
            print(
                f"{r['backend']:<20} {r['fps']:>6.1f} {r['lat_mean_ms']:>8.1f} ms "
                + f"{r['lat_p95_ms']:>7.1f} ms {r['cpu_mean']:>5.1f}%  {r['status']} {tag}"
            )
        else:
            print(f"{r['backend']:<20} {'N/A':>6} {'N/A':>10} {'N/A':>9} {'N/A':>6}  FAILED {tag}")
    print()


if __name__ == "__main__":
    main()
