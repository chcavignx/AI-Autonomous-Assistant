#!/usr/bin/env python3
"""Benchmark face detectors, object detectors, and simultaneous vision combinations.

Backends:
  Face   : cascade, insightface, hailo, imx500
  Object : yolo_cpu, libreyolo_cpu, yolo_ncnn, yolo_hailo, yolo_imx500

Outputs results and summary CSV to results/bench_combinations.csv.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.metrics import ThroughputMeter
from src.utils.sysutils import get_cpu_temperature_c, get_cpu_usage_percent
from src.vision.face_detector import CascadeFaceDetector
from src.vision.face_insight_frame import FaceInsightFrame
from src.vision.object_insight_frame import ObjectInsightFrame

DEFAULT_FACE_DETECTORS = ["cascade", "insightface", "hailo", "imx500"]
DEFAULT_OBJECT_ENGINES = ["yolo_cpu", "libreyolo_cpu", "yolo_ncnn", "yolo_hailo", "yolo_imx500"]


def _load_or_create_frame(
    image_path: str | None, camera_index: int = 0
) -> tuple[np.ndarray | None, cv2.VideoCapture | None]:
    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
        if img is not None:
            return img, None

    sample = project_root / "data" / "sample_face.jpg"
    if sample.exists():
        img = cv2.imread(str(sample))
        if img is not None:
            return img, None

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return None, cap

    cap.release()
    # Synthetic frame fallback
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 80, (200, 200, 200), -1)
    cv2.circle(img, (290, 220), 10, (0, 0, 0), -1)
    cv2.circle(img, (350, 220), 10, (0, 0, 0), -1)
    cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (0, 0, 255), 3)
    return img, None


def _init_object_detector(engine: str, cfg: Any) -> Any:
    if engine == "yolo_cpu":
        from src.vision.yolo_cpu import YoloCpuDetector

        return YoloCpuDetector(cfg)
    if engine == "libreyolo_cpu":
        from src.vision.yolo_cpu import YoloCpuDetector

        model_cand = cfg.paths.models_vision_path / "yolo" / "LibreYOLOXn.onnx"
        model_p = str(model_cand) if model_cand.exists() else "LibreYOLOXn.onnx"
        return YoloCpuDetector(cfg, model_path=model_p)
    if engine == "yolo_ncnn":
        from src.vision.yolo_cpu import Yolo26NcnnDetector

        return Yolo26NcnnDetector(cfg)
    if engine == "yolo_hailo":
        from src.vision.yolo_hailo import YoloHailoDetector

        return YoloHailoDetector(cfg)
    if engine == "yolo_imx500":
        from src.vision.yolo_imx500 import Imx500Config, Imx500Detector

        return Imx500Detector(Imx500Config())
    return None


def benchmark_pipeline(
    face_type: str | None,
    object_engine: str | None,
    duration_sec: float = 5.0,
    static_frame: np.ndarray | None = None,
    cap: cv2.VideoCapture | None = None,
) -> dict:
    """Benchmark a single face detector, object detector, or simultaneous combination."""
    label_parts = []
    if face_type:
        label_parts.append(f"Face: {face_type}")
    if object_engine:
        label_parts.append(f"Object: {object_engine}")
    label = " + ".join(label_parts) if label_parts else "None"

    print(f"\n{'=' * 58}")
    print(f"  Benchmark: {label}")
    print(f"{'=' * 58}")

    cfg = load_config()

    face_proc: FaceInsightFrame | None = None
    obj_proc: ObjectInsightFrame | None = None
    status = "OK"

    if face_type:
        cfg.vision.face_detector_type = face_type
        try:
            face_proc = FaceInsightFrame(cfg=cfg, detector_type=face_type)
            active_det = getattr(face_proc, "detector", None)
            if active_det is None and hasattr(face_proc, "face_recognizer"):
                active_det = getattr(face_proc.face_recognizer, "detector", None)
            if face_type == "hailo" and (
                getattr(active_det, "use_hailo", True) is False or isinstance(active_det, CascadeFaceDetector)
            ):
                status = "FALLBACK"
            elif face_type in {"insightface", "imx500"} and isinstance(active_det, CascadeFaceDetector):
                status = "FALLBACK"
        except Exception as e:
            print(f"[ERROR] Face init failed for '{face_type}': {e}")
            return {
                "label": label,
                "fps": 0.0,
                "latency_ms": 0.0,
                "lat_p95_ms": 0.0,
                "cpu_percent": 0.0,
                "temp_c": 0.0,
                "status": "FAILED",
            }

    if object_engine:
        try:
            raw_detector = _init_object_detector(object_engine, cfg)
            if raw_detector is None:
                raise ValueError(f"Unknown object engine: {object_engine}")
            obj_proc = ObjectInsightFrame(cfg=cfg, detector=raw_detector)
            if object_engine == "yolo_hailo" and getattr(raw_detector, "use_hailo", True) is False:
                status = "FALLBACK"
            elif object_engine == "yolo_imx500" and getattr(raw_detector, "imx500", None) is None:
                status = "FALLBACK"
        except Exception as e:
            print(f"[ERROR] Object init failed for '{object_engine}': {e}")
            if face_proc:
                face_proc.stop()
            return {
                "label": label,
                "fps": 0.0,
                "latency_ms": 0.0,
                "lat_p95_ms": 0.0,
                "cpu_percent": 0.0,
                "temp_c": 0.0,
                "status": "FAILED",
            }

    # Warmup
    for _ in range(5):
        if static_frame is not None:
            f = static_frame
        elif cap is not None:
            ret, f_cap = cap.read()
            f = f_cap if ret else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            f = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            curr = f
            if face_proc:
                curr, _ = face_proc.process_frame(curr, draw=False)
            if obj_proc:
                curr, _, _ = obj_proc.process_frame(curr, draw=False)
        except Exception:
            pass

    meter = ThroughputMeter()
    latencies: list[float] = []
    cpus: list[float] = []
    temps: list[float] = []

    start = time.time()
    while time.time() - start < duration_sec:
        if static_frame is not None:
            frame = static_frame
        elif cap is not None:
            ret, f_cap = cap.read()
            frame = f_cap if ret else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        try:
            curr = frame
            if face_proc:
                curr, _ = face_proc.process_frame(curr, draw=False)
            if obj_proc:
                curr, _, _ = obj_proc.process_frame(curr, draw=False)
        except Exception as e:
            print(f"[WARNING] process_frame failed: {e}")
        latencies.append((time.perf_counter() - t0) * 1000.0)
        meter.tick()
        cpus.append(get_cpu_usage_percent())
        temps.append(get_cpu_temperature_c() or 0.0)
        if static_frame is not None:
            time.sleep(0.005)

    if face_proc:
        face_proc.stop()
    if obj_proc:
        obj_proc.stop()

    result = {
        "label": label,
        "fps": meter.fps(),
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "lat_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "cpu_percent": float(np.mean(cpus)) if cpus else 0.0,
        "temp_c": float(np.mean(temps)) if temps else 0.0,
        "status": status,
    }
    print(
        f"  FPS={result['fps']:.1f}  Lat={result['latency_ms']:.1f} ms  "
        + f"p95={result['lat_p95_ms']:.1f} ms  CPU={result['cpu_percent']:.1f}%  "
        + f"Temp={result['temp_c']:.1f}°C  [{status}]"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face, object, and combined vision configurations.")
    parser.add_argument(
        "--detectors",
        default=",".join(DEFAULT_FACE_DETECTORS),
        help=f"Face detectors to benchmark (choices: {', '.join(DEFAULT_FACE_DETECTORS)}, or 'none').",
    )
    parser.add_argument(
        "--object",
        default="none",
        help=f"Object engines to benchmark (choices: {', '.join(DEFAULT_OBJECT_ENGINES)}, 'none', or 'all').",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per combination (default: 5.0).")
    parser.add_argument("--image", type=str, default=None, help="Path to static image (optional).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    args = parser.parse_args()

    face_list = [] if args.detectors == "none" else [d.strip() for d in args.detectors.split(",") if d.strip()]
    obj_list = [] if args.object == "none" else (
        DEFAULT_OBJECT_ENGINES if args.object == "all" else [o.strip() for o in args.object.split(",") if o.strip()]
    )

    static_frame, cap = _load_or_create_frame(args.image, args.camera)
    all_results: list[dict] = []

    try:
        if face_list and obj_list:
            # Benchmark simultaneous combinations
            for f_det in face_list:
                for o_eng in obj_list:
                    res = benchmark_pipeline(
                        f_det,
                        o_eng,
                        duration_sec=args.duration,
                        static_frame=static_frame,
                        cap=cap,
                    )
                    all_results.append(res)
        elif face_list:
            for f_det in face_list:
                res = benchmark_pipeline(
                    f_det,
                    None,
                    duration_sec=args.duration,
                    static_frame=static_frame,
                    cap=cap,
                )
                all_results.append(res)
        elif obj_list:
            for o_eng in obj_list:
                res = benchmark_pipeline(
                    None,
                    o_eng,
                    duration_sec=args.duration,
                    static_frame=static_frame,
                    cap=cap,
                )
                all_results.append(res)
    finally:
        if cap is not None:
            cap.release()

    # Save summary CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "bench_combinations.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["configuration", "fps", "latency_ms", "lat_p95_ms", "cpu_percent", "temp_c", "status"])
        for m in all_results:
            writer.writerow([
                m["label"],
                f"{m['fps']:.2f}",
                f"{m['latency_ms']:.2f}",
                f"{m['lat_p95_ms']:.2f}",
                f"{m['cpu_percent']:.2f}",
                f"{m['temp_c']:.2f}",
                m["status"],
            ])
    print(f"\nCombinations CSV → {csv_path}")


if __name__ == "__main__":
    main()
