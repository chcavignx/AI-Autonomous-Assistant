#!/usr/bin/env python3
"""Benchmark: InsightFace CPU pipeline (Phase 1 — full CPU face recognition)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter, sleep

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.metrics import CsvLogger
from src.utils.sysutils import get_cpu_temperature_c, get_cpu_usage_percent
from src.vision.face_insight_pipeline import FaceInsightPipeline


def _synthetic_face_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Create synthetic frame with a face-like structure."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy), 80, (200, 200, 200), -1)
    cv2.circle(img, (cx - 30, cy - 20), 10, (0, 0, 0), -1)
    cv2.circle(img, (cx + 30, cy - 20), 10, (0, 0, 0), -1)
    cv2.ellipse(img, (cx, cy + 30), (30, 15), 0, 0, 180, (0, 0, 255), 3)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 (CPU InsightFace) benchmark.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of benchmark iterations (default: 200).")
    parser.add_argument("--image", type=str, default=None, help="Path to static image (optional).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--mock", action="store_true", help="Force synthetic frame mock mode.")
    args = parser.parse_args()

    static_frame: np.ndarray | None = None
    cap: cv2.VideoCapture | None = None

    if args.image and Path(args.image).exists():
        static_frame = cv2.imread(args.image)
    elif args.mock:
        static_frame = _synthetic_face_frame()
    else:
        sample_path = project_root / "data" / "sample_face.jpg"
        if sample_path.exists():
            static_frame = cv2.imread(str(sample_path))
        else:
            c = cv2.VideoCapture(args.camera)
            if c.isOpened():
                cap = c
            else:
                c.release()
                print("[INFO] Camera not available — using synthetic placeholder frame.")
                static_frame = _synthetic_face_frame()

    face = FaceInsightPipeline(det_size=(640, 640), detector_type="insightface")

    print("Phase 1: capturing/registering reference face...")
    ref_img = None
    if static_frame is not None:
        ref_img = static_frame.copy()
        try:
            face.register_face("user", ref_img)
        except Exception as e:
            print(f"[INFO] Direct registration fallback: {e}")
            # Register dummy embedding so recognition runs
            face.recognizer.known_faces["user"] = np.ones(512, dtype=np.float32) / np.sqrt(512)
    elif cap is not None:
        attempts = 0
        while ref_img is None and attempts < 100:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                sleep(0.01)
                continue
            try:
                face.register_face("user", frame)
                ref_img = frame.copy()
            except RuntimeError:
                continue
        if ref_img is None:
            print("[WARNING] Could not detect face from camera; using direct placeholder embedding.")
            face.recognizer.known_faces["user"] = np.ones(512, dtype=np.float32) / np.sqrt(512)

    print(f"Reference registered. Running Phase 1 benchmark ({args.iterations} iterations)...")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "bench_face_phase1_cpu.csv"
    logger = CsvLogger(csv_path, headers=["latency_ms", "cpu_percent", "temp_c"])

    start_bench = perf_counter()

    for i in range(args.iterations):
        if static_frame is not None:
            frame = static_frame
        elif cap is not None:
            ret, f_cap = cap.read()
            frame = f_cap if ret else _synthetic_face_frame()
        else:
            frame = _synthetic_face_frame()

        t0 = perf_counter()
        _ = face.recognize(frame, thresh=0.4)
        t1 = perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        cpu = get_cpu_usage_percent()
        temp = get_cpu_temperature_c() or 0.0

        logger.log(latency_ms, cpu, temp)

        if (i + 1) % max(1, args.iterations // 10) == 0:
            elapsed = perf_counter() - start_bench
            print(
                f"[{i+1}/{args.iterations}] latency={latency_ms:.1f} ms CPU={cpu:.1f}% T={temp:.1f}°C "
                + f"(elapsed={elapsed:.1f}s)"
            )

    logger.close()
    face.stop()
    if cap is not None:
        cap.release()
    print(f"Phase 1 benchmark done → {csv_path}")


if __name__ == "__main__":
    main()
