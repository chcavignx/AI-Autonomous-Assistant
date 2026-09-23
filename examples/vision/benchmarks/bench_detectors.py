#!/usr/bin/env python3
"""Benchmark face and object detection backends on static/synthetic frames.

Compares latency, FPS, and detected count across modular vision detectors.
Uses a static image (or synthetic placeholder) for deterministic comparison.
"""

from __future__ import annotations

import argparse
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
from src.vision.base import BaseDetector
from src.vision.face_detector import CascadeFaceDetector, InsightFaceDetector
from src.vision.face_detector_hailo import HailoFaceDetector
from src.vision.face_detector_imx import Imx500Detector as Imx500FaceDetector
from src.vision.yolo_cpu import Yolo26NcnnDetector, YoloCpuDetector
from src.vision.yolo_hailo import YoloHailoDetector
from src.vision.yolo_imx500 import Imx500Config, Imx500Detector as Imx500ObjectDetector


def load_or_create_image(project_root: Path, image_path: str | None = None) -> np.ndarray:
    """Load image from path, project samples, or create a synthetic placeholder."""
    candidate_paths = []
    if image_path:
        candidate_paths.append(Path(image_path))
    candidate_paths.extend([
        project_root / "data" / "captures" / "input_photo" / "sample.jpg",
        project_root / "data" / "sample_face.jpg",
        project_root / "data" / "bus.jpg",
    ])

    for p in candidate_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                print(f"[INFO] Loaded: {p}")
                return img

    print("[INFO] No sample image found — using synthetic placeholder.")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 80, (200, 200, 200), -1)
    cv2.circle(img, (290, 220), 10, (0, 0, 0), -1)
    cv2.circle(img, (350, 220), 10, (0, 0, 0), -1)
    cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (0, 0, 255), 3)
    return img


def _run_bench_suite(
    detectors: dict[str, tuple[Any, str]],
    img: np.ndarray,
    iterations: int,
    category_label: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    print(f"\nRunning {iterations} iterations per {category_label.lower()} detector...\n")

    for name, (factory, _expected_type) in detectors.items():
        det: BaseDetector | None = None
        status = "OK"
        try:
            det = factory()
            if hasattr(det, "use_hailo") and not det.use_hailo:
                status = "FALLBACK"
            elif hasattr(det, "imx500") and det.imx500 is None:
                status = "FALLBACK"
            print(f"  ✓ {name} initialized ({status})")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            det = None
            status = "FAILED"

        if det is None:
            results[name] = {"avg_ms": float("inf"), "fps": 0.0, "found": 0, "status": status}
            continue

        try:
            det.detect(img)  # warmup
        except Exception:
            results[name] = {"avg_ms": float("inf"), "fps": 0.0, "found": 0, "status": "ERROR"}
            continue

        t0 = time.perf_counter()
        found = 0
        for _ in range(iterations):
            dets = det.detect(img)
            found = len(dets)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / iterations) * 1000.0
        fps = iterations / elapsed if elapsed > 0 else 0.0
        results[name] = {"avg_ms": avg_ms, "fps": fps, "found": found, "status": status}
        print(f"    {name:<22}: {avg_ms:6.1f} ms/frame  ({fps:5.1f} FPS)  dets={found}  [{status}]")

        if hasattr(det, "stop"):
            try:
                det.stop()
            except Exception:
                pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face and object detector modules.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations (default: 50).")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for benchmark.")
    parser.add_argument(
        "--mode",
        choices=["all", "face", "object"],
        default="all",
        help="Categories to benchmark: all, face, or object.",
    )
    args = parser.parse_args()

    cfg = load_config()
    img = load_or_create_image(project_root, args.image)

    face_detectors: dict[str, tuple[Any, str]] = {
        "Haar Cascade": (lambda: CascadeFaceDetector(cfg), "cascade"),
        "InsightFace (CPU)": (lambda: InsightFaceDetector(cfg), "insightface"),
        "Hailo Face (NPU)": (lambda: HailoFaceDetector(cfg), "hailo"),
        "IMX500 Face": (lambda: Imx500FaceDetector(cfg), "imx500"),
    }

    object_detectors: dict[str, tuple[Any, str]] = {
        "YOLO CPU": (lambda: YoloCpuDetector(cfg), "yolo_cpu"),
        "YOLO26 NCNN": (lambda: Yolo26NcnnDetector(cfg), "yolo_ncnn"),
        "Hailo YOLO (NPU)": (lambda: YoloHailoDetector(cfg), "yolo_hailo"),
        "IMX500 Object": (lambda: Imx500ObjectDetector(Imx500Config()), "yolo_imx500"),
    }

    all_sections: list[tuple[str, dict[str, dict[str, Any]]]] = []

    if args.mode in {"all", "face"}:
        res = _run_bench_suite(face_detectors, img, args.iterations, "Face")
        all_sections.append(("Face Detectors", res))

    if args.mode in {"all", "object"}:
        res = _run_bench_suite(object_detectors, img, args.iterations, "Object")
        all_sections.append(("Object Detectors", res))

    print("\n" + "=" * 78)
    for title, results in all_sections:
        print(f"  {title}")
        print("-" * 78)
        print(f"{'Detector':<26} | {'Latency (ms)':<14} | {'FPS':<8} | {'Detections':<10} | Status")
        print("-" * 78)
        for name, m in results.items():
            lat = f"{m['avg_ms']:.1f}" if m["avg_ms"] != float("inf") else "N/A"
            fps = f"{m['fps']:.1f}" if m["fps"] > 0 else "N/A"
            print(f"{name:<26} | {lat:<14} | {fps:<8} | {m['found']:<10} | {m['status']}")
        print("=" * 78)


if __name__ == "__main__":
    main()
