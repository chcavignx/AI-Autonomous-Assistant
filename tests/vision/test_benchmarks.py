"""Unit and integration tests for vision benchmark scripts in examples/vision/benchmarks."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.vision.benchmarks.bench_combinations import (  # ruff: ignore[module-import-not-at-top-of-file]
    _init_object_detector,  # ruff: ignore[import-private-name]
    _load_or_create_frame,  # ruff: ignore[import-private-name]
    benchmark_pipeline,
)
from examples.vision.benchmarks.bench_detectors import (  # ruff: ignore[module-import-not-at-top-of-file]
    _run_bench_suite,  # ruff: ignore[import-private-name]
    load_or_create_image,
)
from examples.vision.benchmarks.bench_vision_all import (  # ruff: ignore[module-import-not-at-top-of-file]
    FrameSource,
    _aggregate,  # ruff: ignore[import-private-name]
    _detect_face_status,  # ruff: ignore[import-private-name]
    _detect_object_status,  # ruff: ignore[import-private-name]
    _fps_bar_section,  # ruff: ignore[import-private-name]
    _latency_bar_section,  # ruff: ignore[import-private-name]
    _null_result,  # ruff: ignore[import-private-name]
    _run_face_benchmark,  # ruff: ignore[import-private-name]
    _run_object_benchmark,  # ruff: ignore[import-private-name]
    save_csv,
    save_markdown,
)
from examples.vision.benchmarks.compare_bench_phases import (  # ruff: ignore[module-import-not-at-top-of-file]
    load_bench_csv,
    summarize,
)
from src.utils.metrics import ThroughputMeter  # ruff: ignore[module-import-not-at-top-of-file]
from src.vision.face_detector import CascadeFaceDetector  # ruff: ignore[module-import-not-at-top-of-file]

# ─────────────────────────────────────────────────────────────────────────────
# bench_vision_all.py tests
# ─────────────────────────────────────────────────────────────────────────────


def test_frame_source_synthetic():
    source = FrameSource(image_path=None, camera_index=999)
    frame = source.read()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    source.release()


def test_frame_source_with_existing_image(tmp_path):
    import cv2

    img_path = tmp_path / "test.jpg"
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(img_path), test_img)

    source = FrameSource(image_path=str(img_path))
    frame = source.read()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (100, 100, 3)
    source.release()


def test_detect_face_status():
    assert _detect_face_status(None, "cascade") == "OK"

    mock_proc = MagicMock()
    mock_proc.detector = MagicMock()
    mock_proc.detector.use_hailo = False
    assert _detect_face_status(mock_proc, "hailo") == "FALLBACK (cascade)"

    mock_proc.detector = MagicMock(spec=CascadeFaceDetector)
    assert _detect_face_status(mock_proc, "insightface") == "FALLBACK (cascade)"
    assert _detect_face_status(mock_proc, "imx500") == "FALLBACK (cascade)"


def test_detect_object_status():
    mock_det = MagicMock()
    mock_det.use_hailo = False
    assert _detect_object_status(mock_det, "yolo_hailo") == "FALLBACK (cpu)"

    mock_det.imx500 = None
    assert _detect_object_status(mock_det, "yolo_imx500") == "FALLBACK (stub)"

    mock_det.use_hailo = True
    assert _detect_object_status(mock_det, "yolo_cpu") == "OK"


def test_aggregate_metrics():
    meter = ThroughputMeter()
    for _ in range(10):
        meter.tick()
    latencies = [10.0, 20.0, 30.0]
    cpus = [50.0, 60.0]
    temps = [40.0, 42.0]

    agg = _aggregate(meter, latencies, cpus, temps, status="OK")
    assert agg["status"] == "OK"
    assert agg["lat_mean_ms"] == pytest.approx(20.0, 0.1)
    assert agg["lat_p50_ms"] == pytest.approx(20.0, 0.1)
    assert agg["cpu_mean"] == pytest.approx(55.0, 0.1)
    assert agg["temp_mean"] == pytest.approx(41.0, 0.1)


def test_null_result():
    res = _null_result()
    assert res["status"] == "FAILED"
    assert res["fps"] == 0.0


def test_save_csv_and_markdown(tmp_path):
    rows = [
        {
            "category": "face",
            "backend": "cascade",
            "fps": 15.0,
            "lat_mean_ms": 66.7,
            "lat_p50_ms": 65.0,
            "lat_p95_ms": 70.0,
            "cpu_mean": 45.0,
            "temp_mean": 50.0,
            "status": "OK",
        },
        {
            "category": "face",
            "backend": "hailo",
            "fps": 15.0,
            "lat_mean_ms": 66.7,
            "lat_p50_ms": 65.0,
            "lat_p95_ms": 70.0,
            "cpu_mean": 45.0,
            "temp_mean": 50.0,
            "status": "FALLBACK (cascade)",
        },
    ]

    csv_p = tmp_path / "bench.csv"
    md_p = tmp_path / "bench.md"

    save_csv(rows, csv_p)
    assert csv_p.exists()
    with csv_p.open() as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["backend"] == "cascade"

    save_markdown(rows, md_p, duration=5.0, ts="20260815T120000Z")
    assert md_p.exists()
    content = md_p.read_text()
    assert "# Vision Benchmark Report" in content
    assert "cascade" in content
    assert "FALLBACK (cascade)" in content


def test_ascii_bar_sections():
    rows = [
        {"backend": "cascade", "fps": 10.0, "lat_mean_ms": 100.0, "status": "OK"},
        {"backend": "hailo", "fps": 5.0, "lat_mean_ms": 200.0, "status": "FALLBACK (cascade)"},
        {"backend": "failed", "fps": 0.0, "lat_mean_ms": 0.0, "status": "FAILED"},
    ]
    fps_bars = _fps_bar_section(rows)
    lat_bars = _latency_bar_section(rows)
    assert "cascade" in fps_bars
    assert "10.0 FPS" in fps_bars
    assert "(FB)" in fps_bars
    assert "N/A" in fps_bars
    assert "100.0 ms" in lat_bars


def test_run_face_and_object_benchmark_with_source(tmp_path):
    source = FrameSource(image_path=None, camera_index=999)

    res_face = _run_face_benchmark("cascade", source, duration_sec=0.1)
    assert res_face["status"] == "OK"
    assert res_face["fps"] > 0

    res_obj = _run_object_benchmark("yolo_cpu", source, duration_sec=0.1)
    assert res_obj["status"] == "OK"
    assert res_obj["fps"] > 0

    source.release()


# ─────────────────────────────────────────────────────────────────────────────
# bench_detectors.py tests
# ─────────────────────────────────────────────────────────────────────────────


def test_bench_detectors_load_image():
    img = load_or_create_image(ROOT_DIR, None)
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3


def test_run_bench_suite():
    mock_det = MagicMock()
    mock_det.detect.return_value = [{"box": [0, 0, 10, 10], "score": 0.9}]
    detectors = {
        "MockDetector": (lambda: mock_det, "mock"),
    }
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = _run_bench_suite(detectors, dummy_img, iterations=5, category_label="Test")
    assert "MockDetector" in res
    assert res["MockDetector"]["status"] == "OK"
    assert res["MockDetector"]["found"] == 1
    assert res["MockDetector"]["fps"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# bench_combinations.py tests
# ─────────────────────────────────────────────────────────────────────────────


def test_bench_combinations_detector():
    static_frame, cap = _load_or_create_frame(None, camera_index=999)
    assert static_frame is not None or cap is not None
    # Face only
    res_face = benchmark_pipeline("cascade", None, duration_sec=0.1, static_frame=static_frame, cap=cap)
    assert res_face["status"] == "OK"
    assert res_face["fps"] > 0

    # Face + Object combination
    res_comb = benchmark_pipeline("cascade", "yolo_cpu", duration_sec=0.1, static_frame=static_frame, cap=cap)
    assert res_comb["status"] == "OK"
    assert res_comb["fps"] > 0


def test_init_object_detector():
    from unittest.mock import patch

    mock_cfg = MagicMock()
    with patch("src.vision.yolo_cpu.YOLO"):
        det = _init_object_detector("yolo_cpu", mock_cfg)
        assert det is not None


# ─────────────────────────────────────────────────────────────────────────────
# compare_bench_phases.py tests
# ─────────────────────────────────────────────────────────────────────────────


def test_compare_bench_phases_loaders(tmp_path):
    # Test per-frame schema
    csv1 = tmp_path / "p1.csv"
    with csv1.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms", "cpu_percent", "temp_c"])
        writer.writerow([10.0, 50.0, 40.0])
        writer.writerow([20.0, 60.0, 42.0])

    d1 = load_bench_csv(csv1)
    assert len(d1["latency_ms"]) == 2
    s1 = summarize("Phase 1", d1)
    assert s1["lat_mean"] == pytest.approx(15.0, 0.1)
    assert s1["fps_mean"] == pytest.approx(1000.0 / 15.0, 0.1)

    # Test aggregate schema
    csv2 = tmp_path / "p2.csv"
    with csv2.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat_mean_ms", "cpu_mean", "temp_mean"])
        writer.writerow([5.0, 20.0, 38.0])

    d2 = load_bench_csv(csv2)
    assert len(d2["latency_ms"]) == 1
    s2 = summarize("Phase 2", d2)
    assert s2["lat_mean"] == pytest.approx(5.0, 0.1)
    assert s2["fps_mean"] == pytest.approx(200.0, 0.1)
