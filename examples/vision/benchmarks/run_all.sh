#!/usr/bin/env bash
set -e

printf "\n=== 1. Master Vision Benchmark ===\n"
uv run python examples/vision/benchmarks/bench_vision_all.py --duration 1 --face cascade,insightface --object yolo_ncnn

printf "\n=== 2. Modular Detectors Benchmark ===\n"
uv run python examples/vision/benchmarks/bench_detectors.py --iterations 5

printf "\n=== 3. Combinations Benchmark ===\n"
uv run python examples/vision/benchmarks/bench_combinations.py --duration 1 --detectors cascade,insightface --object yolo_cpu

printf "\n=== 4. Phase 1 CPU Benchmark ===\n"
uv run python examples/vision/benchmarks/bench_face_phase1_cpu.py --iterations 5 --mock

printf "\n=== 5. Phase 2 IMX500 Benchmark ===\n"
uv run python examples/vision/benchmarks/bench_face_phase2_imx500.py --iterations 5 --mock

printf "\n=== 6. Phase Comparison Summary ===\n"
uv run python examples/vision/benchmarks/compare_bench_phases.py
