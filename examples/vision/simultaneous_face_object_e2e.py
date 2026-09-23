#!/usr/bin/env python3
"""Simultaneous Face Recognition and Object Detection E2E Demonstration & Test.

This script runs the unified vision pipeline with simultaneous YOLO object detection
and ArcFace face recognition powered by asynchronous decoupled tracking.

Usage:
    # Live interactive GUI mode (default camera):
    python examples/vision/simultaneous_face_object_e2e.py

    # Headless mode for SSH / terminal testing (runs for 10 seconds and prints stats):
    python examples/vision/simultaneous_face_object_e2e.py --headless --duration 10

    # Automated E2E test verification mode (exits 0 on success):
    python examples/vision/simultaneous_face_object_e2e.py --test

    # Run on a static test image:
    python examples/vision/simultaneous_face_object_e2e.py --input path/to/image.jpg

Interactive Controls (GUI mode):
    - 'o': Toggle YOLO Object Detection ON / OFF
    - 'f': Toggle Face Detection ON / OFF
    - 'r': Toggle ArcFace Face Recognition ON / OFF
    - 'a': Toggle Asynchronous Recognition Mode ON / OFF
    - 's': Save snapshot of current annotated frame
    - 'h': Print help / status
    - 'q' or ESC: Exit application
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.sysutils import get_cpu_temperature_c, get_cpu_usage_percent  # noqa: E402
from src.vision.video_capture import VideoCapture, compute_iou  # noqa: E402

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End Simultaneous Face Recognition & Object Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI windows (suitable for SSH / servers)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run automated end-to-end test verification suite and exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Execution duration in seconds (0 = run indefinitely)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Maximum number of frames to process (0 = infinite)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to an optional static input image to process",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Directory to save output annotated images or captures",
    )
    return parser.parse_args()


def draw_hud(
    frame: np.ndarray,
    vc: VideoCapture,
    fps: float,
    frame_idx: int,
    detected_objects: list[str],
    detected_faces: list[tuple[str, float]],
) -> np.ndarray:
    """Draw high-visibility HUD banner and diagnostics overlay on top of frame."""
    h, w = frame.shape[:2]
    hud = frame.copy()

    cv2.rectangle(hud, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.rectangle(hud, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX

    title = "SIMULTANEOUS VISION: OBJECT DETECTION + FACE RECOGNITION"
    cv2.putText(frame, title, (15, 24), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    pills_text = (
        f"[Objs: {'ON' if vc.enable_object_detection else 'OFF'}] "
        f"[Face: {'ON' if vc.enable_face_detection else 'OFF'}] "
        f"[Recog: {'ON' if vc.enable_face_recognition else 'OFF'}] "
        f"[Async: {'ON' if vc.async_face_recognition else 'OFF'}]"
    )
    cv2.putText(frame, pills_text, (15, 46), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cpu = get_cpu_usage_percent()
    temp = get_cpu_temperature_c()
    temp_str = f"{temp:.1f}C" if temp is not None else "N/A"

    obj_summary = ", ".join(detected_objects[:3]) if detected_objects else "none"
    if len(detected_objects) > 3:
        obj_summary += f" (+{len(detected_objects) - 3})"

    face_summary = (
        ", ".join([f"{name} ({sim:.2f})" for name, sim in detected_faces[:2]]) if detected_faces else "none"
    )

    metrics_text = f"FPS: {fps:4.1f} | Frame: {frame_idx:5d} | CPU: {cpu:4.1f}% | Temp: {temp_str}"
    dets_text = f"Objs: {obj_summary} | Faces: {face_summary}"

    cv2.putText(frame, metrics_text, (15, h - 22), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, dets_text, (15, h - 6), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def extract_current_detections(vc: VideoCapture) -> tuple[list[str], list[tuple[str, float]]]:
    """Extract human-readable lists of detected objects and faces."""
    objects: list[str] = []

    if vc.latest_results is not None:
        boxes = getattr(vc.latest_results, "boxes", None)
        names_dict = getattr(vc.latest_results, "names", {})
        if boxes is not None and hasattr(boxes, "cls"):
            for cls_id in boxes.cls:
                cid = int(cls_id.item() if hasattr(cls_id, "item") else cls_id)
                objects.append(names_dict.get(cid, f"class_{cid}"))
        elif hasattr(vc.latest_results, "detections") and isinstance(vc.latest_results.detections, (list, tuple)):
            objects.extend(d.get("label", "object") for d in vc.latest_results.detections if isinstance(d, dict))

    faces = [(trk["name"], trk["similarity"]) for trk in vc.active_tracks]
    return objects, faces


def _verify_pipeline_components(vc: VideoCapture) -> None:
    """Verify that all pipeline components are instantiated."""
    if getattr(vc, "object_processor", None) is None:
        msg = "Object processor not initialized"
        raise RuntimeError(msg)
    if getattr(vc, "face_processor", None) is None:
        msg = "Face processor not initialized"
        raise RuntimeError(msg)
    if not vc.has_async_recognition:
        msg = "Async recognition executor not active"
        raise RuntimeError(msg)
    logger.info("✓ Processors and async executor initialized correctly")


def _verify_frame_processing(vc: VideoCapture, test_frame: np.ndarray) -> None:
    """Verify simultaneous frame processing and feature toggles."""
    t0 = time.perf_counter()
    annotated = vc.process_frame(test_frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if annotated.shape != test_frame.shape:
        msg = f"process_frame invalid shape: {annotated.shape}"
        raise RuntimeError(msg)
    logger.info("✓ Simultaneous process_frame executed successfully in %.2f ms", elapsed_ms)

    vc.enable_object_detection = False
    _raw = vc.process_frame(test_frame)
    vc.enable_object_detection = True

    vc.enable_face_detection = False
    _raw2 = vc.process_frame(test_frame)
    vc.enable_face_detection = True
    logger.info("✓ Pipeline toggles operating without errors")


def _execute_test_sequence(vc: VideoCapture) -> None:
    """Execute sequence of checks for automated test."""
    _verify_pipeline_components(vc)

    test_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (220, 240), (200, 200, 200), -1)
    cv2.rectangle(test_frame, (350, 200), (550, 400), (50, 150, 50), -1)

    _verify_frame_processing(vc, test_frame)

    iou_same = compute_iou([10, 10, 50, 50], [10, 10, 50, 50])
    if abs(iou_same - 1.0) >= 1e-4:
        msg = f"IoU calculation error: expected 1.0, got {iou_same}"
        raise RuntimeError(msg)
    logger.info("✓ Spatial IoU tracker verified")


def run_automated_test() -> bool:
    """Run comprehensive automated E2E test verifying simultaneous vision functionality."""
    logger.info("============================================================")
    logger.info("  STARTING AUTOMATED E2E SIMULTANEOUS VISION TEST")
    logger.info("============================================================")

    cfg = load_config()
    cfg.vision.enable_object_detection = True
    cfg.vision.enable_face_detection = True
    cfg.vision.enable_face_recognition = True
    cfg.vision.async_face_recognition = True

    vc = VideoCapture(cfg)
    test_passed = True

    try:
        _execute_test_sequence(vc)
        logger.info("============================================================")
        logger.info("  E2E SIMULTANEOUS VISION TEST PASSED SUCCESSFULLY!")
        logger.info("============================================================")
    except Exception as e:
        logger.exception("✗ Automated E2E test failed: %s", e)
        test_passed = False
    finally:
        vc.stop()

    return test_passed


def _handle_keyboard_events(key: int, vc: VideoCapture) -> bool:
    """Handle interactive GUI keyboard shortcuts. Return True to continue, False to quit."""
    if key in (ord("q"), 27):  # 'q' or ESC
        logger.info("Quit requested by user.")
        return False
    if key == ord("o"):
        vc.enable_object_detection = not vc.enable_object_detection
        logger.info("Toggled Object Detection: %s", vc.enable_object_detection)
    elif key == ord("f"):
        vc.enable_face_detection = not vc.enable_face_detection
        logger.info("Toggled Face Detection: %s", vc.enable_face_detection)
    elif key == ord("r"):
        vc.enable_face_recognition = not vc.enable_face_recognition
        logger.info("Toggled Face Recognition: %s", vc.enable_face_recognition)
    elif key == ord("a"):
        vc.async_face_recognition = not vc.async_face_recognition
        logger.info("Toggled Async Recognition: %s", vc.async_face_recognition)
    elif key == ord("s"):
        ok, msg = vc.capture_photo()
        logger.info("Captured snapshot: %s (%s)", ok, msg)
    elif key == ord("h"):
        logger.info(
            "Hotkeys: [o] Object Detection | [f] Face Detection | [r] Face Recog | [a] Async | [s] Capture | [q] Quit"
        )
    return True


def _log_summary(frame_idx: int, start_time: float, total_objs: int, total_faces: int) -> None:
    """Log terminal summary at the end of execution."""
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    logger.info("============================================================")
    logger.info("  SIMULTANEOUS VISION SESSION SUMMARY")
    logger.info("============================================================")
    logger.info("Total Frames Processed : %d", frame_idx)
    logger.info("Total Elapsed Time     : %.2f seconds", total_time)
    logger.info("Average Pipeline FPS   : %.1f FPS", avg_fps)
    logger.info("Total Object Instances : %d", total_objs)
    logger.info("Total Face Instances   : %d", total_faces)
    logger.info("============================================================")


def _step_live_frame(
    vc: VideoCapture,
    frame_idx: int,
    fps_smooth: float,
    args: argparse.Namespace,
    window_name: str,
) -> tuple[bool, int, int]:
    """Capture, process, and display a single live frame. Return (continue_running, obj_count, face_count)."""
    frame = vc.capture_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 100, 640, dtype=np.uint8)
        frame[:, :, 2] = np.linspace(100, 0, 640, dtype=np.uint8)
        time.sleep(0.033)

    annotated = vc.process_frame(frame)
    objs, faces = extract_current_detections(vc)
    display_frame = draw_hud(annotated, vc, fps_smooth, frame_idx, objs, faces)

    if not args.headless:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if not _handle_keyboard_events(key, vc):
            return False, len(objs), len(faces)
    elif frame_idx % 30 == 0:
        logger.info(
            "Frame %5d | FPS: %4.1f | Objs: %s | Faces: %s",
            frame_idx,
            fps_smooth,
            objs or "none",
            [f[0] for f in faces] or "none",
        )

    return True, len(objs), len(faces)


def _iterate_stream_loop(vc: VideoCapture, args: argparse.Namespace, window_name: str, start_time: float) -> tuple[int, int, int]:
    """Loop through frames until stop condition is met."""
    frame_idx = 0
    fps_smooth = 0.0
    total_objects_detected = 0
    total_faces_detected = 0

    while True:
        t0 = time.perf_counter()
        frame_idx += 1

        keep_running, num_objs, num_faces = _step_live_frame(vc, frame_idx, fps_smooth, args, window_name)
        total_objects_detected += num_objs
        total_faces_detected += num_faces

        if not keep_running:
            break

        dt = time.perf_counter() - t0
        current_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps if fps_smooth > 0 else current_fps

        if args.duration > 0 and (time.time() - start_time) >= args.duration:
            logger.info("Reached target duration of %.1f seconds.", args.duration)
            break
        if args.frames > 0 and frame_idx >= args.frames:
            logger.info("Processed target of %d frames.", args.frames)
            break

    return frame_idx, total_objects_detected, total_faces_detected


def _run_live_stream_loop(vc: VideoCapture, args: argparse.Namespace, window_name: str) -> None:
    """Execute live streaming frame capture and simultaneous processing loop."""
    start_time = time.time()
    frame_idx, total_objs, total_faces = 0, 0, 0

    try:
        frame_idx, total_objs, total_faces = _iterate_stream_loop(vc, args, window_name, start_time)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        _log_summary(frame_idx, start_time, total_objs, total_faces)


def main() -> int:
    """Run E2E simultaneous face recognition and object detection demonstration."""
    args = parse_args()

    if args.test:
        success = run_automated_test()
        return 0 if success else 1

    cfg = load_config()
    cfg.vision.enable_object_detection = True
    cfg.vision.enable_face_detection = True
    cfg.vision.enable_face_recognition = True

    logger.info("Initializing VideoCapture with simultaneous Object Detection & Face Recognition...")
    vc = VideoCapture(cfg)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error("Input file '%s' does not exist.", args.input)
            vc.stop()
            return 1

        logger.info("Processing static input image: %s", input_path)
        img = cv2.imread(str(input_path))
        if img is None:
            logger.error("Could not read image from %s", input_path)
            vc.stop()
            return 1

        annotated = vc.process_frame(img)
        objs, faces = extract_current_detections(vc)
        annotated = draw_hud(annotated, vc, fps=0.0, frame_idx=1, detected_objects=objs, detected_faces=faces)

        out_path = args.save_output or str(input_path.with_name(f"{input_path.stem}_simultaneous_out.jpg"))
        cv2.imwrite(out_path, annotated)
        logger.info("Annotated result saved to: %s", out_path)
        logger.info("Detected Objects: %s | Recognized Faces: %s", objs, faces)
        vc.stop()
        return 0

    vc.start()

    window_name = "Simultaneous Vision: YOLO Detection + ArcFace Recognition"
    if not args.headless:
        if "DISPLAY" not in os.environ:
            logger.warning("No DISPLAY environment variable found. Switching to headless mode.")
            args.headless = True
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1024, 768)

    logger.info("Starting live pipeline. Press 'h' for help, 'q' to quit.")
    try:
        _run_live_stream_loop(vc, args, window_name)
    finally:
        vc.stop()
        if not args.headless:
            with contextlib.suppress(Exception):
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
