#!/usr/bin/env python3
"""Motion detection alert script with modular face and object recognition."""

import logging
import queue
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np # pyright: ignore[reportUnusedImport]

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.vision.face_insight_frame import FaceInsightFrame # pyright: ignore[reportMissingImports]
from src.vision.object_insight_frame import ObjectInsightFrame

import picamera2
from picamera2 import MappedArray, Process, RemoteMappedArray # pyright: ignore[reportUnusedImport]

app_name = __name__.split(".")[-1]
logger = logging.getLogger(app_name)

last_frame = None


def motion_detection_worker(request) -> dict:
    """Calculates motion in a background process using frame differencing."""
    global last_frame

    # Retrieve frame array from remote process request
    # Note: Picamera2 buffers are typically RGB/XRGB or YUV.
    # RemoteMappedArray maps it in-place.
    with RemoteMappedArray(request, "main") as m:
        frame = m.array.copy()

    # Convert to grayscale for efficient difference calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if last_frame is None:
        last_frame = gray
        return {"motion": False}

    frame_delta = cv2.absdiff(last_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Motion sensitivity threshold
            motion_detected = True
            break

    last_frame = gray
    return {"motion": motion_detected}


def return_thread(futures, object_processor, face_processor, cfg) -> None:
    """Listens for motion detection results from background processes."""
    motion_cooldown = 0.0

    while True:
        request = None
        try:
            future, request = futures.get()
            if future is None:
                break

            res = future.result()
            if res and res.get("motion"):
                now = time.time()
                if now > motion_cooldown:
                    logger.info("🚨 MOTION DETECTED via background worker process!")

                    # Map array to BGR for our image processors
                    frame_rgb = request.make_array("main")
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # 1. Run Object Recognition
                    _, objects = object_processor.process_frame(frame_bgr, draw=False)
                    for obj in objects:
                        logger.info(f"  [Object] {obj['label']} (Score: {obj['score']:.2f})")

                    # 2. Run Face Recognition
                    _, faces = face_processor.process_frame(frame_bgr, thresh=cfg.vision.face_recognition_threshold, draw=False)
                    for face in faces:
                        logger.info(f"  [Face] {face['name']} (Similarity: {face['similarity']:.2f})")

                    motion_cooldown = now + 5.0
        except Exception as e:
            logger.error(f"Error in return_thread: {e}")
        finally:
            # Always release request back to the camera system to prevent stall
            if request is not None:
                request.release()


def motion_detection_alert() -> None:
    logger.info("Initializing Motion Detection Alert System (Picamera2 Multiprocessing)...")
    cfg = load_config()

    # Initialize modular processors on the main process
    logger.info("Loading object and face recognition processors...")
    object_processor = ObjectInsightFrame(cfg)
    face_processor = FaceInsightFrame(cfg, detector_type=cfg.vision.face_detector_type)

    # Initialize Picamera2
    # Find camera index or fallback to 0
    from src.utils.camera import discover_pi_cameras
    cameras = discover_pi_cameras()
    camera_index = cameras[0].index if cameras else 0

    picam2 = picamera2.Picamera2(camera_index)

    # Configure main stream
    config = picam2.create_video_configuration(
        main={
            "size": (cfg.vision.camera.frame_width, cfg.vision.camera.frame_height),
            "format": cfg.vision.camera.format
        },
        buffer_count=6  # Keep extra buffers to avoid pipeline stall when queueing
    )
    picam2.configure(config)
    picam2.start()

    # Initialize Picamera2 Process worker
    process = Process(motion_detection_worker, picam2)

    futures = queue.Queue()
    thr = threading.Thread(
        target=return_thread,
        args=(futures, object_processor, face_processor, cfg),
        daemon=True
    )
    thr.start()

    logger.info("Alert system active. Monitoring for motion (Ctrl+C to exit)...")

    try:
        while True:
            # Capture request asynchronously using context manager
            with picam2.captured_request() as request:
                # Increment buffer reference count since it will be queued to the return_thread
                request.acquire()
                future = process.send(request)
                futures.put((future, request))
    except KeyboardInterrupt:
        logger.info("Shutting down alert system...")
    finally:
        # Signal return thread to exit
        futures.put((None, None))
        thr.join(timeout=2.0)
        process.close()
        picam2.stop()


if __name__ == "__main__":
    motion_detection_alert()
