import argparse
import os
import sys
import threading
import time
from functools import lru_cache

import cv2
from flask import Flask, Response, render_template_string

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

last_detections = []
latest_frame_bytes = None
frame_lock = threading.Lock()
fps_counter = 0
fps_value = 0.0
fps_start_time = time.time()


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections

    postprocess = intrinsics.postprocess if intrinsics else ""

    if postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    elif postprocess == "yolov8":
        from picamera2.devices.imx500 import postprocess_yolov8_detection
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes, scores, classes = postprocess_yolov8_detection(
            outputs=np_outputs, conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    elif postprocess == "yolov5":
        from picamera2.devices.imx500 import postprocess_yolov5_detection
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes, scores, classes = postprocess_yolov5_detection(
            outputs=np_outputs, model_input_shape=(input_h, input_w), conf_thres=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata) for box, score, category in zip(boxes, scores, classes) if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels or []

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    metadata = request.get_metadata()
    if not metadata:
        return
    detections = parse_detections(metadata)
    if not detections:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            category_idx = int(detection.category)
            label_str = labels[category_idx] if category_idx < len(labels) else str(category_idx)
            label = f"{label_str} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),  # Background color (white)
                cv2.FILLED,
            )

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), color)


def get_args():
    parser = argparse.ArgumentParser(description="IMX500 Object Detection Demo with Web Stream")
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument(
        "--bbox-order", choices=["yx", "xy"], default="yx", help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet", "yolov8", "yolov5"], default=None, help="Run post process of type")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP Web server host address")
    parser.add_argument("--port", type=int, default=5002, help="HTTP Web server port (default 5002)")
    parser.add_argument("--no-web", action="store_true", help="Disable web streaming server and use native X11 window")
    return parser.parse_args()


# Flask Web Application Setup
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IMX500 Object Detection Web Preview</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0d0e15;
            --bg-card: #161824;
            --border: #282a3a;
            --accent: #3b82f6;
            --accent-green: #10b981;
            --text: #f3f4f6;
            --text-sub: #9ca3af;
        }
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-main);
            color: var(--text);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            background: linear-gradient(135deg, var(--accent), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            max-width: 800px;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        img {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        .stats {
            margin-top: 16px;
            width: 100%;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }
        .stat-item {
            background: rgba(255,255,255,0.03);
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        .stat-label { font-size: 0.8rem; color: var(--text-sub); }
        .stat-value { font-size: 1.1rem; font-weight: 600; color: var(--accent-green); margin-top: 4px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Sony IMX500 AI Camera Live Stream</h1>
        <img src="/video_feed" alt="Live Camera Preview">
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Model Path</div>
                <div class="stat-value" style="font-size:0.85rem; word-break:break-all;">{{ model_path }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Postprocess Type</div>
                <div class="stat-value">{{ postprocess }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Detection Threshold</div>
                <div class="stat-value">{{ threshold }}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    postprocess_str = str(intrinsics.postprocess) if intrinsics and intrinsics.postprocess else "SSD MobileNet (Firmware)"
    return render_template_string(
        HTML_TEMPLATE,
        model_path=args.model,
        postprocess=postprocess_str,
        threshold=args.threshold,
    )


def generate_mjpeg_stream():
    """Yield MJPEG stream from latest drawn frame buffer."""
    while True:
        with frame_lock:
            if latest_frame_bytes is None:
                time.sleep(0.03)
                continue
            frame_data = latest_frame_bytes

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value, "r") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        label_file = "assets/coco_labels.txt"
        if not os.path.exists(label_file):
            label_file = os.path.join(os.path.dirname(__file__), "assets/coco_labels.txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()

    use_web = not args.no_web

    if use_web:
        print(f"\n============================================================")
        print(f"  Sony IMX500 Object Detection - SSH Web Streaming Server")
        print(f"============================================================")
        print(f"Access preview stream at: http://{args.host}:{args.port}")
        print(f"Press Ctrl+C to stop\n")
        picam2.start(config, show_preview=False)
    else:
        picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    def capture_loop():
        global latest_frame_bytes, fps_counter, fps_value, fps_start_time
        while True:
            request = picam2.capture_request()
            draw_detections(request)
            frame_arr = request.make_array("main")
            # RGB -> BGR for OpenCV JPG encoding
            frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)

            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps_value = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_start_time = time.time()

            ret, jpeg_buf = cv2.imencode(".jpg", frame_bgr)
            if ret:
                with frame_lock:
                    latest_frame_bytes = jpeg_buf.tobytes()

            request.release()

    if use_web:
        # Start capture loop in background thread
        cap_thread = threading.Thread(target=capture_loop, daemon=True)
        cap_thread.start()

        # Run Flask server on main thread
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    else:
        picam2.pre_callback = draw_detections
        while True:
            time.sleep(1)
