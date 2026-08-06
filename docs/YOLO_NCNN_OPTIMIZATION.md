# Optimization: YOLO26 & NCNN on Raspberry Pi 5 CPU

This document provides a guide for running object detection with **YOLO26 / YOLO11** optimized for the Raspberry Pi 5 CPU using the **NCNN** inference framework.

---

## 1. Technical Justification: Why NCNN on RPi5 CPU?

Executing **YOLO26n** via **NCNN** is the State-of-the-Art approach for CPU-only inference on Raspberry Pi 5:

- **Speed Improvement**: YOLO26n achieves ~7.79 FPS compared to 6.79 FPS for YOLO11n on Pi 5 CPU (~15% speedup).
- **Higher Accuracy**: Higher mAP (40.1 vs 39.5) while maintaining a smaller footprint.
- **NCNN Optimization**: NCNN is specifically tailored for ARM Cortex architecture, executing a single frame inference in ~67.69 ms.

---

## 2. Installation Guide (Direct Python Environment)

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip -y

# 2. Install Ultralytics with export dependencies
pip install ultralytics[export] ncnn
```

---

## 3. Python Implementation: `src/vision/yolo_cpu.py`

Below is the optimized script pattern that performs automatic NCNN export and utilizes `picamera2` for video capture on Raspberry Pi OS Bookworm:

```python
import cv2
from ultralytics import YOLO

# 1. Load original PyTorch model
model = YOLO("yolo26n.pt")

# 2. Export to NCNN format (Only needed once)
model.export(format="ncnn")

# 3. Load optimized NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

# 4. Perform stream inference
results = ncnn_model.predict(source="0", show=True, stream=True)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        print(f"Detected: {ncnn_model.names[int(box.cls)]} with confidence {box.conf:.2f}")
```

---

## 4. Performance Recommendations

1. **Use Nano Models (`yolo26n`, `yolo11n`)**: Larger models (`s`, `m`, `l`) are too heavy for CPU-only real-time execution.
2. **Active Cooling**: CPU-intensive inference rapidly raises temperature. Active Cooling is required to prevent thermal throttling (which lowers CPU clock from 2.4 GHz down to 1.5 GHz).
3. **Optional Overclocking**: Overclocking CPU from 2.4 GHz to **2.9 GHz** in `/boot/firmware/config.txt` yields ~15-20% additional throughput.
