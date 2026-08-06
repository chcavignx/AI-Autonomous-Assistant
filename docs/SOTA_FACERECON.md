# State of the Art — Face Detection & Recognition on Raspberry Pi 5

## Contexte hardware Pi 5

## Raspberry Pi 5 Hardware Context

The Raspberry Pi 5 runs on a **Cortex-A76 @ 2.4 GHz** with a VideoCore VII GPU — significantly more powerful than the Pi 4, but the CPU alone remains insufficient for heavy real-time deep learning. The real game changer is the **AI HAT+ (Hailo-8L / Hailo-8)** via PCIe 3.0 or the smart camera **Sony IMX500**.

---

## Level 1 — Classic Approach (Pure CPU, no AI)

**Stack :** OpenCV + Haar Cascade + LBPH / Eigenfaces / Fisherfaces
**Stack:** OpenCV + Haar Cascade + LBPH / Eigenfaces / Fisherfaces

| Métrique | Valeur |
|---|---|
| Metric | Value |
|---|---|
| FPS (detection) | ~5–8 FPS |
| FPS (detection + recognition) | ~2–4 FPS |
| Accuracy | Poor (~70–80% controlled conditions) |
| Latency | 250–500ms/frame |

**Verdict:** Obsolete for production use. Sensitive to lighting, angle, and occlusion conditions. To be avoided except for educational PoC.

---

## Level 2 — Lightweight Deep Learning (CPU + Optimizations)

**Stack:** OpenCV DNN / `face_recognition` (dlib) / **MobileNet** via **ncnn** or **ONNX Runtime**

- `face_recognition` (dlib HOG/CNN): ~1–3 FPS on Pi 5 CPU.
- YOLOv8n / YOLO26n via the **ncnn** framework on Pi 5 8GB achieves approximately 12–15 FPS in detection.
- **MobileNetV2 + ArcFace/FaceNet** for recognition: ~3–5 FPS.

**Key Optimizations:**

- INT8 Quantization.
- Processing 1 frame out of N (frame skipping).
- Reduced resolution ($320 \times 320$ instead of $640 \times 640$).
- ncnn backend with Vulkan (VideoCore VII GPU).

---

## Level 3 — Hybrid & Accelerated SOTA: IMX500 & Hailo-8L

### 1. Caméra AI Sony IMX500 (Split Architecture)

### 1. Sony IMX500 AI Camera (Split Architecture)

- **Détection :** Déportée sur le capteur IMX500 (**30 FPS fixe**).
- **Reconnaissance :** **InsightFace `buffalo_l` (ArcFace)** exécuté sur CPU local (112x112 aligned crops).
- **Avantage :** Consommation CPU quasi nulle pour la détection, libérant le Pi 5 pour les modules vocaux (ASR/TTS).
- **Detection:** Offloaded to the IMX500 sensor (**fixed 30 FPS**).
- **Recognition:** **InsightFace `buffalo_l` (ArcFace)** executed on local CPU (112x112 aligned crops).
- **Advantage:** Near-zero CPU consumption for detection, freeing up the Pi 5 for voice modules (ASR/TTS).

### 2. AI HAT+ (Hailo-8L NPU)

- **Stack :** **YOLOv5-personface / YOLOv8** compilé pour Hailo + **ArcFace / InsightFace**.
- **Performance :** Détection `yolov5s_personface` à **150 FPS**, YOLOv8s à 127 FPS à $640 \times 640$.
- **Charge CPU :** < 20% d'utilisation pendant les inférences.
- **Stack:** **YOLOv5-personface / YOLOv8** compiled for Hailo + **ArcFace / InsightFace**.
- **Performance:** `yolov5s_personface` detection at **150 FPS**, YOLOv8s at 127 FPS at $640 \times 640$.
- **CPU Load:** < 20% utilization during inferences.

---

## Synthetic Comparative Table

| Approche | Hardware | FPS Détection | Précision | Complexité |
|---|---|---|---|---|
| Approach | Hardware | Detection FPS | Accuracy | Complexity |
|---|---|---|---|---|
| Haar Cascade + LBPH | Pi 5 CPU | 5–8 | ~75% | Low |
| dlib face_recognition | Pi 5 CPU | 1–3 | ~85% | Medium |
| YOLO26n + ncnn | Pi 5 CPU+GPU | 12–15 | ~92% | Medium |
| IMX500 + InsightFace | Pi 5 + AI Camera | **30 (Fixed)** | ~95%+ | Medium |
| YOLOv5-face + Hailo-8L | Pi 5 + AI HAT | **~150** | ~95%+ | High |

---

## Practical Recommendations

1. **For a real-time & high-accuracy project (> 90%):**
   - Raspberry Pi 5 8GB + **Sony IMX500 AI Camera** or **AI HAT+ Hailo-8L (~$70)**.
   - Pipeline: **IMX500 / YOLO** (detection) $\to$ **ArcFace / InsightFace** (INT8 embeddings).

2. **Without dedicated hardware (Budget constrained):**
   - YOLO26n / MobileNetV2 + ncnn + Vulkan backend.
   - Resolution $320 \times 320$, skip 2 frames out of 3.

3. **Pitfalls to avoid:**
   - Native PyTorch/TensorFlow on CPU without quantization: unusable in real-time.
   - Lack of active cooling: the Pi 5 reaches its thermal throttling limit (80–85°C) in less than 2 minutes of continuous video processing.
