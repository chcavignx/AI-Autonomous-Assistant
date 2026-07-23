# Developpment Process

## Priorities to Establish in Order to Meet Technical and Temporal Constraints

To maximize the chances of success and avoid dispersion, it is essential to prioritize objectives based on technical complexity, impact on the AI assistant, and available time.
Here is a prioritization proposed for the context: Raspberry Pi 5, Hailo-8L, NVMe SSD, official camera, and USB microphone.

### 1. **Top Priority: Essential and Feasible Modules**

- **Offline Voice Recognition (USB Microphone)**
  - *Why?* It is the core of user interaction, quick to set up, and resource-efficient.
  - *Technical Impact:* Low to moderate, with comprehensive documentation, abundant examples, and existing projects.

- **Offline Speech Synthesis**
  - *Why?* Provides immediate audio feedback, easy to install, with natural-sounding voices.
  - *Technical Impact:* Low, with direct integration into Python.

### 2. **Intermediate Priority: Computer Vision**

- **Facial Recognition (face_recognition + official camera)**
  - *Why?* Adds a layer of personalization and security, but requires more resources and optimization.
  - *Technical Impact:* Moderate; requires proper lighting and testing on a reduced dataset.

- **Object Recognition**
  - *Why?* Advanced functionality, but more complex to integrate and optimize.
  - *Technical Impact:* High; requires model management, hardware acceleration, and extensive testing.

### 3. **Secondary Priority: Integration and User Experience**

- **Module Fusion (main script, command management)**
  - *Why?* Necessary for a coherent assistant, but to be executed once the basic modules are functional.
  - *Technical Impact:* Variable, depending on the complexity of the desired interface.

- **User Interface (screen, local web interface)**
  - *Why?* To enhance user experience.
  - *Technical Impact:* Low to moderate, can be postponed until after the AI modules have been validated.

### Summary Table of Priorities

| Objective                    | Priority | Ease of Implementation | Project Impact | Estimated Time |
|------------------------------|----------|------------------------|----------------|----------------|
| Offline Voice Recognition    | 1        | Easy                   | Essential      | 1-3 days       |
| Offline Speech Synthesis     | 1        | Easy                   | Essential      | 1-2 days       |
| Facial Recognition           | 2        | Moderate               | Significant    | 3-5 days       |
| Object Recognition           | 2        | Moderate to Difficult  | Advanced       | 5-8 days       |
| Module Integration           | 3        | Variable               | Coherence      | 3-5 days       |
| User Interface               | 3        | Easy to Moderate       | Optional       | 2-4 days       |

### Proposed Sequence to Meet Constraints and Integrate Modules

- **Audio Modules**:
  1. Confirm the USB microphone and speaker path with [AUDIO_USB_TEST.md](AUDIO_USB_TEST.md).
  2. Validate offline speech recognition with [STT_OFFLINE.md](STT_OFFLINE.md).
  3. Validate offline text-to-speech with [TTS_OFFLINE.md](TTS_OFFLINE.md).
  4. Verify the integrated wake-word, ASR, and TTS loop with `examples/VAD/voice_agent_offline.py`.
- **Follow with Vision Modules**:
    1. Start with facial recognition (simpler than object recognition).
    2. Proceed to object detection.
- **Test each module independently** before attempting integration.
- **Document each step and design decision** to avoid delays during iterations or corrections.
- **Adhere to minimum criteria**: define clear milestones (e.g., "voice command functional", "facial recognition operational") to measure progress.

### Sequenced Guides for Different Modules

- **Audio Modules**
    1. [USB microphone and speaker test](AUDIO_USB_TEST.md)
    2. [Offline Speech Recognition (STT)](STT_OFFLINE.md)
    3. [Offline Text-to-Speech (TTS)](TTS_OFFLINE.md)
    4. [Voice stack and VAD models](STS_VAD_MODELS.md)
    5. [Offline Speech-to-Speech demo](../examples/VAD/voice_agent_offline.md)
        Demo application that listens for a wake word, transcribes the next utterance, generates a keyword response, and speaks it back (`examples/VAD/voice_agent_offline.py`)

- **Vision Modules**
    1. [Vision Pipeline Architecture & 3-Phase Guide](VISION_PIPELINE.md)
        Detailed overview of the 3-phase pipeline (Detection, Alignment & Embedding, Identification), codebase architecture under `src/vision/`, and hardware choices.
    2. [IMX500, InsightFace & YOLO Integration Guide](IMX500_INSIGHTFACE_AND_YOLO.md)
        Complete guide for split hybrid architecture using Sony IMX500 AI Camera on-sensor detection + Host InsightFace `buffalo_l` recognition.
    3. [SOTA Facial Recognition & RPi5 Hardware Roadmap](SOTA_FACERECON.md)
        Comparative benchmark analysis across Haar Cascade, OpenCV DNN, IMX500 AI Camera, and Hailo-8L AI HAT+.
    4. [YOLO & NCNN CPU Optimization Guide](YOLO_NCNN_OPTIMIZATION.md)
        Optimizing YOLO26/11 for CPU-only inference using NCNN on ARM Cortex-A76.
    5. [AI Acceleration Roadmap & Comparative Study](AI_ROADMAP_COMPARATIVE.md)
        4-phase development roadmap comparing CPU baseline, Hailo-8L NPU, and Sony IMX500 smart camera performance.

- **Module Integration**
    1. Voice + Vision multimodal assistant orchestration (`src/vision/face_in_frame.py` + `src/audio/asr.py` + `src/audio/tts.py`).

- **Design Decision**
  1. Voice Agent Solution and Architecture: [Voice Agent Offline Solution](STS_VAD_MODELS.md)
  2. Vision Agent Solution and Architecture: [Vision Pipeline Architecture & 3-Phase Guide](VISION_PIPELINE.md)
  3. Hybrid Sensor-Host Vision Strategy: [IMX500, InsightFace & YOLO Guide](IMX500_INSIGHTFACE_AND_YOLO.md)
  4. Hardware Acceleration & Benchmarking Roadmap: [AI Acceleration Roadmap & Comparative Study](AI_ROADMAP_COMPARATIVE.md)
