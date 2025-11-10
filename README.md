# AI Autonomous Assistant

Development of an embedded AI assistant fully operating locally on a Raspberry Pi 5.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Hardware Configuration](#hardware-configuration)
- [Results and Benchmarks](#results-and-benchmarks)
- [Development Process](#development-process)
- [Contributors](#contributors)
- [License](#license)

## Overview

Develop an embedded AI assistant that operates **entirely locally** on a Raspberry Pi, capable of:

- Facial recognition (identifying individuals)
- Object recognition (detection and classification)
- Voice comprehension (offline speech recognition)
- Speech synthesis (offline text-to-speech)
- Seamless interaction without reliance on the Internet

### Objectives

- **Complete cloud independence**: All features must work without an Internet connection, ensuring confidentiality and swift responses.
- **Natural user experience**: The assistant should be able to listen, understand, respond, and act via voice, similar to Siri or Jarvis, but operating locally.
- **Versatility**: Ability to recognize multiple users (through facial or voice recognition) and detect various everyday objects.
- **Responsiveness**: Fast response times for each interaction (ideally less than 1 second for both speech synthesis and recognition).
- **Customization**: Easily add new faces, objects, or voice commands.

### Main Use Cases

- **Local Voice Interaction**: Enable the user to provide voice commands to the assistant without an internet connection.
- **Natural Vocal Feedback**: The assistant responds using text-to-speech synthesis in a fluent and intelligible manner.
- **Facial Recognition**: Identify known individuals in front of the camera to personalize the experience or trigger specific actions.
- **Object Detection**: Recognize and label common objects within the immediate environment.
- **Command Management**: Utilize speech recognition to trigger actions (e.g., “Who is in front of the camera?”, “What objects do you see?”, etc.).
- **User Interface**: Provide a local web interface to enhance the user experience and facilitate the addition or removal of faces, objects, and commands, integrating the various modules.

### Expectations and Success Criteria

| Objective                     | Expected Accuracy       | Expected Performance                            | Limitations & Remarks                                                    |
|-------------------------------|-------------------------|-------------------------------------------------|--------------------------------------------------------------------------|
| **Facial Recognition**        | 95–98%                  | < 1 s/face, 2–5 FPS (up to 25 FPS with Hailo)     | Optimal performance with fewer than 20 faces; sensitive to lighting and angles |
| **Object Recognition**        | 80–90%                  | 2–5 FPS on Pi alone, up to 25 FPS (with Hailo-8L)| Accuracy decreases for similar objects and crowded scenes                 |
| **Offline Voice Recognition** | 85–95%                  | < 1 s, 2–5 s for longer phrases                  | Noise, accents, and command complexity can impact accuracy                |
| **Offline Speech Synthesis**  | Natural voice           | Real-time or near real-time                      | Expressive/multilingual voices require more resources; quality depends on the vocal model |

### Example Criteria

- Short commands, limited vocabulary, latency under 1 s
- Natural voice (FR/EN), immediate response
- Reliable recognition of 2–5 individuals
- Detection of 10–20 common objects (> 80%)
- Simple management of around a dozen commands, with easy scenario addition

## Installation

#### 1. **Clone the repository**

```bash
git clone https://github.com/chcavignx/AI-Autonomous-Assistant.git
cd AI-Autonomous-Assistant
```

#### 2. **Install dependencies**

Install Python packages:

```bash
pip install -r requirements.txt
```

Still to be completed

#### 3. **Hardware and Software Configuration**

    - (Specify the necessary connections and any hardware-specific steps)
    - *Example: Connect the official camera, USB microphone, SSD NVMe, AI accelerator Hailo-8L, etc.*

#### 4. **Launching the Assistant**

    - Example: 

```bash
python main.py
```

## Usage

TODO

- Explain how to interact with the assistant (typical voice commands, interface, etc.).
- Add screenshots if available or launch scripts.

## Features

- Local voice recognition (offline)
- Local speech synthesis
- Facial recognition (Hailo-8L optimizations)
- Object detection (Hailo-8L optimizations)
- User personalization (adding profiles, commands, scenarios)
- Complete privacy: no data sent to the cloud

## Hardware Configuration

| Component                    | Function / Details                                |
|------------------------------|---------------------------------------------------|
| Raspberry Pi 5 (8GB)         | Main computing unit with advanced connectivity    |
| SSD NVMe                     | High-speed storage for OS, data, and AI models    |
| Hailo-8L                     | Local AI acceleration for vision and detection    |
| Official Pi 5 Camera         | Image and video acquisition                       |
| Micro USB                    | High-quality audio capture                        |
| Optional Sound Card          | Enhanced audio output                             |

- **Preferred AI models**: “lite”, “tiny”, or quantized versions.
- **Training on PC**: transfer trained models for inference on the Pi.
- **Simplicity first**: progressive complexity and optimization.
- **Comparisons** (for documentation): test various cameras and configurations (8GB/16GB, Hailo-8L/8).

## Results and Benchmarks

| Objective             | Expected Accuracy | Expected Performance         | Remarks                                      |
|-----------------------|-------------------|------------------------------|----------------------------------------------|
| Facial Recognition    | 95–98%            | < 1 s/face, up to 25 FPS       | Optimized with Hailo-8L |
| Object Detection      | 80–90%            | 5–25 FPS depending on model  | Depends on the scenario and environment      |
| Voice Recognition     | 85–95%            | < 1 s to 2–5 s               | Short commands, quality microphone           |
| Speech Synthesis      | Natural voice     | Real-time                    | Prioritize responsiveness                    |

## Developpment Process

[Developpment Process Guide](DEV_PROCESS.md)

## Contributors

- Christophe Cavigneaux
