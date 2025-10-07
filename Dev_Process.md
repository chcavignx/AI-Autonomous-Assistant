# Developpment Process

## Priorities to Establish in Order to Meet Technical and Temporal Constraints

To maximize the chances of success and avoid dispersion, it is essential to prioritize objectives based on technical complexity, impact on the AI assistant, and available time.
Here is a prioritization proposed for the context: Raspberry Pi 5, Hailo-8L, NVMe SSD, official camera, and USB microphone.

### 1. **Top Priority: Essential and Feasible Modules**

- **Offline Voice Recognition (USB Microphone)**
  - *Why?* It is the core of user interaction, quick to set up, and resource-efficient.
  - *Technical Impact:* Low to moderate, with comprehensive documentation, abundant examples, and existing projects.
  - *Estimated Time:* 1 to 3 days for a proof of concept.

- **Offline Speech Synthesis**
  - *Why?* Provides immediate audio feedback, easy to install, with natural-sounding voices.
  - *Technical Impact:* Low, with direct integration into Python.
  - *Estimated Time:* 1 to 2 days to test and integrate.

### 2. **Intermediate Priority: Computer Vision**

- **Facial Recognition (face_recognition + official camera)**
  - *Why?* Adds a layer of personalization and security, but requires more resources and optimization.
  - *Technical Impact:* Moderate; requires proper lighting and testing on a reduced dataset.
  - *Estimated Time:* 3 to 5 days for reliable detection on a few faces.

- **Object Recognition**
  - *Why?* Advanced functionality, but more complex to integrate and optimize.
  - *Technical Impact:* High; requires model management, hardware acceleration, and extensive testing.
  - *Estimated Time:* 5 to 8 days for smooth detection of common objects.

### 3. **Secondary Priority: Integration and User Experience**

- **Module Fusion (main script, command management)**
  - *Why?* Necessary for a coherent assistant, but to be executed once the basic modules are functional.
  - *Technical Impact:* Variable, depending on the complexity of the desired interface.
  - *Estimated Time:* 3 to 5 days for a basic integration.

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

- **Audio Modules** (voice recognition and speech synthesis): they are quick to deploy and validate basic interaction.
- **Follow with Vision Modules**:
    1. Start with facial recognition (simpler than object recognition).
    2. Proceed to object detection.
- **Test each module independently** before attempting integration.
- **Document each step and design decision** to avoid delays during iterations or corrections.
- **Adhere to minimum criteria**: define clear milestones (e.g., "voice command functional", "facial recognition operational") to measure progress.

### Sequenced Guides for Different Modules

- **Audio Modules**
    1. [Offline Speech-to-Text (STT) Guide](STT_offline.md)
    2. [Offline Text-to-Speech (TTS) Guide](TTS_offline.md)
    3. [Offline Speech-to-Speech demo](demo/voice_agent_offline.md)
        Demo application that listens to voice input from the microphone and responds with speech for specific intents. (**TO DO**)

- **Vision Modules**
    1. [Facial Recognition Guide](facial_recognition.md **TO DO**)
    2. [Object Recognition Guide](object_recognition.md **TO DO**)
    3. [Object Recognition demo](object_recognition_demo.md **TO DO**)
    Demo application that detects and labels objects in real-time. (**TO DO**)

- **Module Integration**

- **Design Decision**
  1. Voice Agent Solution and Architecture: [Voice Agent Offline Solution](demo/STS_VAD_models.md)