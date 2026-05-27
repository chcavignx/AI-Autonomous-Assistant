# Audio Library Integration Tests

This directory contains **integration tests for the audio library**, designed to validate hardware availability and functionality in deployment environments.

## Philosophy

These tests follow a **simple-first, build-progressively** approach:

1. **Hardware Detection** - Can we find audio devices?
2. **Stream Control** - Can we open/close audio streams?
3. **Playback** - Can we play audio through output?
4. **Recording** - Can we capture audio from input?
5. **Recorder Standalone** - Can the simplified AudioRecorder capture and format chunks?
6. **ASR Integration** - Can the high-level ASREngine capture and transcribe chunks?
7. **ASR with TTS** - Can ASREngine transcribe TTS audio output?
8. **ASR Recording Validation** - Can we transcribe microphone recordings?
9. **Wake Word Standalone** - Can the WakeWordDetector load models and capture audio in background?
10. **VAD Standalone** - Can the deep learning Silero VAD engine detect speech segments and handle resampling?
11. **TTS Lifecycle & Utils** - Can we manage non-blocking TTS, interruptions, and utility conversions?

Each test is **self-contained and can run independently** on any system with audio hardware.

---

## Test Descriptions

### 1. Hardware Detection (`test_hardware_detection.py`)

**Purpose:** Verify that PyAudio can detect audio devices.

**What it tests:**

- Is PyAudio functional?
- Are any audio input/output devices available?
- Lists all detected devices (name, channels, index)

**Run:**

```bash
python examples/test_hardware_detection.py
```

**Expected output:**

```bash
✓ Total audio devices detected: 4
Input devices (2):
  [0] USB Device (1 channels)
  [2] default (64 channels)
Output devices (2):
  [1] USB Device (2 channels)
  [3] default (64 channels)
✓ PASSED: Audio hardware detected
```

**When it skips:** Never (unless PyAudio is broken).

---

### 2. Stream Open/Close (`test_stream_open_close.py`)

**Purpose:** Validate that audio streams can be properly configured and controlled.

**What it tests:**

- Can we open an input stream?
- Can we open an output stream?
- Can we read/write data?
- Does stopping work correctly?

**Run:**

```bash
python examples/test_stream_open_close.py
```

**Key validations:**

- Tries multiple sample rates (16kHz, 44.1kHz, 48kHz) to find device compatibility
- Verifies stream state (active before stop, inactive after)
- Reads one chunk from input, writes one chunk to output

**Skips if:** No input or output devices available.

---

### 3. Audio Playback (`test_playback.py`)

**Purpose:** Verify that audio data can be sent to output devices.

**What it tests:**

- Generate a sine wave (440 Hz)
- Send it to output stream
- Also tests silence playback (zeros)
- **New:** Validates `AudioPlayer.play_file()` with real WAV files.

**Run:**

```bash
python examples/test_playback.py
```

**Key validations:**

- Stream opens successfully
- Data writes without errors
- Stream responds to stop command
- File-based playback routes correctly through backend

**Skips if:** No output device available.

**Note:** You may hear an *audible 440 Hz tone* for ~2 seconds if speakers are enabled. Use `--mute` or disconnect speakers to silence it.

---

### 4. Audio Recording (`test_recording.py`)

**Purpose:** Verify microphone input and WAV file creation.

**What it tests:**

- Record 2 seconds from microphone
- Save to WAV file
- Read WAV file back and verify format
- Re-play the recorded audio

**Run:**

```bash
python examples/test_recording.py
```

**Key validations:**

- Input stream opens
- Data is captured (non-zero file size)
- WAV file is valid
- Recorded audio can be read and played back

**Output:** Saves test recordings to `/tmp/test_recording.wav` and `/tmp/test_record_playback.wav`

**Skips if:** No input or output device available.

---

### 5. Recorder Standalone (`test_recorder_standalone.py`)

**Purpose:** Validate the simplified `AudioRecorder` utility class.

**What it tests:**

- Can the recorder start/stop cleanly?
- Does `read_numpy()` return valid 16-bit PCM?
- Does `read_float()` return normalized [-1, 1] data?

**Run:**

```bash
python examples/test_recorder_standalone.py
```

---

### 6. ASR Engine Integration (`test_asr_integration.py`)

**Purpose:** Verify the full ASR pipeline using the production `ASREngine`.

**What it tests:**

- Loads the ASR model (Whisper/Faster-Whisper)
- Validates that the engine thread captures audio chunks
- Checks that the VAD (Voice Activity Detection) system is initialized
- Verifies the callback system for transcriptions

**Run:**

```bash
python examples/test_asr_integration.py
```

**Skips if:** No input device available.

---

### 7. ASR with TTS (`test_asr_with_tts.py`)

**Purpose:** Verify that the high-level `ASREngine` can accurately transcribe speech generated dynamically by the `TTSEngine` (loopback integration).

**What it tests:**

- Generates voice samples via PIPER `TTSEngine`
- Feeds synthesized WAV bytes directly to the Whisper `ASREngine`
- Asserts that original text matches transcribed text within word-error tolerance thresholds

**Run:**

```bash
python examples/test_asr_with_tts.py
```

---

### 8. ASR Recording Validation (`test_asr_recording_validation.py`)

**Purpose:** Validate dynamic user recording capture and transcription pipeline.

**What it tests:**

- Records 3 seconds of active microphone input
- Streams recording frames to `ASREngine`
- Returns real-time speech transcription to verify complete mic-to-text pipeline

**Run:**

```bash
python examples/test_asr_recording_validation.py
```

**Skips if:** No input device available.

---

### 9. Wake Word Standalone (`test_wake_word_standalone.py`)

**Purpose:** Validate the `WakeWordDetector` engine in isolation.

**What it tests:**

- Loads openWakeWord models
- Spawns background capture and detection threads
- Verifies callback triggers upon detection

**Run:**

```bash
python examples/test_wake_word_standalone.py
```

---

### 10. VAD Standalone Flow (`test_vad_standalone.py`)

**Purpose:** Validate voice activity detection state machines and audio sample rate converters.

**What it tests:**

- Initializes standalone `VADEngine` with deep-learning Silero VAD
- Loads a real speech sample (`data/test.wav`) to verify positive speech identification (`is_speech_detected() -> True`) and correct voice active timestamps/segments (`get_speech_segments()`)
- Tests negative identification against synthetic silence (`is_speech_detected() -> False`)
- Tests native audio high-quality downsampling/upsampling logic (`resample_audio()`) between 16000Hz and 22050Hz

**Run:**

```bash
python examples/test_vad_standalone.py
```

---

### 11. TTS Lifecycle & Utils (`test_tts_lifecycle_and_utils.py`)

**Purpose:** Validate advanced non-blocking TTSEngine APIs and low-level utility operations.

**What it tests:**

- Initializes and loads PIPER `TTSEngine` with active local models
- Tests asynchronous dynamic speech generation (`speak(..., blocking=False)`)
- Tests state queries (`is_speaking`) and queue interruption/abort triggers (`interrupt()`)
- Tests synchronization blocks (`wait()`) and teardowns (`unload()`)
- Asserts correctness of untested `audio_utils.py` functions: `convert_to_float32()`, `convert_to_int16()`, `resolve_device_index()`, `suppress_pa_stderr()`, and `install_alsa_error_handler()`

**Run:**

```bash
python examples/test_tts_lifecycle_and_utils.py
```

---

## Run All Tests

```bash
python examples/run_all_audio_tests.py
```

Runs all 11 tests in sequence with a summary report.

---

## Interpreting Results

### PASSED ✓

- All checks succeeded
- Hardware is functional and properly configured

### SKIPPED ⊘

- Test was not applicable (e.g., no output device for playback test)
- Not a failure—just indicates that part of hardware isn't available

### FAILED ✗

- A required check failed
- Indicates a real issue with audio setup or hardware

---

## Common Issues & Fixes

### "No input-capable audio device detected"

- Check that a microphone is connected and recognized by the OS
- On Linux: `arecord -l` to list input devices
- On macOS: `System Preferences > Sound > Input`

### "No output device available"

- Check that speakers/headphones are connected
- On Linux: `aplay -l` to list output devices

### "Invalid sample rate" error

- Certain USB devices don't support all sample rates
- The test automatically tries multiple rates and picks one that works

### ALSA/Jack warnings

- These are debug messages from the audio stack
- Do **not** indicate test failure
- Safe to ignore if test shows "✓ PASSED"

---

## Integration with CI/CD

Use these tests for **deployment health checks**:

```bash
#!/bin/bash
# health_check.sh - Verify audio subsystem on deployment

python examples/test_hardware_detection.py || exit 1
python examples/test_stream_open_close.py || exit 1
echo "Audio subsystem healthy"
```

---

## Next Steps After Passing

If all tests pass:

1. **Run existing unit tests:**

   ```bash
   pytest tests/audio/ -v
   ```

2. **Test the full ASR engine:**

   ```bash
   python examples/check_audio_flow.py
   ```

3. **Test TTS playback:**

   ```bash
   python examples/TTS/test_piper_tts.py  # if available
   ```

---

## Architecture

Each test:

- **Imports minimal dependencies** (PyAudio + numpy)
- **Cleans up resources** (closes streams, terminates PyAudio)
- **Reports clearly** (✓/✗/⊘ status, logs device info)
- **Handles errors** gracefully (timeouts, missing devices, bad sample rates)

No external fixtures or pytest machinery—they run standalone with `python`.

---

## Development

To add a new integration test:

1. Create `test_new_feature.py` in this directory
2. Follow the structure:
   - Single clear test function
   - Print progress with status symbols (✓/✗/⊘)
   - Exit with code 0 (success) or 1 (failure)
3. Add to `run_all_audio_tests.py` in the `TESTS` list
4. Run manually first: `python examples/test_new_feature.py`

---

## Hardware Tested

These tests have been validated on:

- **USB Audio Devices** (encoding/microphone)
- **Pipewire audio stack** (Linux)
- **ALSA** (Linux - raw kernel audio)
- **Combinations:** Default playback + USB input

---

## Questions?

For issues:

1. Run `test_hardware_detection.py` first to confirm devices are visible
2. Check OS audio settings (volume, input device selection)
3. Review logs from the specific failing test
4. Open an issue with test output and device info
