# Tutorial: Testing a USB Microphone and a USB Sound Card on Raspberry Pi 5

This complete, step-by-step guide will help you connect, configure, and test a USB microphone and a USB sound card on a Raspberry Pi 5. It covers the key aspects you need to get your audio hardware working.

## Key Points of the Tutorial

### Hardware Setup

- **Raspberry Pi 5 specifics:** There is no built-in audio jack on the Raspberry Pi 5.
- **Connect USB audio devices:** Plug in your USB microphone and/or USB sound card.
- **Verify device detection:** Use commands to confirm devices are recognized.

### Testing and Configuration

- **Use `arecord` and `aplay` for audio tests:**
  - `arecord` is used to record audio from your microphone.
  - `aplay` is used to play back audio.

- **Adjust levels with `alsamixer`:** Launch `alsamixer` in the terminal to set microphone and output levels.

- **Audio quality testing:** Test with different audio formats for best performance.

### Advanced Technical Aspects

- **ALSA vs PipeWire:** Be aware that on Raspberry Pi OS Bookworm, PipeWire may replace or supplement ALSA. Configuration steps may differ.
- **Automation scripts:** You can create scripts to automate audio device tests.
- **Troubleshooting:** Includes common problems and their fixes.

### Essential Commands

- `lsusb` — List all connected USB devices.
- `aplay -l` and `arecord -l` — List audio playback and capture devices.
- `speaker-test` — Test speaker output.
- `arecord` — Test microphone input.
- `aplay` — Playback recorded audio.
- `alsamixer` — Adjust playback and recording levels.

## Step-by-Step Instructions

1. **Connect your USB mic and/or USB sound card to the Pi.**
2. **Check device recognition:**

    ```bash
    lsusb
    aplay -l -L
    arecord -l -L
    ```

    output example for arecord -l -L command

    ```text
    null
    Discard all samples (playback) or generate zero samples (capture)
    sysdefault
    Default Audio Device
    default
    mic
    hw:CARD=Device,DEV=0
        USB ENC Audio Device, USB Audio
        Direct hardware device without any conversions
    plughw:CARD=Device,DEV=0
        USB ENC Audio Device, USB Audio
        Hardware device with all software conversions
    sysdefault:CARD=Device
        USB ENC Audio Device, USB Audio
        Default Audio Device
    front:CARD=Device,DEV=0
        USB ENC Audio Device, USB Audio
        Front output / input
    dsnoop:CARD=Device,DEV=0
        USB ENC Audio Device, USB Audio
        Direct sample snooping device
    **** List of CAPTURE Hardware Devices ****
    card 0: Device [USB ENC Audio Device], device 0: USB Audio [USB Audio]
    Subdevices: 1/1
    Subdevice #0: subdevice #0
    ```

3. **Install ALSA utilities (if not already present):**

    ```bash
    sudo apt update
    sudo apt install alsa-utils
    ```

4. **Test audio output:**

    ```bash
    speaker-test -D plughw:1,0 -c2 -t wav
    ```

5. **Test microphone input:**

    ```bash
    arecord -D plughw:0,0 -f cd test.wav
    ```

    (Replace `0,0` with your device's card and device number from `arecord -l`)
6. **Playback your recording:**

    ```bash
    aplay -D plughw:1,0 test.wav
    ```

7. **Set levels with alsamixer:**

    ```bash
    alsamixer
    ```

    - Press F6 to select your card.
    - Adjust levels as needed.

8. **Configuration File:**

    To explicitly set a default audio device for ALSA, create or edit the `.asoundrc` file in your home directory.

    1. **Open a terminal on your Raspberry Pi.**
    2. **Create or edit `.asoundrc`:** Use a text editor (e.g., `vim`, `nano`) to open `~/.asoundrc`.
    3. **Add the following configuration:** This example sets `hw:1,0` as the default playback device. Adjust the card and device numbers as needed for your hardware.

    ```bash
    pcm.!default {
    type asym
    playback.pcm "plughw:1,0" 
    capture.pcm "plughw:0,0"
    }
    ```

    Make sure that  "plughw:1,0"  (or whatever numbers match your device) refers to a valid playback-capable device, and  "plughw:0,0"  to a valid capture-capable device.

    4. **Reboot:** Reboot your Raspberry Pi to ensure the new configuration is loaded correctly.

    ```Bash
    sudo reboot
    ```

## Troubleshooting

- **No audio devices found:** Make sure your devices are fully compatible and recognized (`lsusb`, `aplay -l`, `arecord -l`).
- **Permission issues:** Run commands with `sudo` if necessary.
- **Distorted audio:** Check levels in `alsamixer` and try different USB ports.

## Automation Example

Create and run a simple test script (save as `audio_test.sh`):

```bash
#!/bin/bash
arecord -f cd -d 5 test.wav
aplay test.wav
```

Give it executable permissions:

```bash
chmod +x audio_test.sh
./audio_test.sh
```
