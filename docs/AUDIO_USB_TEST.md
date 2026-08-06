# Tutorial: Testing a USB Microphone and a USB Speaker on Raspberry Pi 5

This guide helps you verify the basic audio path used by the repository audio stack:

- USB microphone capture
- Speaker playback
- ALSA and PyAudio device visibility
- Optional device pinning in `config.yaml`

The audio engines in `src/audio` can resample device audio in software when the hardware sample rate does not match the configured model rate, so exact hardware matching is helpful but not mandatory.

## Hardware Setup

- Raspberry Pi 5 has no built-in analog audio jack
- Connect your USB microphone and/or USB speaker or sound card
- Confirm that the devices appear in ALSA and, if applicable, in PipeWire

## Useful Commands

- `lsusb` to list USB devices
- `aplay -l` and `aplay -L` to list playback devices
- `arecord -l` and `arecord -L` to list capture devices
- `speaker-test` to test output
- `arecord` to test microphone input
- `aplay` to play a recorded file
- `alsamixer` to adjust input and output levels

## Step By Step

1. Plug in the USB microphone and speaker.
2. Confirm the devices are visible.

   ```bash
   lsusb
   aplay -l
   arecord -l
   ```

3. Install the ALSA tools if needed.

   ```bash
   sudo apt update
   sudo apt install -y alsa-utils
   ```

4. Test speaker output.

   ```bash
   speaker-test -c2 -t wav
   ```

5. Test microphone input.

   ```bash
   arecord -f cd -d 5 test.wav
   ```

6. Play the recording back.

   ```bash
   aplay test.wav
   ```

7. Adjust levels if necessary.

   ```bash
   alsamixer
   ```

   - Press `F6` to select the card
   - Raise or lower capture and playback levels as needed

## Device Selection In The Project

The current audio config supports explicit device indices:

- `audio.input_device_index`
- `audio.output_device_index`

If you need to pin a device, identify the correct ALSA card first and then set the matching index in `config.yaml`.

## Optional ALSA Default Routing

If you want to set a default ALSA route for testing, you can create `~/.asoundrc` with a simple playback and capture mapping. Adjust the `plughw` entries to match your hardware.

```bash
pcm.!default {
type asym
playback.pcm "plughw:1,0"
capture.pcm "plughw:0,0"
}
```

## Troubleshooting

- No device listed: reconnect the USB hardware and check `lsusb`
- No capture in `arecord`: verify the mic is selected as the input device
- Low volume or clipping: adjust levels in `alsamixer`
- Wrong device chosen by default: set `audio.input_device_index` and `audio.output_device_index`

## Automation Script

The repository includes a small shell script for a basic record-and-playback check:

- `scripts/tests/audio_test.sh`

It records five seconds of audio and plays it back locally.
