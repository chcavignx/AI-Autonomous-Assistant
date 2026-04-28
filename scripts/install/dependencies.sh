#!/bin/bash
set -euo pipefail

# Install dependencies
sudo apt-get install -y \
ffmpeg
python3
python3-pip
python3-all-dev
git
portaudio19-dev
python3-pyaudio
alsa-utils
pipewire-alsa
espeak-ng
libspeexdsp-dev

# Installation Hailo-8L sur Raspberry Pi 5
sudo apt install -y hailo-all
