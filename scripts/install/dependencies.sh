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

# Install vision dependencies for raspberry pi 5 camera (USB) and AI camera
sudo apt-get install -y \
python3-picamera2
imx500-all
python3-opencv
python3-munkres
python3-numpy
python3-matplotlib


# Installation Hailo-8L sur Raspberry Pi 5 see "https://www.raspberrypi.com/documentation/computers/ai.html"
sudo apt install -y hailo-all
