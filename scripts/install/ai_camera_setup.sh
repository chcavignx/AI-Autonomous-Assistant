#!/bin/bash
# Raspberry Pi AI Camera Installation Script

echo "=== Raspberry Pi AI Camera Setup Script ==="
echo "Ensure AI Camera is physically connected before running"

# Step 1: Update system
echo "Step 1: Updating system..."
sudo apt update && sudo apt full-upgrade -y

# Step 2: Install IMX500 firmware and models
echo "Step 2: Installing IMX500 firmware..."
sudo apt install -y imx500-all

# Step 3: Install picamera2 with GUI support
echo "Step 3: Installing Picamera2..."
sudo apt install -y python3-picamera2
pip3 install "picamera2[gui]"

# Step 4: Install additional dependencies
echo "Step 4: Installing additional dependencies..."
sudo apt install -y python3-opencv python3-numpy python3-matplotlib

# Step 5: Verify firmware installation
echo "Step 5: Checking firmware files..."
if [ -f "/lib/firmware/imx500_loader.fpk" ] && [ -f "/lib/firmware/imx500_firmware.fpk" ]; then
    echo "âœ“ IMX500 firmware files found"
else
    echo "âœ— IMX500 firmware files missing"
fi

# Check model files
if [ -d "/usr/share/imx500-models/" ]; then
    echo "âœ“ AI models directory found"
    echo "Available models:"
    ls /usr/share/imx500-models/
else
    echo "âœ— AI models directory missing"
fi

echo "AI Camera setup complete! Reboot recommended."
echo "After reboot, run: python3 ai_camera_verification.py"
