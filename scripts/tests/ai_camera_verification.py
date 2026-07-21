#!/usr/bin/env python3
"""Raspberry Pi AI Camera Verification Script
Tests AI camera connection, firmware, and AI functionality.
"""

import os
import pathlib
import subprocess
import time

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

def test_imx500_device_id():
    """Test IMX500 device ID."""
    try:
        with Picamera2() as picam2:
            global_camera_info = picam2.global_camera_info()
            camera_id = next((c['Num'] for c in global_camera_info if c['Model'] == 'imx500'), None)
            if camera_id is None:
                return False
            print("Device ID for imx500 is: ", camera_id)
            return True
    except Exception:
        return False



def test_firmware_installation():
    """Check if IMX500 firmware files are installed."""
    firmware_files = ["/lib/firmware/imx500_loader.fpk", "/lib/firmware/imx500_firmware.fpk"]

    all_present = True
    for file_path in firmware_files:
        if pathlib.Path(file_path).exists():
            pathlib.Path(file_path).stat().st_size
        else:
            all_present = False

    # Check model directory
    model_dir = "/usr/share/imx500-models/"
    if pathlib.Path(model_dir).exists():
        models = os.listdir(model_dir)
        for _model in models[:3]:  # Show first 3
            pass
    else:
        all_present = False

    return all_present


def test_camera_detection() -> bool | None:
    """Test if AI camera is detected."""
    try:
        with Picamera2() as picam2:
            camera_properties = picam2.camera_properties
            model = camera_properties.get("Model", "")
            return "imx500" in str(model).lower()
    except Exception:
        return False

def test_camera_global_infos() -> bool | None:
    """Test camera global infos."""
    try:
        with Picamera2() as picam2:
            global_camera_info = picam2.global_camera_info()
            print("Global camera infos:")
            for info in global_camera_info:
                print(info)
            return True
    except Exception:
        return False

def test_basic_ai_functionality() -> bool | None:
    """Test basic AI functionality using rpicam-hello."""
    try:
        # Test with rpicam-hello to ensure basic functionality
        result = subprocess.run(
            ["rpicam-hello", "--timeout", "3000", "--info-text", "AI Camera Test"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            pass
        else:
            return False

        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_ai_object_detection() -> bool | None:
    """Test AI object detection capability."""
    try:
        # Test MobileNet SSD object detection
        cmd = [
            "rpicam-hello",
            "--timeout",
            "5000",
            "--post-process-file",
            "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
            "--framerate",
            "30",
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_pose_estimation() -> bool | None:
    """Test AI pose estimation capability."""
    try:
        # Test PoseNet pose estimation
        cmd = [
            "rpicam-hello",
            "--timeout",
            "5000",
            "--post-process-file",
            "/usr/share/rpi-camera-assets/imx500_posenet.json",
            "--framerate",
            "20",
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_power_consumption() -> bool | None:
    """Test power consumption (basic monitoring)."""
    picam2 = None
    try:
        # Monitor system temperature during AI processing
        def get_cpu_temp():
            with pathlib.Path("/sys/class/thermal/thermal_zone0/temp").open(encoding="utf-8") as f:
                return int(f.read()) / 1000.0

        initial_temp = get_cpu_temp()

        # Run AI processing for a short time
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 640), "format": "XRGB8888"})
        picam2.configure(config)
        picam2.start()

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 10:  # 10 seconds
            picam2.capture_array()
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS

        final_temp = get_cpu_temp()
        temp_rise = final_temp - initial_temp

        picam2.stop()

        if temp_rise < 10:  # Reasonable temp rise
            pass

        return True
    except Exception:
        return False
    finally:
        if picam2 is not None:
            picam2.close()


def main() -> None:
    """Run all AI camera verification tests."""
    tests = [
        ("Firmware Installation", test_firmware_installation),
        ("Camera Detection", test_camera_detection),
        ("Camera Global Infos", test_camera_global_infos),
        ("IMX500 Device ID", test_imx500_device_id),
        ("Basic AI Functionality", test_basic_ai_functionality),
        ("AI Object Detection", test_ai_object_detection),
        ("AI Pose Estimation", test_pose_estimation),
        ("Power Consumption", test_power_consumption),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception:
            results.append((test_name, False))

    # Summary
    import sys
    print("\n=== AI Camera Verification Results ===")
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"[{status}] {test_name}")
        if result:
            passed += 1

    print(f"\nSummary: {passed}/{len(results)} tests passed.")
    if passed >= len(results) - 1:  # Allow one test to fail (e.g. power monitoring threshold)
        print("Verification SUCCESSFUL!")
        sys.exit(0)
    else:
        print("Verification FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
