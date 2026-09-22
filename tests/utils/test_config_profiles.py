"""Unit tests for modular configuration profiles and vision config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.utils.config import Config, load_config

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def test_load_default_config_with_vision_profile() -> None:
    """Test default config.yaml loads with vision_config properly merged."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.vision is not None
    assert cfg.vision.object_model_type in {"yolo", "yolo_hailo", "yolo_imx500", "yolo26_ncnn", "libreyolo"}
    assert cfg.vision.face_detector_type in {"insightface", "cascade", "hailo", "imx500"}


def test_load_each_vision_profile() -> None:
    """Test each vision profile can be loaded directly and has valid schema."""
    profiles = [
        ("config/vision/config_vision_hailo.yaml", "yolo_hailo", "hailo"),
        ("config/vision/config_vision_cpu.yaml", "yolo", "cascade"),
        ("config/vision/config_vision_imx500.yaml", "yolo_imx500", "imx500"),
        ("config/vision/config_vision_ncnn.yaml", "yolo26_ncnn", "cascade"),
        ("config/vision/config_vision_libreyolo.yaml", "libreyolo", "cascade"),
    ]

    for profile_path, expected_obj, expected_face in profiles:
        abs_p = ROOT_DIR / profile_path
        assert abs_p.exists(), f"Profile {profile_path} does not exist"

        # Load yaml directly to test schema
        with abs_p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["object_model_type"] == expected_obj
        assert data["face_detector_type"] == expected_face

        # Test loading via dummy config using this profile
        cfg = Config.from_mapping({"vision": data})
        assert cfg.vision.object_model_type == expected_obj
        assert cfg.vision.face_detector_type == expected_face


def test_vision_config_inline_override(tmp_path: Path) -> None:
    """Test that inline vision keys in config.yaml override the included profile."""
    custom_cfg_file = tmp_path / "custom_config.yaml"
    custom_cfg_file.write_text(
        yaml.dump(
            {
                "vision_config": "config/vision/config_vision_hailo.yaml",
                "vision": {
                    "object_recognition_threshold": 0.88,
                    "object_model_name": "custom_yolo.hef",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(custom_cfg_file)
    # Overridden fields
    assert cfg.vision.object_recognition_threshold == pytest.approx(0.88)
    assert cfg.vision.object_model_name == "custom_yolo.hef"
    # Inherited fields from profile
    assert cfg.vision.object_model_type == "yolo_hailo"
    assert cfg.vision.face_detector_type == "hailo"


def test_generic_includes_merging(tmp_path: Path) -> None:
    """Test generic includes list merging in YAML config."""
    sub1 = tmp_path / "sub1.yaml"
    sub1.write_text(yaml.dump({"audio": {"volume": 0.4, "input_sample_rate": 16000}}), encoding="utf-8")

    main_cfg = tmp_path / "main.yaml"
    main_cfg.write_text(
        yaml.dump(
            {
                "includes": [str(sub1)],
                "audio": {"volume": 0.9},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(main_cfg)
    assert cfg.audio.volume == pytest.approx(0.9)
    assert cfg.audio.input_sample_rate == 16000
