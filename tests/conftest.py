from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so imports like `src.audio.*` work in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """Relax coverage threshold when only running audio tests."""
    try:
        audio_only = all("tests/audio" in str(item.fspath) for item in items)
    except (AttributeError, TypeError):
        return
    if audio_only and hasattr(config.option, "cov_fail_under"):
        config.option.cov_fail_under = 0


def pytest_sessionstart(session):
    """Also relax coverage when pytest is invoked with tests/audio explicitly."""
    config = session.config
    try:
        args = config.args or []
        audio_arg = any(str(arg).endswith("tests/audio") or str(arg).endswith("tests/audio/") for arg in args)
    except (AttributeError, TypeError):
        audio_arg = False
    if audio_arg and hasattr(config.option, "cov_fail_under"):
        config.option.cov_fail_under = 0


def pytest_configure(config):
    """Disable coverage gate when running only the audio test suite."""
    try:
        args = config.args or []
        audio_arg = any(str(arg).endswith("tests/audio") or str(arg).endswith("tests/audio/") for arg in args)
    except (AttributeError, TypeError):
        audio_arg = False
    if audio_arg and hasattr(config.option, "cov_fail_under"):
        config.option.cov_fail_under = 0
