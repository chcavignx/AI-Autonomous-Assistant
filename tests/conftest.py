from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

# Ensure repository root is on sys.path so imports like `src.audio.*` work in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_marker_only_run(config: pytest.Config, marker: str) -> bool:
    """Return True when the session is filtered exclusively to a single marker."""
    try:
        expr: str = getattr(config.option, "markexpr", "") or ""
        return expr.strip() == marker
    except (AttributeError, TypeError):
        return False


def _is_audio_or_llm_run(config: pytest.Config) -> bool:
    """Return True when pytest was invoked targeting only tests/audio or tests/llm."""
    try:
        args = config.args or []
        targets = {"tests/audio", "tests/audio/", "tests/llm", "tests/llm/"}
        return any(any(str(a).endswith(t) for t in targets) for a in args)
    except (AttributeError, TypeError):
        return False


def _disable_cov_fail_under(config: pytest.Config) -> None:
    """Zero out the coverage fail-under threshold on the pytest-cov plugin object."""
    # pytest-cov stores the threshold at plugin.options.cov_fail_under (an argparse Namespace).
    # config.option.cov_fail_under is a belt-and-suspenders copy; patch both.
    plugin = config.pluginmanager.get_plugin("_cov")
    if plugin is not None:
        if hasattr(plugin, "options") and hasattr(plugin.options, "cov_fail_under"):
            plugin.options.cov_fail_under = 0
        if hasattr(plugin, "cov_fail_under"):  # older versions store it directly
            plugin.cov_fail_under = 0
    if hasattr(config, "option") and hasattr(config.option, "cov_fail_under"):
        config.option.cov_fail_under = 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Relax coverage threshold when only running audio/llm or basic-marker tests."""
    try:
        audio_or_llm_only = all("tests/audio" in str(item.fspath) or "tests/llm" in str(item.fspath) for item in items)
    except (AttributeError, TypeError):
        return
    if audio_or_llm_only or _is_marker_only_run(config, "basic"):
        _disable_cov_fail_under(config)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Disable coverage gate just before pytest-cov enforces it at terminal_summary."""
    config = session.config
    if _is_audio_or_llm_run(config) or _is_marker_only_run(config, "basic"):
        _disable_cov_fail_under(config)
