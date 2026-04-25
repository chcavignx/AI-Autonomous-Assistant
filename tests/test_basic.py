"""Basic test to verify pytest is working."""

import sys

import pytest
from src import main as app_main


@pytest.mark.basic
def test_basic_sanity() -> None:
    """Verify the test infrastructure is running."""
    assert True


@pytest.mark.basic
def test_supported_python_version() -> None:
    """Verify the project runs on the supported Python versions."""
    assert sys.version_info >= (3, 10)


@pytest.mark.integration
def test_main_entrypoint_returns_none() -> None:
    """Verify the current application entrypoint is importable and callable."""
    assert app_main.main() is None
