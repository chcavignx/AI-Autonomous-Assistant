"""Basic test to verify pytest is working."""

import sys

import pytest


@pytest.mark.basic
def test_basic():
    """Basic assertion to verify test infrastructure works."""
    assert True


@pytest.mark.basic
def test_imports():
    """Test that we can import basic Python packages."""

    assert sys.version_info >= (3, 10)


@pytest.mark.integration
def test_integration():
    """Test that we can import basic Python packages."""
    assert True
