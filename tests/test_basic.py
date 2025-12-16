"""Basic test to verify pytest is working."""


def test_basic():
    """Basic assertion to verify test infrastructure works."""
    assert True


def test_imports():
    """Test that we can import basic Python packages."""
    import sys

    assert sys.version_info >= (3, 10)
