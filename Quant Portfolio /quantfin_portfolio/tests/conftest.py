"""Pytest configuration file."""

import pytest
import numpy as np

# Set random seed for reproducibility in all tests
np.random.seed(42)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def set_random_seed():
    """Set random seed for all tests."""
    np.random.seed(42)
    yield
    # Reset after tests
    np.random.seed(None)