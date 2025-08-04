import os
import sys

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_neural_data():
    """Generate consistent test neural data"""
    np.random.seed(42)
    return np.random.randn(32, 10000)


@pytest.fixture
def small_neural_data():
    """Generate small test data for quick tests"""
    np.random.seed(42)
    return np.random.randn(8, 1000)
