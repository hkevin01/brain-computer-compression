"""
Unit tests for synthetic data generation (Phase 1)
"""
import numpy as np
from bci_compression.data_processing.synthetic import generate_synthetic_neural_data

def test_generate_synthetic_neural_data_shape():
    data, meta = generate_synthetic_neural_data(num_channels=32, n_samples=1000)
    assert data.shape == (32, 1000)
    assert isinstance(meta, dict)
