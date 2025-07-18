"""
Unit tests for synthetic data generation and loading (Phase 1)
"""
import numpy as np
from src.data_processing.synthetic import generate_neural_data, load_data

def test_generate_neural_data_shape():
    data = generate_neural_data(num_channels=32, num_samples=1000)
    assert data.shape == (32, 1000)

def test_load_data_valid():
    data = load_data('tests/test_data/sample.h5')
    assert data is not None
