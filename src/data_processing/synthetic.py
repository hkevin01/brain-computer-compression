"""
Synthetic neural data generation and loading utilities
"""
import numpy as np


def generate_neural_data(num_channels: int, num_samples: int):
    return np.random.randn(num_channels, num_samples)


def load_data(path: str):
    # Dummy loader for test purposes
    return np.ones((32, 1000))
