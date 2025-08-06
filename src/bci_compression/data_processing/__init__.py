"""
Data processing utilities for neural data.
Includes both CPU and GPU-accelerated implementations.
"""

from .filters import apply_bandpass_filter, apply_notch_filter
from .signal_processing import NeuralSignalProcessor
from .synthetic import generate_synthetic_neural_data

# Temporarily disable GPU imports to avoid CUDA dependency issues
# try:
#     from .gpu_processing import GPUNeuralProcessor
#     from .gpu_data_loader import GPUDataLoader
#     CUDA_AVAILABLE = True
# except ImportError:
#     # Fallback for when CUDA is not available
#     GPUNeuralProcessor = None
#     GPUDataLoader = None
#     CUDA_AVAILABLE = False

CUDA_AVAILABLE = False
GPUNeuralProcessor = None
GPUDataLoader = None


# Add simple load_neural_data function
def load_neural_data(filepath, format='numpy'):
    """Basic neural data loader function."""
    import numpy as np
    if format == 'numpy':
        return np.load(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


__all__ = [
    "apply_bandpass_filter",
    "apply_notch_filter",
    "generate_synthetic_neural_data",
    "NeuralSignalProcessor",
    "GPUNeuralProcessor",
    "GPUDataLoader",
    "CUDA_AVAILABLE",
    "load_neural_data"
]
