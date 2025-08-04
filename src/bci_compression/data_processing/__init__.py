"""
Data processing utilities for neural data.
Includes both CPU and GPU-accelerated implementations.
"""

from .filters import apply_bandpass_filter, apply_notch_filter
from .synthetic import generate_synthetic_neural_data
from .signal_processing import NeuralSignalProcessor
from .gpu_processing import GPUNeuralProcessor
from .gpu_data_loader import GPUDataLoader

try:
    import cudf
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    "apply_bandpass_filter",
    "apply_notch_filter",
    "generate_synthetic_neural_data",
    "NeuralSignalProcessor",
    "GPUNeuralProcessor",
    "GPUDataLoader",
    "CUDA_AVAILABLE"
]
