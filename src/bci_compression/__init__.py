"""
BCI Compression Toolkit

A comprehensive toolkit for developing and benchmarking compression algorithms
specifically designed for neural data streams in brain-computer interfaces.
"""

__version__ = "0.1.0"
__author__ = "BCI Compression Team"
__email__ = "contact@bci-compression.org"

from .core import NeuralCompressor, load_neural_data
from .algorithms import *
from .data_processing import *

__all__ = [
    "NeuralCompressor",
    "load_neural_data",
]
