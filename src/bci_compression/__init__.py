"""
BCI Compression Toolkit

A comprehensive toolkit for developing and benchmarking compression algorithms
specifically designed for neural data streams in brain-computer interfaces.
"""

__version__ = "1.0.0"
__author__ = "BCI Compression Team"
__email__ = "contact@bci-compression.org"

# Import key components to the top level
from .core import Compressor
from .data_io import load_neural_data
from .plugins import get_plugin, register_plugin
from .utils.configuration import load_config, setup_logging

__all__ = [
    "Compressor",
    "load_neural_data",
    "get_plugin",
    "register_plugin",
    "load_config",
    "setup_logging",
]

# Initialize logging when the package is imported
setup_logging()
