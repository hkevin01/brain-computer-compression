"""
BCI Compression Toolkit

A comprehensive toolkit for developing and benchmarking compression algorithms
specifically designed for neural data streams in brain-computer interfaces.
"""

__version__ = "1.0.0"
__author__ = "BCI Compression Team"
__email__ = "contact@bci-compression.org"

# Core modules (no external dependencies)
from .core import NeuralCompressor, load_neural_data

# Import modules with dependencies conditionally
try:
    _has_signal_processing = True
except ImportError:
    _has_signal_processing = False

try:
    _has_neural_decoder = True
except ImportError:
    _has_neural_decoder = False

try:
    _has_data_acquisition = True
except ImportError:
    _has_data_acquisition = False

# Base exports
__all__ = [
    "NeuralCompressor",
    "load_neural_data",
]

# Add conditional exports
if _has_signal_processing:
    __all__.append("NeuralSignalProcessor")

if _has_neural_decoder:
    __all__.extend([
        "NeuralDecoder",
        "MotorImageryDecoder",
        "RealTimeDecoder",
        "DeviceController",
        "create_motor_imagery_system",
    ])

if _has_data_acquisition:
    __all__.extend([
        "BaseDataAcquisition",
        "SimulatedDataAcquisition",
        "FileDataAcquisition",
        "DataAcquisitionManager",
        "create_test_acquisition_system",
    ])

# bci_compression package init
