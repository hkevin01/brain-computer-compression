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
from .neural_decoder import (
    NeuralDecoder, 
    MotorImageryDecoder, 
    RealTimeDecoder,
    DeviceController,
    create_motor_imagery_system
)
from .data_acquisition import (
    BaseDataAcquisition,
    SimulatedDataAcquisition,
    FileDataAcquisition,
    DataAcquisitionManager,
    create_test_acquisition_system
)

__all__ = [
    "NeuralCompressor",
    "load_neural_data",
    # Neural decoding
    "NeuralDecoder",
    "MotorImageryDecoder",
    "RealTimeDecoder",
    "DeviceController",
    "create_motor_imagery_system",
    # Data acquisition
    "BaseDataAcquisition",
    "SimulatedDataAcquisition",
    "FileDataAcquisition",
    "DataAcquisitionManager",
    "create_test_acquisition_system",
]
