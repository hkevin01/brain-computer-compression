"""
BCI Compression Toolkit - Neural Data Compression for Brain-Computer Interfaces

A state-of-the-art toolkit for compressing neural data from brain-computer interfaces,
with a focus on real-time performance, signal fidelity, and mobile optimization.

Features:
- Neural-optimized compression algorithms
- Real-time processing with GPU acceleration
- Mobile-optimized implementations
- Comprehensive benchmarking framework
- Extensible plugin system
"""

__version__ = "0.8.0"
__author__ = "Kevin"
__email__ = "contact@bci-compression.org"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Kevin"

# Import core modules with error handling
try:
    from .core import Compressor, create_compressor
    from .algorithms import (
        create_neural_lz_compressor,
        create_neural_arithmetic_coder,
        PerceptualQuantizer,
        create_predictive_compressor,
        create_context_aware_compressor,
        create_gpu_compression_system,
    )
    from .mobile import (
        MobileBCICompressor,
        MobileStreamingPipeline,
        PowerOptimizer,
        MobileMetrics,
    )
    from .plugins import get_plugin, register_plugin
    from .utils.configuration import load_config, setup_logging
    from .data_processing import load_neural_data
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")

# Package exports
__all__ = [
    # Core functionality
    "Compressor",
    "create_compressor",
    "load_neural_data",

    # Compression algorithms
    "create_neural_lz_compressor",
    "create_neural_arithmetic_coder",
    "PerceptualQuantizer",
    "create_predictive_compressor",
    "create_context_aware_compressor",
    "create_gpu_compression_system",

    # Mobile optimization
    "MobileBCICompressor",
    "MobileStreamingPipeline",
    "PowerOptimizer",
    "MobileMetrics",

    # Plugin system
    "get_plugin",
    "register_plugin",

    # Configuration
    "load_config",
    "setup_logging",
]

# Initialize logging
setup_logging()


def get_version():
    """Return the current version of the package."""
    return __version__
