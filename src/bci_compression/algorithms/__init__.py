"""
Compression algorithms module with error handling and fallbacks
"""

import warnings
from typing import Any, Dict, Optional

# Import with error handling
try:
    from .neural_lz import create_neural_lz_compressor
except ImportError as e:
    warnings.warn(f"Neural LZ module not available: {e}")

    def create_neural_lz_compressor(*args, **kwargs):
        raise NotImplementedError("Neural LZ compressor not available")


try:
    from .neural_arithmetic import create_neural_arithmetic_coder
except ImportError as e:
    warnings.warn(f"Neural arithmetic module not available: {e}")

    def create_neural_arithmetic_coder(*args, **kwargs):
        raise NotImplementedError("Neural arithmetic coder not available")


try:
    from .lossy_neural import PerceptualQuantizer
except ImportError as e:
    warnings.warn(f"Lossy neural module not available: {e}")

    class PerceptualQuantizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Perceptual quantizer not available")


# Export all available functions
__all__ = [
    'create_neural_lz_compressor',
    'create_neural_arithmetic_coder',
    'PerceptualQuantizer',
]

try:
    _has_gpu_acceleration = True
except ImportError:
    _has_gpu_acceleration = False
    warnings.warn("GPU acceleration modules not available. Some features will be disabled.")

# Phase 3: Advanced techniques
try:
    _has_predictive = True
except ImportError:
    _has_predictive = False
    warnings.warn("Predictive compression algorithms not available. Some features will be disabled.")

try:
    _has_context_aware = True
except ImportError:
    _has_context_aware = False
    warnings.warn("Context-aware compression algorithms not available. Some features will be disabled.")

# Base exports
__all__ = [
    "AdaptiveLZCompressor",
    "DictionaryCompressor",
    "QuantizationCompressor",
    "WaveletCompressor",
    "AutoencoderCompressor",
]

# Add Phase 2 exports conditionally
if _has_neural_lz:
    __all__.extend([
        "NeuralLZ77Compressor",
        "MultiChannelNeuralLZ",
        "create_neural_lz_compressor"
    ])

if _has_neural_arithmetic:
    __all__.extend([
        "NeuralArithmeticModel",
        "NeuralArithmeticCoder",
        "MultiChannelArithmeticCoder",
        "create_neural_arithmetic_coder"
    ])

if _has_lossy_neural:
    __all__.extend([
        "PerceptualQuantizer",
        "AdaptiveWaveletCompressor",
        "NeuralAutoencoder",
        "create_lossy_compressor_suite"
    ])

if _has_gpu_acceleration:
    __all__.extend([
        "GPUCompressionBackend",
        "RealTimeGPUPipeline",
        "create_gpu_compression_system"
    ])

# Feature availability flags
FEATURES = {
    'neural_lz': _has_neural_lz,
    'neural_arithmetic': _has_neural_arithmetic,
    'lossy_neural': _has_lossy_neural,
    'gpu_acceleration': _has_gpu_acceleration
}

# At the end, aggregate missing features and warn if any are missing
missing_features = []
if not _has_base_lossy:
    missing_features.append("Lossy compression")
if not _has_autoencoder:
    missing_features.append("Deep learning compression")
if not _has_neural_lz:
    missing_features.append("Neural LZ compression")
if not _has_neural_arithmetic:
    missing_features.append("Neural arithmetic coding")
if not _has_lossy_neural:
    missing_features.append("Advanced lossy neural compression")
if not _has_gpu_acceleration:
    missing_features.append("GPU acceleration")
if not _has_predictive:
    missing_features.append("Predictive compression")
if not _has_context_aware:
    missing_features.append("Context-aware compression")
if missing_features:
    warnings.warn(f"The following features are unavailable due to missing dependencies: {', '.join(missing_features)}.")

# algorithms package init
