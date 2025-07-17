"""
Compression algorithms for neural data.

This package contains various compression algorithms optimized for
brain-computer interface applications, including both lossless and
lossy compression methods.
"""

import warnings

# Core compression algorithms
from .lossless import AdaptiveLZCompressor, DictionaryCompressor

# Try to import lossy algorithms with graceful fallback
try:
    from .lossy import QuantizationCompressor, WaveletCompressor
    _has_base_lossy = True
except ImportError:
    _has_base_lossy = False
    warnings.warn("Lossy compression algorithms (QuantizationCompressor, WaveletCompressor) not available. Some features will be disabled.")

try:
    from .deep_learning import AutoencoderCompressor
    _has_autoencoder = True
except ImportError:
    _has_autoencoder = False
    warnings.warn("Deep learning compression (AutoencoderCompressor) not available. Some features will be disabled.")

# Phase 2: Advanced neural-optimized algorithms
try:
    from .neural_lz import (
        MultiChannelNeuralLZ,
        NeuralLZ77Compressor,
        create_neural_lz_compressor,
    )
    _has_neural_lz = True
except ImportError:
    _has_neural_lz = False
    warnings.warn("Neural LZ compression algorithms not available. Some features will be disabled.")

try:
    from .neural_arithmetic import (
        MultiChannelArithmeticCoder,
        NeuralArithmeticCoder,
        NeuralArithmeticModel,
        create_neural_arithmetic_coder,
    )
    _has_neural_arithmetic = True
except ImportError:
    _has_neural_arithmetic = False
    warnings.warn("Neural arithmetic coding algorithms not available. Some features will be disabled.")

try:
    from .lossy_neural import (
        AdaptiveWaveletCompressor,
        NeuralAutoencoder,
        PerceptualQuantizer,
        create_lossy_compressor_suite,
    )
    _has_lossy_neural = True
except ImportError:
    _has_lossy_neural = False
    warnings.warn("Advanced lossy neural compression algorithms not available. Some features will be disabled.")

try:
    from .gpu_acceleration import (
        GPUCompressionBackend,
        RealTimeGPUPipeline,
        create_gpu_compression_system,
    )
    _has_gpu_acceleration = True
except ImportError:
    _has_gpu_acceleration = False
    warnings.warn("GPU acceleration modules not available. Some features will be disabled.")

# Phase 3: Advanced techniques
try:
    from .predictive import (
        AdaptiveNeuralPredictor,
        MultiChannelPredictiveCompressor,
        NeuralLinearPredictor,
        create_predictive_compressor,
    )
    _has_predictive = True
except ImportError:
    _has_predictive = False
    warnings.warn("Predictive compression algorithms not available. Some features will be disabled.")

try:
    from .context_aware import (
        BrainStateDetector,
        ContextAwareCompressor,
        HierarchicalContextModel,
        SpatialContextModel,
        create_context_aware_compressor,
    )
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
