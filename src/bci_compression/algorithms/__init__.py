"""
Compression algorithms for neural data.

This package contains various compression algorithms optimized for
brain-computer interface applications, including both lossless and
lossy compression methods.
"""

# Core compression algorithms
from .lossless import AdaptiveLZCompressor, DictionaryCompressor

# Try to import lossy algorithms with graceful fallback
try:
    from .lossy import QuantizationCompressor, WaveletCompressor
    _has_base_lossy = True
except ImportError:
    _has_base_lossy = False

try:
    from .deep_learning import AutoencoderCompressor
    _has_autoencoder = True
except ImportError:
    _has_autoencoder = False

# Phase 2: Advanced neural-optimized algorithms
try:
    from .neural_lz import NeuralLZ77Compressor, MultiChannelNeuralLZ, create_neural_lz_compressor
    _has_neural_lz = True
except ImportError:
    _has_neural_lz = False

try:
    from .neural_arithmetic import (
        NeuralArithmeticModel,
        NeuralArithmeticCoder,
        MultiChannelArithmeticCoder,
        create_neural_arithmetic_coder
    )
    _has_neural_arithmetic = True
except ImportError:
    _has_neural_arithmetic = False

try:
    from .lossy_neural import (
        PerceptualQuantizer,
        AdaptiveWaveletCompressor,
        NeuralAutoencoder,
        create_lossy_compressor_suite
    )
    _has_lossy_neural = True
except ImportError:
    _has_lossy_neural = False

try:
    from .gpu_acceleration import (
        GPUCompressionBackend,
        RealTimeGPUPipeline,
        create_gpu_compression_system
    )
    _has_gpu_acceleration = True
except ImportError:
    _has_gpu_acceleration = False

# Phase 3: Advanced techniques
try:
    from .predictive import (
        NeuralLinearPredictor,
        AdaptiveNeuralPredictor,
        MultiChannelPredictiveCompressor,
        create_predictive_compressor
    )
    _has_predictive = True
except ImportError:
    _has_predictive = False

try:
    from .context_aware import (
        BrainStateDetector,
        HierarchicalContextModel,
        SpatialContextModel,
        ContextAwareCompressor,
        create_context_aware_compressor
    )
    _has_context_aware = True
except ImportError:
    _has_context_aware = False

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
