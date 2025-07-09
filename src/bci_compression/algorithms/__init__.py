"""
Compression algorithms for neural data.
"""

from .lossless import AdaptiveLZCompressor, DictionaryCompressor
from .lossy import QuantizationCompressor, WaveletCompressor
from .deep_learning import AutoencoderCompressor

__all__ = [
    "AdaptiveLZCompressor",
    "DictionaryCompressor", 
    "QuantizationCompressor",
    "WaveletCompressor",
    "AutoencoderCompressor",
]
