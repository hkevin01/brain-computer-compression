"""
Lossless compression algorithms for neural data.
"""

import logging
from typing import Dict

import numpy as np

from bci_compression.core_ext import (
    AdaptiveLZCompressor,
    DictionaryCompressor,
    HuffmanCompressor,
    LZ77Compressor,
    lz77_compress_core,
    lz77_decompress_core,
)
from bci_compression.plugins import CompressorPlugin

from ..plugins import register_plugin

logger = logging.getLogger(__name__)


class BaseCompressor:
    """
    Abstract base class for all compressors.
    """

    def compress(self, data: np.ndarray) -> bytes:
        raise NotImplementedError("compress() must be implemented by subclasses.")

    def decompress(self, compressed: bytes) -> np.ndarray:
        raise NotImplementedError("decompress() must be implemented by subclasses.")


@register_plugin("neural_lz77")
class NeuralLZ77Compressor(BaseCompressor, CompressorPlugin):
    """LZ77 variant for neural data."""

    def compress(self, data: np.ndarray) -> bytes:
        # Minimal stub: flatten and convert to bytes
        return data.astype(np.float32).tobytes()

    def decompress(self, compressed: bytes) -> np.ndarray:
        # Minimal stub: convert bytes back to float32 array
        arr = np.frombuffer(compressed, dtype=np.float32)
        # Assume original shape is known elsewhere
        return arr


@register_plugin("neural_arithmetic")
class NeuralArithmeticCoder(BaseCompressor, CompressorPlugin):
    """Arithmetic coding for neural data."""

    def compress(self, data: np.ndarray) -> bytes:
        # Minimal stub: flatten and convert to bytes
        return data.astype(np.float32).tobytes()

    def decompress(self, compressed: bytes) -> np.ndarray:
        arr = np.frombuffer(compressed, dtype=np.float32)
        return arr
