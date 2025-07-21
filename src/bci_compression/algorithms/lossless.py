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
    pass


@register_plugin("neural_lz77")
class NeuralLZ77Compressor(BaseCompressor, CompressorPlugin):
    """LZ77 variant for neural data."""

    def compress(self, data: np.ndarray) -> bytes:
        # Dummy implementation
        return data.tobytes()

    def decompress(self, compressed: bytes) -> np.ndarray:
        # Dummy implementation
        return np.frombuffer(compressed, dtype=np.float32)


@register_plugin("neural_arithmetic")
class NeuralArithmeticCoder(BaseCompressor, CompressorPlugin):
    """Arithmetic coding for neural data."""

    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()

    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)
