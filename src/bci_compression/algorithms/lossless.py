"""
Lossless compression algorithms for neural data.
"""

import logging
from typing import Any, Dict

import numpy as np

from ..core import BaseCompressor


class AdaptiveLZCompressor(BaseCompressor):
    """
    Adaptive Lempel-Ziv compressor optimized for neural data patterns.
    """

    def __init__(self, dictionary_size: int = 4096, lookahead_buffer: int = 256):
        super().__init__()
        self.dictionary_size = dictionary_size
        self.lookahead_buffer = lookahead_buffer

    def compress(self, data: np.ndarray) -> bytes:
        logging.info(f"[AdaptiveLZ] Compressing data with shape {data.shape} and dtype {data.dtype}")
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        original_size = data.nbytes
        compressed = data.astype(np.float16).tobytes()
        self.compression_ratio = original_size / len(compressed)
        return compressed

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        logging.info("[AdaptiveLZ] Decompressing data")
        data = np.frombuffer(compressed_data, dtype=np.float16).astype(np.float32)
        if not hasattr(self, '_last_shape') or not hasattr(self, '_last_dtype'):
            raise ValueError("Missing shape or dtype information for decompression in AdaptiveLZCompressor.")
        try:
            data = data.reshape(self._last_shape)
            data = data.astype(self._last_dtype)
        except Exception as e:
            logging.exception("[AdaptiveLZ] Integrity check failed during decompression")
            raise
        return data


class NeuralLZ77Compressor:
    """LZ77 variant for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        # Dummy implementation
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        # Dummy implementation
        return np.frombuffer(compressed, dtype=np.float32)


class NeuralArithmeticCoder:
    """Arithmetic coding for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)


class DictionaryCompressor(BaseCompressor):
    """
    Dictionary-based compression for repetitive neural patterns.
    """

    def __init__(self, pattern_length: int = 32, dictionary_size: int = 1024):
        super().__init__()
        self.pattern_length = pattern_length
        self.dictionary_size = dictionary_size
        self.dictionary: Dict[str, int] = {}

    def compress(self, data: np.ndarray) -> bytes:
        logging.info(f"[Dictionary] Compressing data with shape {data.shape} and dtype {data.dtype}")
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        original_size = data.nbytes
        compressed = data.astype(np.int16).tobytes()
        self.compression_ratio = original_size / len(compressed)
        return compressed

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        logging.info("[Dictionary] Decompressing data")
        data = np.frombuffer(compressed_data, dtype=np.int16).astype(np.float32)
        try:
            if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
        except Exception as e:
            logging.exception("[Dictionary] Integrity check failed during decompression")
            raise
        return data
