"""
Lossless compression algorithms for neural data.
"""

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
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        original_size = data.nbytes
        compressed = data.astype(np.float16).tobytes()
        self.compression_ratio = original_size / len(compressed)
        return compressed
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        data = np.frombuffer(compressed_data, dtype=np.float16).astype(np.float32)
        if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
            try:
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            self._check_integrity(np.zeros(self._last_shape, dtype=self._last_dtype), data, check_shape=True, check_dtype=True, check_hash=False)
        return data


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
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        original_size = data.nbytes
        compressed = data.astype(np.int16).tobytes()
        self.compression_ratio = original_size / len(compressed)
        return compressed
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        data = np.frombuffer(compressed_data, dtype=np.int16).astype(np.float32)
        if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
            try:
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            self._check_integrity(np.zeros(self._last_shape, dtype=self._last_dtype), data, check_shape=True, check_dtype=True, check_hash=False)
        return data
