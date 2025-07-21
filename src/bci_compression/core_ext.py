"""
Core extension: lossless compression algorithms for neural data streams.
Stub implementations for plugin registration and test compatibility.
"""

import numpy as np
from typing import Any

class AdaptiveLZCompressor:
    """Stub Adaptive LZ Compressor for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)

class DictionaryCompressor:
    """Stub Dictionary Compressor for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)

class HuffmanCompressor:
    """Stub Huffman Compressor for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)

class LZ77Compressor:
    """Stub LZ77 Compressor for neural data."""
    def compress(self, data: np.ndarray) -> bytes:
        return data.tobytes()
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(compressed, dtype=np.float32)

def lz77_compress_core(data: np.ndarray, **kwargs: Any) -> bytes:
    """Stub core LZ77 compression function."""
    return data.tobytes()

def lz77_decompress_core(compressed: bytes, **kwargs: Any) -> np.ndarray:
    """Stub core LZ77 decompression function."""
    return np.frombuffer(compressed, dtype=np.float32)
