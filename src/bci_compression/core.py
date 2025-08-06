"""Core compression functionality with robust error handling."""

import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Base exception for compression errors"""
    pass


class BaseCompressor:
    """Base class for all compressors with error handling"""

    def __init__(self, name: str = "base"):
        self.name = name
        self._is_initialized = False

    def compress(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with error handling"""
        try:
            if not isinstance(data, np.ndarray):
                raise CompressionError(f"Expected numpy array, got {type(data)}")

            if data.size == 0:
                raise CompressionError("Cannot compress empty array")

            return self._compress_impl(data)

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Compression failed: {e}") from e

    def decompress(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress data with error handling"""
        try:
            if not compressed:
                raise CompressionError("Cannot decompress empty data")

            return self._decompress_impl(compressed, metadata)

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise CompressionError(f"Decompression failed: {e}") from e

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Override in subclasses"""
        raise NotImplementedError

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Override in subclasses"""
        raise NotImplementedError


def validate_compression_integrity(
    original: np.ndarray,
    decompressed: np.ndarray,
    check_shape: bool = True,
    check_dtype: bool = True,
    check_hash: bool = False
) -> None:
    """
    Validate compression/decompression integrity.

    Parameters
    ----------
    original : np.ndarray
        Original data
    decompressed : np.ndarray
        Decompressed data
    check_shape : bool, default=True
        Whether to check shape matches
    check_dtype : bool, default=True
        Whether to check dtype matches
    check_hash : bool, default=False
        Whether to check hash matches (expensive)
    """
    import hashlib

    if check_shape and original.shape != decompressed.shape:
        raise ValueError(f"Decompressed data shape {decompressed.shape} does not match original {original.shape}")
    if check_dtype and original.dtype != decompressed.dtype:
        raise ValueError(f"Decompressed data dtype {decompressed.dtype} does not match original {original.dtype}")
    if check_hash:
        orig_hash = hashlib.sha256(original.tobytes()).hexdigest()
        decomp_hash = hashlib.sha256(decompressed.tobytes()).hexdigest()
        if orig_hash != decomp_hash:
            raise ValueError("Decompressed data hash does not match original (possible corruption)")


class Compressor(BaseCompressor):
    """Generic compressor using plugin system."""

    def __init__(self, algorithm: str, **kwargs):
        super().__init__(name=algorithm)
        self.algorithm_name = algorithm

        # Import here to avoid circular imports
        try:
            from .plugins import get_plugin
            self.compressor_plugin = get_plugin(self.algorithm_name)
            if not self.compressor_plugin:
                raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
            self.compressor_instance = self.compressor_plugin(**kwargs)
        except ImportError:
            raise ValueError(f"Plugin system not available for algorithm: {self.algorithm_name}")

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Implement compression using plugin."""
        if hasattr(self.compressor_instance, '_compress_impl'):
            return self.compressor_instance._compress_impl(data)
        else:
            # Fallback for legacy interface
            compressed = self.compressor_instance.compress(data)
            metadata = {'algorithm': self.algorithm_name}
            return compressed, metadata

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Implement decompression using plugin."""
        if hasattr(self.compressor_instance, '_decompress_impl'):
            return self.compressor_instance._decompress_impl(compressed, metadata)
        else:
            # Fallback for legacy interface
            return self.compressor_instance.decompress(compressed)


def create_compressor(algorithm: str, **kwargs) -> Compressor:
    """Factory function to create compressor instances."""
    return Compressor(algorithm, **kwargs)
class Compressor(BaseCompressor):
    def __init__(self, algorithm: str, **kwargs):
        super().__init__(kwargs)
        self.algorithm_name = algorithm
        self.compressor_plugin = get_plugin(self.algorithm_name)
        if not self.compressor_plugin:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
        self.compressor_instance = self.compressor_plugin(**kwargs)

    def compress(self, data: np.ndarray) -> bytes:
        return self.compressor_instance.compress(data)

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        return self.compressor_instance.decompress(compressed_data)

    def fit(self, data: np.ndarray) -> None:
        self.compressor_instance.fit(data)
