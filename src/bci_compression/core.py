# TEST: This comment was added to verify auto-accept of changes via .cursor/settings.json
"""
Core compression functionality for neural data.
"""

import hashlib
import logging
import logging.config
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .plugins import get_plugin


class BaseCompressor(ABC):
    """Abstract base class for neural data compressors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compression_ratio: float = 0.0
        self._is_fitted = False

    @abstractmethod
    def compress(self, data: np.ndarray) -> bytes:
        """Compress neural data."""

    @abstractmethod
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress neural data."""

    def fit(self, data: np.ndarray) -> None:
        """Fit compressor parameters to training data."""
        self._is_fitted = True

    def get_compression_ratio(self) -> float:
        """Get the compression ratio of the last compression operation."""
        return self.compression_ratio

    def _check_integrity(
            self,
            original: np.ndarray,
            decompressed: np.ndarray,
            check_shape: bool = True,
            check_dtype: bool = True,
            check_hash: bool = False) -> None:
        """Check integrity between original and decompressed data. Optionally check shape, dtype, and hash."""
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
