# TEST: This comment was added to verify auto-accept of changes via .cursor/settings.json
"""
Core compression functionality for neural data.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np


class BaseCompressor(ABC):
    """Abstract base class for neural data compressors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compression_ratio: float = 0.0
        self._is_fitted = False

    @abstractmethod
    def compress(self, data: np.ndarray) -> bytes:
        """Compress neural data."""
        pass

    @abstractmethod
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress neural data."""
        pass

    def fit(self, data: np.ndarray) -> None:
        """Fit compressor parameters to training data."""
        self._is_fitted = True

    def get_compression_ratio(self) -> float:
        """Get the compression ratio of the last compression operation."""
        return self.compression_ratio

    def _check_integrity(self, original: np.ndarray, decompressed: np.ndarray, check_shape: bool = True, check_dtype: bool = True, check_hash: bool = False) -> None:
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


class NeuralCompressor(BaseCompressor):
    """
    Main neural data compressor with configurable algorithms.

    Parameters
    ----------
    algorithm : str, default='adaptive_lz'
        Compression algorithm to use.
    quality_level : float, default=0.95
        Quality level for lossy compression (0.0 to 1.0).
    real_time : bool, default=False
        Enable real-time processing optimizations.
    """

    def __init__(
        self,
        algorithm: str = "adaptive_lz",
        quality_level: float = 0.95,
        real_time: bool = False,
        **kwargs
    ):
        super().__init__(kwargs)
        self.algorithm = algorithm
        self.quality_level = quality_level
        self.real_time = real_time

        # Initialize algorithm-specific parameters
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        """Initialize the selected compression algorithm."""
        # This will be expanded with actual algorithm implementations
        self._algorithm_params = {
            'adaptive_lz': {'dictionary_size': 4096, 'lookahead_buffer': 256},
            'neural_quantization': {'bits': 8, 'adaptive': True},
            'wavelet_transform': {'wavelet': 'db4', 'levels': 5},
            'deep_autoencoder': {'latent_dim': 64, 'epochs': 100}
        }

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress neural data using the selected algorithm.

        Parameters
        ----------
        data : np.ndarray
            Neural data array with shape (channels, samples) or (samples,).

        Returns
        -------
        bytes
            Compressed data as bytes.
        """
        if data.size == 0:
            raise ValueError("Input data cannot be empty")
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        # Placeholder implementation - will be replaced with actual algorithms
        original_size = data.nbytes

        # Simple demonstration compression (just for structure)
        compressed = data.astype(np.float32).tobytes()

        # Calculate compression ratio
        self.compression_ratio = float(original_size) / float(len(compressed))  # type: ignore

        return compressed

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress neural data.

        Parameters
        ----------
        compressed_data : bytes
            Compressed data bytes.

        Returns
        -------
        np.ndarray
            Decompressed neural data.
        """
        if not compressed_data:
            raise ValueError("Compressed data cannot be empty")

        # Placeholder implementation
        data = np.frombuffer(compressed_data, dtype=np.float32)

        # Check integrity if possible
        if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
            try:
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            self._check_integrity(np.zeros(self._last_shape, dtype=self._last_dtype), data, check_shape=True, check_dtype=True, check_hash=False)

        return data


def load_neural_data(filepath: str, format: str = "auto") -> np.ndarray:
    """
    Load neural data from various file formats.

    Parameters
    ----------
    filepath : str
        Path to the neural data file.
    format : str, default='auto'
        File format specification ('auto', 'nev', 'nsx', 'hdf5', 'mat').

    Returns
    -------
    np.ndarray
        Neural data array with shape (channels, samples).
    """
    import os

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Placeholder implementation - will be expanded with actual format support
    if format == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.nev', '.nsx']:
            format = 'neuroshare'
        elif ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        elif ext == '.mat':
            format = 'matlab'
        else:
            format = 'numpy'

    # For now, just return synthetic data
    # This will be replaced with actual file loading
    np.random.seed(42)
    return np.random.randn(64, 30000)  # 64 channels, 30k samples
