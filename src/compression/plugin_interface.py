"""
CompressionPluginInterface
=========================

Abstract base class for third-party compression algorithm plugins.

Usage:
    - Inherit from CompressionPluginInterface to implement a new plugin.
    - Implement all abstract methods: compress, decompress, get_name, get_config.
    - Register plugins with the plugin manager for dynamic loading.

Extension:
    - Add custom configuration parameters via get_config.
    - Ensure thread safety and error handling for real-time use.

References:
    - PEP 8, PEP 484 (type hints)
    - BCI compression toolkit plugin guidelines
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class CompressionPluginInterface(ABC):
    """
    Abstract base class for compression algorithm plugins.

    All plugins must implement the following methods.
    """

    @abstractmethod
    def compress(self, data: np.ndarray, config: Dict[str, Any]) -> bytes:
        """
        Compress neural data.

        Args:
            data: Input neural data array (channels x samples).
            config: Compression configuration parameters.
        Returns:
            Compressed data as bytes.
        """
        pass

    @abstractmethod
    def decompress(self, compressed: bytes, config: Dict[str, Any]) -> np.ndarray:
        """
        Decompress neural data.

        Args:
            compressed: Compressed data as bytes.
            config: Compression configuration parameters.
        Returns:
            Decompressed neural data array.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the plugin/algorithm.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the plugin's configuration parameters and defaults.
        """
        pass
