"""
Adaptive Compression Pipeline for Network-Aware Edge-Cloud Scenarios
Implements dynamic offloading, edge-cloud optimization, and real-time adaptation.
"""

import numpy as np
from typing import Any, Dict

class AdaptiveCompressionPipeline:
    """
    Pipeline for adaptive compression based on network conditions.
    Supports dynamic offloading and real-time adaptation for edge-cloud BCI systems.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('mode', 'auto')
        self.network_status = 'normal'

    def analyze_network(self, metrics: Dict[str, Any]) -> str:
        """Analyze network metrics and select compression mode."""
        latency = metrics.get('latency', 10)
        bandwidth = metrics.get('bandwidth', 100)
        if latency > 50 or bandwidth < 10:
            self.mode = 'edge'
        elif bandwidth > 100:
            self.mode = 'cloud'
        else:
            self.mode = 'hybrid'
        self.network_status = self.mode
        return self.mode

    def compress(self, data: np.ndarray) -> bytes:
        """Compress data using selected mode, preserving shape metadata."""
        # Store shape and dtype for faithful reconstruction on decompress
        self._last_shape = data.shape
        self._last_dtype = data.dtype

        if self.mode == 'edge':
            return self._edge_compress(data)
        elif self.mode == 'cloud':
            return self._cloud_compress(data)
        else:
            return self._hybrid_compress(data)

    def _edge_compress(self, data: np.ndarray) -> bytes:
        # Lightweight edge compression: quantise to float16 to halve bandwidth
        return data.astype(np.float16).tobytes()

    def _cloud_compress(self, data: np.ndarray) -> bytes:
        # High-quality cloud compression: preserve full float32 precision
        return data.astype(np.float32).tobytes()

    def _hybrid_compress(self, data: np.ndarray) -> bytes:
        # Hybrid: full float32 (shape preserved; edge quantisation applied
        # at packet boundary in a real implementation)
        return data.astype(np.float32).tobytes()

    def decompress(self, compressed: bytes) -> np.ndarray:
        """Decompress bytes back to the original ndarray shape."""
        if self.mode == 'edge':
            arr = np.frombuffer(compressed, dtype=np.float16).astype(np.float32)
        else:
            arr = np.frombuffer(compressed, dtype=np.float32)

        # Restore original shape if available
        if hasattr(self, '_last_shape'):
            arr = arr.reshape(self._last_shape)
        return arr
