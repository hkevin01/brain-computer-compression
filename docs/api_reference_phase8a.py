"""
Brain-Computer Interface Compression Toolkit - API Reference (Phase 8a)

This module provides comprehensive API documentation for all compression algorithms,
utilities, and dashboard components with Phase 8a enhancements.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class CompressionAPI:
    """
    Main API interface for BCI compression algorithms with Phase 8a enhancements.

    This class provides a unified interface for all compression algorithms,
    including the new transformer-based, VAE, and adaptive selection systems.

    Attributes:
        available_algorithms (List[str]): List of available compression algorithms
        default_config (Dict[str, Any]): Default configuration parameters
        performance_monitor (PerformanceMonitor): Real-time performance tracking

    Examples:
        Basic usage:

        >>> from bci_compression import CompressionAPI
        >>> api = CompressionAPI()
        >>>
        >>> # Compress neural data with auto algorithm selection
        >>> neural_data = np.random.randn(64, 2000)  # 64 channels, 2000 samples
        >>> result = api.compress(neural_data, algorithm='auto')
        >>> print(f"Compression ratio: {result['compression_ratio']:.2f}x")

        Advanced transformer compression:

        >>> # Configure transformer for high-quality compression
        >>> config = {
        >>>     'algorithm': 'transformer',
        >>>     'd_model': 256,
        >>>     'n_heads': 8,
        >>>     'compression_ratio_target': 4.0,
        >>>     'quality_mode': 'high'
        >>> }
        >>> result = api.compress(neural_data, **config)
        >>> print(f"SNR: {result['snr_db']:.1f} dB")

        Real-time processing:

        >>> # Process streaming data chunks
        >>> for chunk in data_stream:
        >>>     result = api.compress_real_time(chunk)
        >>>     if result['latency_ms'] > 2.0:
        >>>         print("Warning: Latency exceeds real-time target")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compression API.

        Args:
            config: Optional configuration dictionary with algorithm parameters

        Raises:
            ImportError: If required dependencies are not available
            ValueError: If invalid configuration is provided
        """
        self.config = config or self._get_default_config()
        self.available_algorithms = self._get_available_algorithms()
        self.performance_monitor = None
        self._initialize_algorithms()

    def compress(
        self,
        data: np.ndarray,
        algorithm: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compress neural data using specified algorithm.

        Args:
            data: Neural data array with shape (channels, samples) or (samples,)
            algorithm: Compression algorithm to use:
                - 'auto': Automatic selection based on signal characteristics
                - 'transformer': Transformer-based compression (Phase 8a)
                - 'vae': Variational autoencoder compression (Phase 8a)
                - 'neural_lz': Neural-optimized LZ compression
                - 'arithmetic': Arithmetic coding
                - 'perceptual': Perceptual quantization
                - 'predictive': Predictive coding
                - 'spike': Spike detection compression (Phase 8a)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dictionary containing:
                - compressed_data (bytes): Compressed data
                - compression_ratio (float): Achieved compression ratio
                - snr_db (float): Signal-to-noise ratio (for lossy compression)
                - latency_ms (float): Compression time in milliseconds
                - algorithm_used (str): Actually used algorithm
                - metadata (dict): Additional compression metadata

        Raises:
            ValueError: If data shape is invalid or algorithm not supported
            RuntimeError: If compression fails

        Examples:
            >>> # Basic compression
            >>> data = np.random.randn(64, 1000)
            >>> result = api.compress(data, algorithm='transformer')
            >>>
            >>> # High-quality compression
            >>> result = api.compress(
            >>>     data,
            >>>     algorithm='transformer',
            >>>     quality_mode='high',
            >>>     compression_ratio_target=3.0
            >>> )
        """
        pass  # Implementation stub

    def decompress(
        self,
        compressed_data: bytes,
        metadata: Dict[str, Any],
        original_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Decompress neural data.

        Args:
            compressed_data: Compressed data bytes
            metadata: Compression metadata from compress()
            original_shape: Original data shape (if not in metadata)

        Returns:
            Decompressed neural data array

        Raises:
            ValueError: If metadata is invalid or incompatible
            RuntimeError: If decompression fails
        """
        pass  # Implementation stub

    def compress_real_time(
        self,
        data: np.ndarray,
        algorithm: str = 'auto',
        target_latency_ms: float = 2.0
    ) -> Dict[str, Any]:
        """
        Real-time compression with latency optimization (Phase 8a).

        Optimized for streaming BCI applications with strict latency requirements.
        Automatically adjusts compression parameters to meet latency targets.

        Args:
            data: Neural data chunk (typically 256-1024 samples)
            algorithm: Compression algorithm or 'auto' for adaptive selection
            target_latency_ms: Maximum acceptable latency in milliseconds

        Returns:
            Dictionary with compression results and real-time metrics:
                - compressed_data (bytes): Compressed data
                - compression_ratio (float): Achieved compression ratio
                - latency_ms (float): Actual compression time
                - real_time_capable (bool): Whether latency target was met
                - performance_grade (str): Performance grade A-F
                - algorithm_used (str): Selected algorithm

        Examples:
            >>> # Real-time processing loop
            >>> for chunk in stream:
            >>>     result = api.compress_real_time(chunk, target_latency_ms=1.5)
            >>>     if not result['real_time_capable']:
            >>>         print(f"Latency warning: {result['latency_ms']:.2f}ms")
        """
        pass  # Implementation stub

    def benchmark_algorithm(
        self,
        algorithm: str,
        test_data: Optional[np.ndarray] = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark compression algorithm performance.

        Args:
            algorithm: Algorithm to benchmark
            test_data: Test data (generates synthetic if None)
            iterations: Number of benchmark iterations

        Returns:
            Benchmark results including timing, quality, and memory usage
        """
        pass  # Implementation stub

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all recent operations.

        Returns:
            Performance summary with averages, trends, and recommendations
        """
        pass  # Implementation stub

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all algorithms."""
        return {
            'transformer': {
                'd_model': 256,
                'n_heads': 8,
                'num_layers': 6,
                'compression_ratio_target': 4.0,
                'causal_masking': True,
                'optimization_level': 2
            },
            'vae': {
                'latent_dim': 64,
                'beta': 1.0,
                'quality_target': 25.0,
                'uncertainty_estimation': True
            },
            'adaptive': {
                'selection_algorithm': 'signal_analysis',
                'fallback_algorithm': 'neural_lz',
                'quality_threshold': 20.0
            }
        }

    def _get_available_algorithms(self) -> List[str]:
        """Get list of available compression algorithms."""
        return [
            'auto', 'transformer', 'vae', 'neural_lz',
            'arithmetic', 'perceptual', 'predictive', 'spike'
        ]

    def _initialize_algorithms(self):
        """Initialize all available algorithms."""
        pass  # Implementation stub


class TransformerCompressionAPI:
    """
    API for transformer-based neural compression (Phase 8a).

    Advanced transformer architecture optimized for neural signal compression
    with multi-head attention mechanisms for temporal pattern detection.

    Performance Targets:
        - Compression Ratio: 3-5x typical, up to 8x for correlated signals
        - Signal Quality: 25-35 dB SNR
        - Latency: <2ms for real-time processing
        - Memory Usage: O(n log n) complexity

    Examples:
        Basic transformer compression:

        >>> from bci_compression import TransformerCompressionAPI
        >>> transformer = TransformerCompressionAPI()
        >>>
        >>> neural_data = np.random.randn(64, 2000)
        >>> result = transformer.compress(neural_data)
        >>> print(f"Transformer compression: {result['compression_ratio']:.2f}x")

        Custom configuration:

        >>> config = {
        >>>     'd_model': 512,  # Larger model for better quality
        >>>     'n_heads': 16,   # More attention heads
        >>>     'compression_ratio_target': 5.0
        >>> }
        >>> transformer = TransformerCompressionAPI(config)
        >>> result = transformer.compress(neural_data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transformer compression.

        Args:
            config: Transformer configuration parameters
        """
        pass  # Implementation stub

    def compress(
        self,
        data: np.ndarray,
        quality_mode: str = 'balanced',
        real_time_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Compress neural data using transformer architecture.

        Args:
            data: Neural data array (channels, samples)
            quality_mode: 'speed', 'balanced', or 'quality'
            real_time_mode: Enable real-time optimizations

        Returns:
            Compression results with transformer-specific metrics
        """
        pass  # Implementation stub

    def get_attention_weights(self, data: np.ndarray) -> np.ndarray:
        """
        Get attention weights for analysis of temporal patterns.

        Args:
            data: Neural data to analyze

        Returns:
            Attention weight matrix for visualization
        """
        pass  # Implementation stub


class VAECompressionAPI:
    """
    API for Variational Autoencoder neural compression (Phase 8a).

    Quality-controlled neural compression with uncertainty modeling
    and disentangled neural representations.

    Performance Targets:
        - Compression Ratio: 2-4x typical
        - Signal Quality: 20-30 dB SNR
        - Uncertainty Quantification: Bayesian estimation
        - Latency: <1ms for real-time processing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VAE compression."""
        pass  # Implementation stub

    def compress(
        self,
        data: np.ndarray,
        beta: float = 1.0,
        quality_target: float = 25.0
    ) -> Dict[str, Any]:
        """
        Compress neural data using VAE with quality control.

        Args:
            data: Neural data array
            beta: Beta parameter for beta-VAE (higher = more disentangled)
            quality_target: Target SNR in dB

        Returns:
            Compression results with uncertainty estimates
        """
        pass  # Implementation stub

    def get_latent_representation(self, data: np.ndarray) -> np.ndarray:
        """Get latent space representation of neural data."""
        pass  # Implementation stub

    def estimate_uncertainty(self, data: np.ndarray) -> Dict[str, float]:
        """Estimate compression uncertainty metrics."""
        pass  # Implementation stub


class AdaptiveSelectionAPI:
    """
    API for adaptive algorithm selection (Phase 8a).

    Real-time algorithm switching based on signal characteristics
    and performance requirements.
    """

    def __init__(self):
        """Initialize adaptive selection system."""
        pass  # Implementation stub

    def select_optimal_algorithm(
        self,
        data: np.ndarray,
        constraints: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select optimal compression algorithm for given data.

        Args:
            data: Neural data to analyze
            constraints: Performance constraints (latency_ms, quality_min, etc.)

        Returns:
            Recommended algorithm name
        """
        pass  # Implementation stub

    def analyze_signal_characteristics(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze signal characteristics for algorithm selection."""
        pass  # Implementation stub


class DashboardAPI:
    """
    API for real-time dashboard integration (Phase 8a).

    Provides real-time metrics, monitoring, and control interface
    for the React + Vite web dashboard.
    """

    def __init__(self):
        """Initialize dashboard API."""
        pass  # Implementation stub

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time compression metrics."""
        pass  # Implementation stub

    def get_algorithm_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance comparison across all algorithms."""
        pass  # Implementation stub

    def update_algorithm_config(
        self,
        algorithm: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update algorithm configuration from dashboard."""
        pass  # Implementation stub


# Type definitions for better IDE support
CompressionResult = Dict[str, Union[bytes, float, str, Dict[str, Any]]]
NeuralData = np.ndarray
AlgorithmConfig = Dict[str, Any]
PerformanceMetrics = Dict[str, float]

# Version information
__version__ = "8.0.0"  # Phase 8a
__api_version__ = "2.0"
__author__ = "BCI Compression Team"
__license__ = "MIT"

# Export main API classes
__all__ = [
    'CompressionAPI',
    'TransformerCompressionAPI',
    'VAECompressionAPI',
    'AdaptiveSelectionAPI',
    'DashboardAPI',
    'CompressionResult',
    'NeuralData',
    'AlgorithmConfig',
    'PerformanceMetrics'
]


if __name__ == "__main__":
    # Example usage demonstration
    print("Brain-Computer Interface Compression Toolkit - API Reference")
    print(f"Version: {__version__}")
    print("\nAvailable APIs:")
    for api_class in __all__[:5]:  # Main API classes
        print(f"  - {api_class}")

    print("\nExample usage:")
    print("""
    from bci_compression import CompressionAPI

    # Initialize API
    api = CompressionAPI()

    # Compress neural data
    neural_data = np.random.randn(64, 2000)
    result = api.compress(neural_data, algorithm='transformer')

    print(f"Compression ratio: {result['compression_ratio']:.2f}x")
    print(f"Signal quality: {result['snr_db']:.1f} dB")
    print(f"Latency: {result['latency_ms']:.2f} ms")
    """)
