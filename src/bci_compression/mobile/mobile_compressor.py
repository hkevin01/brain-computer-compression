"""
Mobile-optimized BCI compressor for real-time applications.

This module provides lightweight compression algorithms specifically designed
for mobile and embedded BCI devices with power and memory constraints.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..core import BaseCompressor


class MobileBCICompressor(BaseCompressor):
    """
    Mobile-optimized BCI compressor with real-time capabilities.

    Features:
    - Lightweight algorithms for power-constrained devices
    - Real-time processing with bounded memory usage
    - Adaptive quality control based on device capabilities
    - Streaming support for continuous data processing
    """

    def __init__(
        self,
        algorithm: str = "mobile_lz",
        quality_level: float = 0.8,
        buffer_size: int = 512,
        power_mode: str = "balanced",
        max_memory_mb: int = 50
    ):
        """
        Initialize mobile BCI compressor.

        Parameters
        ----------
        algorithm : str, default="mobile_lz"
            Compression algorithm: "mobile_lz", "lightweight_quant", "fast_predict"
        quality_level : float, default=0.8
            Quality level (0.0-1.0) for lossy compression
        buffer_size : int, default=512
            Processing buffer size (smaller = lower latency, higher CPU)
        power_mode : str, default="balanced"
            Power optimization mode: "battery_save", "balanced", "performance"
        max_memory_mb : int, default=50
            Maximum memory usage in MB
        """
        super().__init__()
        self.algorithm = algorithm
        self.quality_level = quality_level
        self.buffer_size = buffer_size
        self.power_mode = power_mode
        self.max_memory_mb = max_memory_mb

        # Performance tracking
        self.processing_times = []
        self.memory_usage = []
        self.compression_ratios = []

        # Initialize algorithm-specific components
        self._init_algorithm()

    def _init_algorithm(self):
        """Initialize algorithm-specific parameters and components."""
        if self.algorithm == "mobile_lz":
            self._init_mobile_lz()
        elif self.algorithm == "lightweight_quant":
            self._init_lightweight_quant()
        elif self.algorithm == "fast_predict":
            self._init_fast_predict()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _init_mobile_lz(self):
        """Initialize mobile LZ compression."""
        # Optimized for mobile: smaller dictionary, faster lookups
        self.dictionary_size = min(2048, self.buffer_size // 2)
        self.lookahead_size = min(64, self.buffer_size // 8)
        self.dictionary = {}
        self.dict_counter = 0

    def _init_lightweight_quant(self):
        """Initialize lightweight quantization."""
        # Adaptive bit allocation based on quality level (increased minimum)
        self.bits = max(8, int(12 * self.quality_level))
        self.scale_factor = 1.0
        self.offset = 0.0

    def _init_fast_predict(self):
        """Initialize fast prediction-based compression."""
        # Simple linear prediction with minimal coefficients
        self.prediction_order = 3  # Reduced from typical 8-16
        self.coefficients = np.zeros(self.prediction_order)
        self.history = np.zeros(self.prediction_order)

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress neural data using mobile-optimized algorithms.

        Parameters
        ----------
        data : np.ndarray
            Input neural data (channels, samples) or (samples,)

        Returns
        -------
        bytes
            Compressed data
        """
        start_time = time.time()

        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self._last_shape = data.shape
        self._last_dtype = data.dtype

        # Algorithm-specific compression
        if self.algorithm == "mobile_lz":
            compressed = self._compress_mobile_lz(data)
        elif self.algorithm == "lightweight_quant":
            compressed = self._compress_lightweight_quant(data)
        elif self.algorithm == "fast_predict":
            compressed = self._compress_fast_predict(data)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Calculate metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        original_size = data.nbytes
        compressed_size = len(compressed)
        self.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        self.compression_ratios.append(self.compression_ratio)

        # Memory usage tracking (simplified)
        self.memory_usage.append(compressed_size / (1024 * 1024))  # MB

        return compressed

    def _compress_mobile_lz(self, data: np.ndarray) -> bytes:
        """Enhanced mobile LZ compression with dictionary."""
        compressed = []

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            # Enhanced compression: combine RLE with simple dictionary
            encoded = self._enhanced_compress_channel(channel_data)
            compressed.extend(encoded)

        return bytes(compressed)

    def _find_patterns(self, data: np.ndarray) -> List[np.ndarray]:
        """Find common patterns in the data."""
        patterns = []
        pattern_length = 4

        for i in range(0, len(data) - pattern_length, pattern_length):
            pattern = data[i:i + pattern_length]
            if len(patterns) < 16:  # Limit dictionary size
                patterns.append(pattern)

        return patterns

    def _find_longest_pattern(self, data: np.ndarray, start: int, patterns: List[np.ndarray]) -> Optional[np.ndarray]:
        """Find the longest matching pattern starting at position."""
        best_match = None
        best_length = 0

        for pattern in patterns:
            pattern_len = len(pattern)
            if start + pattern_len <= len(data):
                if np.allclose(data[start:start + pattern_len], pattern, atol=0.1):
                    # Check if pattern repeats
                    repeat_count = 1
                    pos = start + pattern_len
                    while pos + pattern_len <= len(data) and np.allclose(data[pos:pos + pattern_len], pattern, atol=0.1):
                        repeat_count += 1
                        pos += pattern_len

                    total_length = pattern_len * repeat_count
                    if total_length > best_length:
                        best_length = total_length
                        best_match = np.tile(pattern, repeat_count)

        return best_match

    def _enhanced_compress_channel(self, data: np.ndarray) -> List[int]:
        """Enhanced compression with improved RLE and simple pattern detection."""
        if len(data) == 0:
            return []

        # Simple pattern detection: look for repeated values
        encoded = []
        i = 0
        while i < len(data):
            # Check for repeated values (simple pattern)
            current_val = data[i]
            count = 1

            # Count consecutive similar values
            while i + count < len(data) and abs(data[i + count] - current_val) < 0.01 and count < 254:
                count += 1

            if count > 3:
                # Encode as pattern
                scaled_val = int(np.clip(current_val * 50 + 127, 0, 253))
                encoded.extend([255, scaled_val, count])  # Special marker for patterns
            else:
                # Regular RLE encoding
                scaled_val = int(np.clip(current_val * 50 + 127, 0, 253))
                encoded.extend([count, scaled_val])

            i += count

        return encoded

    def _compress_lightweight_quant(self, data: np.ndarray) -> bytes:
        """Improved lightweight quantization with minimal dithering."""
        # Adaptive quantization with minimal dithering for better quality
        data_min, data_max = data.min(), data.max()
        self.scale_factor = (data_max - data_min) / (2**self.bits - 1) if data_max > data_min else 1.0
        self.offset = data_min

        # Add minimal dithering to reduce quantization artifacts
        dither = np.random.normal(0, self.scale_factor * 0.01, data.shape)
        dithered_data = data + dither

        # Quantize with improved scaling
        quantized = np.clip(((dithered_data - self.offset) / self.scale_factor), 0, 2**self.bits - 1).astype(np.uint8)

        # Pack metadata and data
        metadata = np.array([self.scale_factor, self.offset, data_min, data_max], dtype=np.float32)
        return metadata.tobytes() + quantized.tobytes()

    def _compress_fast_predict(self, data: np.ndarray) -> bytes:
        """Improved fast prediction-based compression."""
        compressed = []

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            if len(channel_data) > self.prediction_order:
                # Improved coefficient estimation
                self._improved_update_coefficients(channel_data)

                # Predict and encode residuals with adaptive quantization
                residuals = []
                for i in range(self.prediction_order, len(channel_data)):
                    prediction = np.dot(self.coefficients, self.history)
                    residual = channel_data[i] - prediction

                    # Adaptive quantization of residuals
                    residual_quantized = self._quantize_residual(residual)
                    residuals.append(residual_quantized)

                    # Update history
                    self.history = np.roll(self.history, 1)
                    self.history[0] = channel_data[i]

                # Encode residuals more efficiently
                residual_bytes = np.array(residuals, dtype=np.int8).tobytes()
                compressed.extend(residual_bytes)

        return bytes(compressed)

    def _improved_update_coefficients(self, data: np.ndarray):
        """Improved coefficient estimation using autocorrelation."""
        if len(data) >= self.prediction_order * 2:
            # Use autocorrelation for better coefficient estimation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + self.prediction_order + 1]

            # Simple Levinson-Durbin approximation
            if autocorr[0] > 0:
                for i in range(self.prediction_order):
                    self.coefficients[i] = -autocorr[i + 1] / (autocorr[0] + 1e-8) * 0.5

    def _quantize_residual(self, residual: float) -> int:
        """Adaptive quantization of prediction residuals."""
        # Scale residual to int8 range with adaptive scaling
        scale = max(0.1, np.abs(residual) / 10.0)
        quantized = int(np.clip(residual / scale, -127, 127))
        return quantized

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress data using mobile-optimized algorithms.

        Parameters
        ----------
        compressed_data : bytes
            Compressed data

        Returns
        -------
        np.ndarray
            Decompressed neural data
        """
        if self.algorithm == "mobile_lz":
            return self._decompress_mobile_lz(compressed_data)
        elif self.algorithm == "lightweight_quant":
            return self._decompress_lightweight_quant(compressed_data)
        elif self.algorithm == "fast_predict":
            return self._decompress_fast_predict(compressed_data)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _decompress_mobile_lz(self, compressed_data: bytes) -> np.ndarray:
        """Decompress enhanced mobile LZ data."""
        data_list = list(compressed_data)
        decompressed = []

        i = 0
        while i < len(data_list):
            if i + 2 < len(data_list) and data_list[i] == 255:
                # Pattern reference
                scaled_val = data_list[i + 1]
                count = data_list[i + 2]
                # Reverse scaling: (scaled_val - 127) / 50
                value = (scaled_val - 127) / 50.0
                decompressed.extend([value] * count)
                i += 3
            else:
                # Run-length encoded value
                if i + 1 < len(data_list):
                    count = data_list[i]
                    scaled_val = data_list[i + 1]
                    # Reverse scaling: (scaled_val - 127) / 50
                    value = (scaled_val - 127) / 50.0
                    decompressed.extend([value] * count)
                    i += 2
                else:
                    break

        return np.array(decompressed).reshape(self._last_shape)

    def _decompress_lightweight_quant(self, compressed_data: bytes) -> np.ndarray:
        """Decompress improved lightweight quantized data."""
        # Extract metadata (now 4 float32 values)
        metadata_size = 16  # 4 float32 values
        metadata = np.frombuffer(compressed_data[:metadata_size], dtype=np.float32)
        self.scale_factor, self.offset, data_min, data_max = metadata

        # Extract quantized data
        quantized = np.frombuffer(compressed_data[metadata_size:], dtype=np.uint8)

        # Dequantize
        decompressed = quantized.astype(np.float32) * self.scale_factor + self.offset
        return decompressed.reshape(self._last_shape)

    def _decompress_fast_predict(self, compressed_data: bytes) -> np.ndarray:
        """Decompress improved fast prediction data."""
        # Calculate total residuals per channel
        total_samples = self._last_shape[1]
        residuals_per_channel = total_samples - self.prediction_order
        total_residuals = residuals_per_channel * self._last_shape[0]

        # Extract residuals
        residuals = np.frombuffer(compressed_data, dtype=np.int8)

        # Reconstruct each channel
        reconstructed_channels = []
        residual_idx = 0

        for ch in range(self._last_shape[0]):
            channel_residuals = residuals[residual_idx:residual_idx + residuals_per_channel]
            residual_idx += residuals_per_channel

            # Reconstruct signal with improved prediction
            reconstructed = np.zeros(total_samples)
            reconstructed[:self.prediction_order] = np.random.randn(self.prediction_order) * 0.1

            for i, residual_quantized in enumerate(channel_residuals):
                # Dequantize residual
                scale = max(0.1, np.abs(residual_quantized) / 10.0)
                residual = residual_quantized * scale

                # Predict and reconstruct
                prediction = np.dot(self.coefficients, reconstructed[i:i + self.prediction_order])
                reconstructed[i + self.prediction_order] = prediction + residual

            reconstructed_channels.append(reconstructed)

        return np.array(reconstructed_channels)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.processing_times:
            return {}

        return {
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'avg_compression_ratio': np.mean(self.compression_ratios),
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'total_samples_processed': len(self.processing_times)
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.processing_times.clear()
        self.memory_usage.clear()
        self.compression_ratios.clear()
