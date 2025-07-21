"""
Benchmarking Metrics for BCI Compression Toolkit

Defines standardized evaluation metrics for neural data compression algorithms.

Metrics:
- Compression Ratio
- Processing Speed (Latency)
- Signal Quality (SNR, PSNR)
- Memory Usage
- GPU Utilization

References:
- Neuralink Compression Challenge
- IEEE Signal Processing standards
"""

from typing import Any

import numpy as np


class BenchmarkMetrics:
    """
    Standardized metrics for evaluating compression algorithms.
    """
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio (original/compressed)."""
        if compressed_size == 0:
            raise ValueError("Compressed size must be > 0")
        return original_size / compressed_size

    @staticmethod
    def processing_latency(start_time: float, end_time: float) -> float:
        """Calculate processing latency in milliseconds."""
        return (end_time - start_time) * 1000

    @staticmethod
    def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Signal-to-noise ratio (SNR) in dB."""
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray, max_value: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio (PSNR) in dB."""
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_value / np.sqrt(mse))

    @staticmethod
    def memory_usage(obj: Any) -> int:
        """Estimate memory usage of an object in bytes."""
        import sys
        return sys.getsizeof(obj)

    @staticmethod
    def power_estimate(processing_time_ms: float, device_power_mw: float = 100.0) -> float:
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
