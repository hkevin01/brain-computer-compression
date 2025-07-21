"""
Mobile Metrics for BCI Compression on Mobile Devices

Provides lightweight, mobile-appropriate performance and quality metrics.
"""

import numpy as np

from src.bci_compression.benchmarking.metrics import BenchmarkMetrics


class MobileMetrics:
    """
    Computes performance and quality metrics for mobile BCI compression.
    Delegates to BenchmarkMetrics for all calculations.
    """
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        return BenchmarkMetrics.compression_ratio(original_size, compressed_size)

    @staticmethod
    def latency_ms(start_time: float, end_time: float) -> float:
        return BenchmarkMetrics.processing_latency(start_time, end_time)

    @staticmethod
    def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        return BenchmarkMetrics.snr(original, reconstructed)

    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        return BenchmarkMetrics.psnr(original, reconstructed)

    @staticmethod
    def power_estimate(processing_time_ms: float, device_power_mw: float = 100.0) -> float:
        return BenchmarkMetrics.power_estimate(processing_time_ms, device_power_mw)
