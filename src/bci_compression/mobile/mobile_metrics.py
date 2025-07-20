"""
Mobile Metrics for BCI Compression on Mobile Devices

Provides lightweight, mobile-appropriate performance and quality metrics.
"""

import numpy as np


class MobileMetrics:
    """
    Computes performance and quality metrics for mobile BCI compression.
    Metrics include compression ratio, latency, power estimate, SNR, PSNR, etc.
    """
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        if compressed_size == 0:
            return 0.0
        return original_size / compressed_size

    @staticmethod
    def latency_ms(start_time: float, end_time: float) -> float:
        return (end_time - start_time) * 1000

    @staticmethod
    def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_val = np.max(np.abs(original))
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def power_estimate(processing_time_ms: float, device_power_mw: float = 100.0) -> float:
        """
        Estimate energy usage (mJ) for a given processing time and device power.
        device_power_mw: average power draw in milliwatts (default 100mW)
        """
        return (processing_time_ms / 1000.0) * device_power_mw
