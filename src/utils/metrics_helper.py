"""
Metrics helper for BCI toolkit.
Provides SNR, compression ratio, and other formulas.

References:
- Mathematical formulas in docstrings
"""
import numpy as np

def snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculates signal-to-noise ratio (SNR) in dB.
    Formula: SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculates compression ratio.
    Formula: Compression Ratio = original_size / compressed_size
    """
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size
