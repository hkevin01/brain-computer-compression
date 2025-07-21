"""
Signal processing utilities for BCI compression toolkit.
Includes FFT, IIR filtering, and wavelet transforms.

References:
- NumPy, SciPy
- BCI signal integrity
"""
from typing import Tuple, Optional
import numpy as np
import scipy.signal
import pywt


def apply_fft(signal: np.ndarray) -> np.ndarray:
    """
    Computes the FFT of the input signal.
    Args:
        signal: Input signal array (channels x samples)
    Returns:
        FFT-transformed signal
    """
    return np.fft.fft(signal, axis=-1)


def apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Applies an IIR filter to the signal.
    Args:
        signal: Input signal array (channels x samples)
        b, a: IIR filter coefficients
    Returns:
        Filtered signal
    """
    return scipy.signal.lfilter(b, a, signal, axis=-1)


def apply_wavelet_transform(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> Tuple[list, list]:
    """
    Applies discrete wavelet transform to the signal.
    Args:
        signal: Input signal array (channels x samples)
        wavelet: Wavelet type
        level: Decomposition level
    Returns:
        Coefficients and reconstruction
    """
    coeffs = pywt.wavedec(signal, wavelet, axis=-1, level=level)
    reconstructed = pywt.waverec(coeffs, wavelet, axis=-1)
    return coeffs, reconstructed
