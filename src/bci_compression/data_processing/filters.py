"""
Signal filtering utilities for neural data.
"""

import numpy as np
from scipy import signal
from typing import Tuple


def apply_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
    filter_type: str = 'butter'
) -> np.ndarray:
    """
    Apply bandpass filter to neural data.

    Parameters
    ----------
    data : np.ndarray
        Input neural data with shape (channels, samples) or (samples,).
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, default=4
        Filter order.
    filter_type : str, default='butter'
        Filter type ('butter', 'ellip', 'cheby1', 'cheby2').

    Returns
    -------
    np.ndarray
        Filtered neural data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if filter_type == 'butter':
        b, a = signal.butter(order, [low, high], btype='band')
    elif filter_type == 'ellip':
        b, a = signal.ellip(order, 0.1, 40, [low, high], btype='band')
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    # Apply filter
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Apply to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
        return filtered_data


def apply_notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency (e.g., 50/60 Hz power line).

    Parameters
    ----------
    data : np.ndarray
        Input neural data.
    notch_freq : float
        Frequency to notch out in Hz.
    fs : float
        Sampling frequency in Hz.
    quality_factor : float, default=30.0
        Quality factor of the notch filter.

    Returns
    -------
    np.ndarray
        Filtered neural data.
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
        return filtered_data


def apply_iir_filter(
    data: np.ndarray,
    filter_coeffs: Tuple[np.ndarray, np.ndarray],
    axis: int = -1
) -> np.ndarray:
    """
    Apply IIR filter with given coefficients.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    filter_coeffs : tuple
        Filter coefficients (b, a).
    axis : int, default=-1
        Axis along which to apply the filter.

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    b, a = filter_coeffs
    return signal.filtfilt(b, a, data, axis=axis)
