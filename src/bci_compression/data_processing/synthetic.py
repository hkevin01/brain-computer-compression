"""
Synthetic neural data generation for testing and benchmarking.
"""

import numpy as np
from typing import Optional, Tuple


def generate_synthetic_neural_data(
    n_channels: int = 64,
    n_samples: int = 30000,
    fs: float = 1000.0,
    noise_level: float = 0.1,
    spike_rate: float = 5.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate synthetic neural data for testing compression algorithms.

    Parameters
    ----------
    n_channels : int, default=64
        Number of neural channels.
    n_samples : int, default=30000
        Number of samples per channel.
    fs : float, default=1000.0
        Sampling frequency in Hz.
    noise_level : float, default=0.1
        Noise level (std deviation).
    spike_rate : float, default=5.0
        Average spike rate per second per channel.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Synthetic neural data with shape (n_channels, n_samples).
    metadata : dict
        Metadata about the generated data.
    """
    if seed is not None:
        np.random.seed(seed)

    # Time vector
    t = np.arange(n_samples) / fs

    # Initialize data array
    data = np.zeros((n_channels, n_samples))

    # Generate base neural activity for each channel
    for ch in range(n_channels):
        # Background noise
        noise = np.random.normal(0, noise_level, n_samples)

        # Oscillatory components (alpha, beta, gamma)
        alpha_freq = 8 + np.random.uniform(-2, 2)  # 6-12 Hz
        beta_freq = 20 + np.random.uniform(-5, 5)  # 15-25 Hz
        gamma_freq = 40 + np.random.uniform(-10, 10)  # 30-50 Hz

        alpha_amp = np.random.uniform(0.1, 0.3)
        beta_amp = np.random.uniform(0.05, 0.15)
        gamma_amp = np.random.uniform(0.02, 0.08)

        oscillations = (
            alpha_amp * np.sin(2 * np.pi * alpha_freq * t) +
            beta_amp * np.sin(2 * np.pi * beta_freq * t) +
            gamma_amp * np.sin(2 * np.pi * gamma_freq * t)
        )

        # Add spikes
        n_spikes = int(spike_rate * (n_samples / fs))
        spike_times = np.random.choice(n_samples, n_spikes, replace=False)

        spikes = np.zeros(n_samples)
        for spike_time in spike_times:
            # Simple spike waveform
            spike_start = max(0, spike_time - 5)
            spike_end = min(n_samples, spike_time + 15)
            spike_duration = spike_end - spike_start

            # Biphasic spike shape
            spike_t = np.linspace(-1, 3, spike_duration)
            spike_shape = (
                np.exp(-spike_t) * np.sin(np.pi * spike_t) *
                (spike_t >= 0)
            )

            spikes[spike_start:spike_end] += spike_shape * np.random.uniform(0.5, 2.0)

        # Combine components
        data[ch, :] = oscillations + spikes + noise

    # Add some cross-channel correlations
    correlation_matrix = generate_correlation_matrix(n_channels)
    data = apply_correlations(data, correlation_matrix)

    # Metadata
    metadata = {
        'n_channels': n_channels,
        'n_samples': n_samples,
        'fs': fs,
        'noise_level': noise_level,
        'spike_rate': spike_rate,
        'duration': n_samples / fs,
        'seed': seed
    }

    return data, metadata


def generate_correlation_matrix(n_channels: int, max_correlation: float = 0.3) -> np.ndarray:
    """Generate a realistic correlation matrix for neural channels."""
    # Start with identity matrix
    corr_matrix = np.eye(n_channels)

    # Add spatial correlations (nearby channels are more correlated)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Distance-based correlation
            distance = abs(i - j)
            correlation = max_correlation * np.exp(-distance / 5.0)

            # Add some randomness
            correlation *= np.random.uniform(0.5, 1.0)

            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation

    return corr_matrix


def apply_correlations(data: np.ndarray, correlation_matrix: np.ndarray) -> np.ndarray:
    """Apply correlation structure to neural data."""
    # Use Cholesky decomposition for correlation
    try:
        L = np.linalg.cholesky(correlation_matrix)
        return L @ data
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, return original data
        return data


def generate_spike_train(
    duration: float,
    spike_rate: float,
    fs: float,
    refractory_period: float = 0.002
) -> np.ndarray:
    """
    Generate a realistic spike train with refractory period.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    spike_rate : float
        Average spike rate in Hz.
    fs : float
        Sampling frequency in Hz.
    refractory_period : float, default=0.002
        Refractory period in seconds.

    Returns
    -------
    np.ndarray
        Binary spike train.
    """
    n_samples = int(duration * fs)
    refractory_samples = int(refractory_period * fs)

    spike_train = np.zeros(n_samples)
    last_spike = -refractory_samples

    # Poisson process with refractory period
    for i in range(n_samples):
        if i - last_spike > refractory_samples:
            # Probability of spike in this time bin
            p_spike = spike_rate / fs
            if np.random.random() < p_spike:
                spike_train[i] = 1
                last_spike = i

    return spike_train


def generate_synthetic_neural_data(num_channels: int, num_samples: int) -> np.ndarray:
    """
    Generate synthetic neural data.

    Parameters
    ----------
    num_channels : int
        Number of channels.
    num_samples : int
        Number of samples per channel.

    Returns
    -------
    np.ndarray
        Synthetic neural data with shape (num_channels, num_samples).
    """
    return np.random.randn(num_channels, num_samples)


def generate_synthetic_neural_data(n_channels: int, n_samples: int, fs: float = 1000.0, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    data = np.random.randn(n_channels, n_samples)
    meta = {
        'n_channels': n_channels,
        'n_samples': n_samples,
        'fs': fs,
        'seed': seed
    }
    return data, meta
