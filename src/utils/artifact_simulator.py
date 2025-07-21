"""
Artifact simulation utilities for neural data.
Supports spike, noise, drift, and custom artifact injection.

References:
- Neural data artifact characteristics
"""
from typing import Optional
import numpy as np


def inject_spike(signal: np.ndarray, severity: float = 0.5) -> np.ndarray:
    signal = signal.copy()
    num_spikes = int(severity * signal.size * 0.01)
    idx = np.random.choice(signal.size, num_spikes, replace=False)
    signal.flat[idx] += np.random.uniform(5, 10, num_spikes)
    return signal


def inject_noise(signal: np.ndarray, severity: float = 0.5) -> np.ndarray:
    signal = signal.copy()
    signal += np.random.normal(0, severity, signal.shape)
    return signal


def inject_drift(signal: np.ndarray, severity: float = 0.5) -> np.ndarray:
    signal = signal.copy()
    drift = np.linspace(0, severity, signal.shape[1])
    signal += drift
    return signal
