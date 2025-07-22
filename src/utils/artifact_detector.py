"""
Advanced artifact detection for neural data streams.
Detects spikes, noise, drift, and other anomalies.

References:
- Neural data artifact characteristics
"""
import numpy as np
from typing import Dict, Any

def detect_artifacts(signal: np.ndarray, threshold: float = 5.0) -> Dict[str, Any]:
    """
    Detects artifacts in neural signal.
    Returns dict with counts and indices for each type.
    """
    spikes = np.where(signal > threshold)
    drift = np.where(np.abs(np.diff(signal, axis=1)) > threshold / 2)
    noise_level = np.std(signal)
    return {
        "spike_count": len(spikes[0]),
        "spike_indices": spikes,
        "drift_count": len(drift[0]),
        "drift_indices": drift,
        "noise_level": noise_level
    }

class ArtifactDetector:
    def __init__(self, spike_threshold: float = 5.0, noise_std_threshold: float = 2.0, drift_threshold: float = 1.0):
        """
        Initialize detector with configurable thresholds.
        Args:
            spike_threshold: Amplitude threshold for spike detection
            noise_std_threshold: Std deviation threshold for noise detection
            drift_threshold: Slope threshold for drift detection
        """
        self.spike_threshold = spike_threshold
        self.noise_std_threshold = noise_std_threshold
        self.drift_threshold = drift_threshold

    def detect_spikes(self, signal: np.ndarray) -> np.ndarray:
        """
        Detects spikes in neural signal.
        Returns boolean mask of spike locations.
        """
        return np.abs(signal) > self.spike_threshold

    def detect_noise(self, signal: np.ndarray) -> bool:
        """
        Detects excessive noise in signal (high std deviation).
        Returns True if noise detected.
        """
        return np.std(signal) > self.noise_std_threshold

    def detect_drift(self, signal: np.ndarray) -> bool:
        """
        Detects drift by fitting a line to each channel and checking slope.
        Returns True if drift detected in any channel.
        """
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
        for ch in range(signal.shape[0]):
            x = np.arange(signal.shape[1])
            coeffs = np.polyfit(x, signal[ch], 1)
            if np.abs(coeffs[0]) > self.drift_threshold:
                return True
        return False

    def detect_all(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Runs all artifact detectors and returns summary.
        """
        return {
            "spikes": self.detect_spikes(signal),
            "noise": self.detect_noise(signal),
            "drift": self.detect_drift(signal)
        }
