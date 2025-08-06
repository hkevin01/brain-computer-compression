"""
EMG Quality Metrics

This module provides specialized quality metrics for evaluating EMG compression
algorithms, focusing on clinically relevant measures like muscle activation
detection accuracy, envelope preservation, and spectral content fidelity.

References:
- Hogrel, J. Y. "Clinical applications of surface electromyography in
  neuromuscular disorders." Clinical Neurophysiology 35.2 (2005): 59-71.
- Merletti, R., & Parker, P. A. (Eds.). "Electromyography: physiology,
  engineering, and non-invasive applications." John Wiley & Sons, 2004.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import fftfreq
from scipy.signal import hilbert
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class EMGQualityMetrics:
    """
    Comprehensive quality metrics for EMG compression evaluation.

    Provides metrics specific to EMG signal characteristics and clinical
    requirements, including muscle activation detection, envelope preservation,
    spectral fidelity, and timing accuracy.
    """

    def __init__(self, sampling_rate: float = 2000.0):
        """
        Initialize EMG quality metrics.

        Parameters
        ----------
        sampling_rate : float, default=2000.0
            EMG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2

        # EMG-specific frequency bands
        self.emg_bands = {
            'low_frequency': (20, 100),
            'mid_frequency': (100, 300),
            'high_frequency': (300, 500)
        }

    def muscle_activation_detection_accuracy(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        threshold: float = 0.1,
        min_duration: float = 0.05
    ) -> Dict[str, float]:
        """
        Evaluate muscle activation detection accuracy.

        Compares detected muscle bursts between original and reconstructed
        signals to assess preservation of activation timing and amplitude.

        Parameters
        ----------
        original : np.ndarray
            Original EMG signal (channels, samples)
        reconstructed : np.ndarray
            Reconstructed EMG signal (channels, samples)
        threshold : float, default=0.1
            Activation detection threshold (normalized)
        min_duration : float, default=0.05
            Minimum activation duration in seconds

        Returns
        -------
        dict
            Dictionary containing accuracy metrics:
            - 'detection_accuracy': Overall detection accuracy
            - 'sensitivity': True positive rate
            - 'specificity': True negative rate
            - 'temporal_accuracy': Timing accuracy of detected activations
        """
        if original.ndim == 1:
            original = original.reshape(1, -1)
            reconstructed = reconstructed.reshape(1, -1)

        results = {
            'detection_accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'temporal_accuracy': []
        }

        min_samples = int(min_duration * self.sampling_rate)

        for ch in range(original.shape[0]):
            # Detect activations in both signals
            orig_activations = self._detect_muscle_activations(
                original[ch], threshold, min_samples
            )
            recon_activations = self._detect_muscle_activations(
                reconstructed[ch], threshold, min_samples
            )

            # Calculate detection metrics
            tp = np.sum(orig_activations & recon_activations)
            tn = np.sum(~orig_activations & ~recon_activations)
            fp = np.sum(~orig_activations & recon_activations)
            fn = np.sum(orig_activations & ~recon_activations)

            # Accuracy metrics
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Temporal accuracy - compare activation timing
            temp_accuracy = self._calculate_temporal_accuracy(
                orig_activations, recon_activations
            )

            results['detection_accuracy'].append(accuracy)
            results['sensitivity'].append(sensitivity)
            results['specificity'].append(specificity)
            results['temporal_accuracy'].append(temp_accuracy)

        # Average across channels
        return {
            key: np.mean(values) for key, values in results.items()
        }

    def emg_envelope_correlation(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        envelope_method: str = 'hilbert'
    ) -> Dict[str, float]:
        """
        Calculate EMG envelope preservation correlation.

        EMG envelope contains crucial information about muscle force and
        activation patterns. This metric evaluates how well the compression
        preserves the envelope characteristics.

        Parameters
        ----------
        original : np.ndarray
            Original EMG signal
        reconstructed : np.ndarray
            Reconstructed EMG signal
        envelope_method : str, default='hilbert'
            Method for envelope calculation ('hilbert', 'rms', 'abs')

        Returns
        -------
        dict
            Envelope correlation metrics
        """
        if original.ndim == 1:
            original = original.reshape(1, -1)
            reconstructed = reconstructed.reshape(1, -1)

        correlations = []
        rmse_values = []

        for ch in range(original.shape[0]):
            # Calculate envelopes
            orig_envelope = self._calculate_envelope(
                original[ch], method=envelope_method
            )
            recon_envelope = self._calculate_envelope(
                reconstructed[ch], method=envelope_method
            )

            # Correlation
            correlation, _ = pearsonr(orig_envelope, recon_envelope)
            correlations.append(correlation)

            # RMSE of envelopes
            rmse = np.sqrt(np.mean((orig_envelope - recon_envelope) ** 2))
            rmse_values.append(rmse)

        return {
            'envelope_correlation': np.mean(correlations),
            'envelope_rmse': np.mean(rmse_values),
            'envelope_correlation_std': np.std(correlations)
        }

    def emg_spectral_fidelity(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        frequency_range: Tuple[float, float] = (20, 500)
    ) -> Dict[str, float]:
        """
        Evaluate spectral content preservation in EMG frequency range.

        EMG signals contain important spectral information related to muscle
        fiber types, firing rates, and fatigue. This metric assesses how well
        the compression preserves spectral characteristics.

        Parameters
        ----------
        original : np.ndarray
            Original EMG signal
        reconstructed : np.ndarray
            Reconstructed EMG signal
        frequency_range : tuple, default=(20, 500)
            Frequency range for EMG analysis in Hz

        Returns
        -------
        dict
            Spectral fidelity metrics
        """
        if original.ndim == 1:
            original = original.reshape(1, -1)
            reconstructed = reconstructed.reshape(1, -1)

        spectral_correlations = []
        spectral_distances = []
        band_powers = {'low': [], 'mid': [], 'high': []}

        for ch in range(original.shape[0]):
            # Calculate power spectral densities
            orig_psd = self._calculate_psd(original[ch])
            recon_psd = self._calculate_psd(reconstructed[ch])

            # Limit to EMG frequency range
            freqs = fftfreq(len(original[ch]), 1/self.sampling_rate)
            freq_mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])

            orig_psd_limited = orig_psd[freq_mask]
            recon_psd_limited = recon_psd[freq_mask]

            # Spectral correlation
            correlation, _ = pearsonr(orig_psd_limited, recon_psd_limited)
            spectral_correlations.append(correlation)

            # Spectral distance (Jensen-Shannon divergence)
            js_distance = self._jensen_shannon_distance(
                orig_psd_limited, recon_psd_limited
            )
            spectral_distances.append(js_distance)

            # Band power preservation
            band_power_metrics = self._calculate_band_power_preservation(
                original[ch], reconstructed[ch]
            )
            for band, value in band_power_metrics.items():
                band_powers[band].append(value)

        return {
            'spectral_correlation': np.mean(spectral_correlations),
            'spectral_distance': np.mean(spectral_distances),
            'low_freq_power_preservation': np.mean(band_powers['low']),
            'mid_freq_power_preservation': np.mean(band_powers['mid']),
            'high_freq_power_preservation': np.mean(band_powers['high'])
        }

    def emg_timing_precision(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        onset_threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Evaluate timing precision of muscle activation onset and offset.

        Critical for applications like prosthetic control where timing of
        muscle activations is crucial for real-time control.

        Parameters
        ----------
        original : np.ndarray
            Original EMG signal
        reconstructed : np.ndarray
            Reconstructed EMG signal
        onset_threshold : float, default=0.05
            Threshold for onset/offset detection (as fraction of max)

        Returns
        -------
        dict
            Timing precision metrics
        """
        if original.ndim == 1:
            original = original.reshape(1, -1)
            reconstructed = reconstructed.reshape(1, -1)

        onset_errors = []
        offset_errors = []

        for ch in range(original.shape[0]):
            # Detect onset/offset times
            orig_events = self._detect_onset_offset(original[ch], onset_threshold)
            recon_events = self._detect_onset_offset(reconstructed[ch], onset_threshold)

            # Match events and calculate timing errors
            onset_error, offset_error = self._match_and_calculate_timing_errors(
                orig_events, recon_events
            )

            onset_errors.extend(onset_error)
            offset_errors.extend(offset_error)

        return {
            'mean_onset_error_ms': np.mean(onset_errors) * 1000 / self.sampling_rate,
            'std_onset_error_ms': np.std(onset_errors) * 1000 / self.sampling_rate,
            'mean_offset_error_ms': np.mean(offset_errors) * 1000 / self.sampling_rate,
            'std_offset_error_ms': np.std(offset_errors) * 1000 / self.sampling_rate,
            'timing_precision_score': 1.0 / (1.0 + np.mean(onset_errors + offset_errors))
        }

    def comprehensive_emg_quality_score(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive EMG quality score combining all metrics.

        Parameters
        ----------
        original : np.ndarray
            Original EMG signal
        reconstructed : np.ndarray
            Reconstructed EMG signal
        weights : dict, optional
            Weights for combining different metrics

        Returns
        -------
        dict
            Comprehensive quality metrics
        """
        if weights is None:
            weights = {
                'activation_detection': 0.3,
                'envelope_correlation': 0.25,
                'spectral_fidelity': 0.25,
                'timing_precision': 0.2
            }

        # Calculate individual metrics
        activation_metrics = self.muscle_activation_detection_accuracy(
            original, reconstructed
        )
        envelope_metrics = self.emg_envelope_correlation(original, reconstructed)
        spectral_metrics = self.emg_spectral_fidelity(original, reconstructed)
        timing_metrics = self.emg_timing_precision(original, reconstructed)

        # Combine into overall score
        overall_score = (
            weights['activation_detection'] * activation_metrics['detection_accuracy'] +
            weights['envelope_correlation'] * envelope_metrics['envelope_correlation'] +
            weights['spectral_fidelity'] * spectral_metrics['spectral_correlation'] +
            weights['timing_precision'] * timing_metrics['timing_precision_score']
        )

        return {
            'overall_quality_score': overall_score,
            'activation_detection': activation_metrics,
            'envelope_preservation': envelope_metrics,
            'spectral_fidelity': spectral_metrics,
            'timing_precision': timing_metrics
        }

    def _detect_muscle_activations(
        self,
        signal: np.ndarray,
        threshold: float,
        min_samples: int
    ) -> np.ndarray:
        """Detect muscle activation periods."""
        # Calculate envelope
        envelope = np.abs(hilbert(signal))

        # Smooth envelope
        window_size = int(0.02 * self.sampling_rate)  # 20ms
        if window_size % 2 == 0:
            window_size += 1
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)

        # Normalize and threshold
        envelope_norm = envelope_smooth / (np.max(envelope_smooth) + 1e-8)
        activations = envelope_norm > threshold

        # Remove short activations
        return self._remove_short_segments(activations, min_samples)

    def _remove_short_segments(
        self,
        binary_signal: np.ndarray,
        min_length: int
    ) -> np.ndarray:
        """Remove segments shorter than minimum length."""
        # Find segment boundaries
        diff = np.diff(binary_signal.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if binary_signal[0]:
            starts = np.concatenate([[0], starts])
        if binary_signal[-1]:
            ends = np.concatenate([ends, [len(binary_signal)]])

        # Remove short segments
        filtered = binary_signal.copy()
        for start, end in zip(starts, ends):
            if end - start < min_length:
                filtered[start:end] = False

        return filtered

    def _calculate_temporal_accuracy(
        self,
        orig_activations: np.ndarray,
        recon_activations: np.ndarray
    ) -> float:
        """Calculate temporal accuracy of activation detection."""
        # Simple overlap-based accuracy
        overlap = np.sum(orig_activations & recon_activations)
        union = np.sum(orig_activations | recon_activations)

        return overlap / union if union > 0 else 0.0

    def _calculate_envelope(
        self,
        signal: np.ndarray,
        method: str = 'hilbert'
    ) -> np.ndarray:
        """Calculate signal envelope using specified method."""
        if method == 'hilbert':
            return np.abs(hilbert(signal))
        elif method == 'rms':
            window_size = int(0.025 * self.sampling_rate)  # 25ms RMS window
            return self._moving_rms(signal, window_size)
        elif method == 'abs':
            return np.abs(signal)
        else:
            raise ValueError(f"Unknown envelope method: {method}")

    def _moving_rms(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving RMS of signal."""
        rms = np.zeros(len(signal))
        for i in range(len(signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2 + 1)
            rms[i] = np.sqrt(np.mean(signal[start_idx:end_idx] ** 2))
        return rms

    def _calculate_psd(self, signal: np.ndarray) -> np.ndarray:
        """Calculate power spectral density."""
        # Use Welch's method for better PSD estimation
        freqs, psd = signal.welch(
            signal,
            fs=self.sampling_rate,
            nperseg=min(len(signal) // 4, 1024)
        )
        return psd

    def _jensen_shannon_distance(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """Calculate Jensen-Shannon distance between two distributions."""
        # Normalize to probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate Jensen-Shannon divergence
        m = 0.5 * (p + q)
        js_divergence = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

        # Return distance (square root of divergence)
        return np.sqrt(js_divergence)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)

        return np.sum(p * np.log(p / q))

    def _calculate_band_power_preservation(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """Calculate power preservation in different frequency bands."""
        preservation = {}

        for band_name, (low_freq, high_freq) in self.emg_bands.items():
            # Filter signals to band
            orig_band = self._bandpass_filter(original, low_freq, high_freq)
            recon_band = self._bandpass_filter(reconstructed, low_freq, high_freq)

            # Calculate power
            orig_power = np.mean(orig_band ** 2)
            recon_power = np.mean(recon_band ** 2)

            # Power preservation ratio
            preservation[band_name] = recon_power / (orig_power + 1e-8)

        return preservation

    def _bandpass_filter(
        self,
        signal: np.ndarray,
        low_freq: float,
        high_freq: float
    ) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq, nyquist * 0.95) / nyquist

        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, signal)

    def _detect_onset_offset(
        self,
        signal: np.ndarray,
        threshold: float
    ) -> Dict[str, List[int]]:
        """Detect onset and offset times of muscle activations."""
        # Calculate envelope
        envelope = np.abs(hilbert(signal))
        envelope_norm = envelope / (np.max(envelope) + 1e-8)

        # Find crossings
        above_threshold = envelope_norm > threshold
        diff = np.diff(above_threshold.astype(int))

        onsets = np.where(diff == 1)[0] + 1
        offsets = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            onsets = np.concatenate([[0], onsets])
        if above_threshold[-1]:
            offsets = np.concatenate([offsets, [len(signal) - 1]])

        return {'onsets': onsets.tolist(), 'offsets': offsets.tolist()}

    def _match_and_calculate_timing_errors(
        self,
        orig_events: Dict[str, List[int]],
        recon_events: Dict[str, List[int]]
    ) -> Tuple[List[float], List[float]]:
        """Match events and calculate timing errors."""
        onset_errors = []
        offset_errors = []

        # Match onsets
        for orig_onset in orig_events['onsets']:
            # Find closest reconstructed onset
            if recon_events['onsets']:
                distances = [abs(orig_onset - recon_onset)
                           for recon_onset in recon_events['onsets']]
                min_distance = min(distances)
                if min_distance < 0.1 * self.sampling_rate:  # Within 100ms
                    onset_errors.append(min_distance)

        # Match offsets
        for orig_offset in orig_events['offsets']:
            # Find closest reconstructed offset
            if recon_events['offsets']:
                distances = [abs(orig_offset - recon_offset)
                           for recon_offset in recon_events['offsets']]
                min_distance = min(distances)
                if min_distance < 0.1 * self.sampling_rate:  # Within 100ms
                    offset_errors.append(min_distance)

        return onset_errors, offset_errors


# Convenience functions for common EMG quality assessments
def evaluate_emg_compression_quality(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampling_rate: float = 2000.0,
    comprehensive: bool = True
) -> Dict[str, float]:
    """
    Evaluate EMG compression quality with standard metrics.

    Parameters
    ----------
    original : np.ndarray
        Original EMG signal
    reconstructed : np.ndarray
        Reconstructed EMG signal
    sampling_rate : float, default=2000.0
        Sampling rate in Hz
    comprehensive : bool, default=True
        Whether to calculate comprehensive quality score

    Returns
    -------
    dict
        Quality metrics
    """
    metrics = EMGQualityMetrics(sampling_rate)

    if comprehensive:
        return metrics.comprehensive_emg_quality_score(original, reconstructed)
    else:
        return {
            'envelope_correlation': metrics.emg_envelope_correlation(
                original, reconstructed
            )['envelope_correlation'],
            'spectral_correlation': metrics.emg_spectral_fidelity(
                original, reconstructed
            )['spectral_correlation']
        }


def quick_emg_quality_check(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampling_rate: float = 2000.0
) -> float:
    """
    Quick EMG quality check returning single score.

    Parameters
    ----------
    original : np.ndarray
        Original EMG signal
    reconstructed : np.ndarray
        Reconstructed EMG signal
    sampling_rate : float, default=2000.0
        Sampling rate in Hz

    Returns
    -------
    float
        Overall quality score (0-1, higher is better)
    """
    metrics = EMGQualityMetrics(sampling_rate)
    result = metrics.comprehensive_emg_quality_score(original, reconstructed)
    return result['overall_quality_score']
