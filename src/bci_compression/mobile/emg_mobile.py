"""
Mobile optimization extensions for wearable EMG devices.

This module extends the mobile compression framework to handle EMG-specific
requirements for wearable devices, including power optimization for muscle
monitoring applications and real-time processing constraints.
"""

import logging
import time
from typing import Dict

import numpy as np

from ..algorithms.emg_compression import EMGLZCompressor
from ..mobile.mobile_compressor import MobileBCICompressor
from ..mobile.power_optimizer import PowerOptimizer

logger = logging.getLogger(__name__)


class MobileEMGCompressor(MobileBCICompressor):
    """
    Mobile-optimized EMG compressor for wearable devices.

    Extends the mobile BCI compressor with EMG-specific optimizations:
    - Lower sampling rate handling (1-2kHz vs 30kHz)
    - Muscle activation-aware power management
    - EMG-specific buffering strategies
    - Clinical data preservation priorities
    """

    def __init__(
        self,
        algorithm: str = "emg_mobile_lz",
        quality_level: float = 0.8,
        buffer_size: int = 256,  # Smaller buffer for EMG
        power_mode: str = "emg_optimized",
        max_memory_mb: int = 25,  # Lower memory for wearables
        emg_sampling_rate: float = 2000.0,
        activation_threshold: float = 0.1
    ):
        """
        Initialize mobile EMG compressor.

        Parameters
        ----------
        algorithm : str, default="emg_mobile_lz"
            EMG compression algorithm
        quality_level : float, default=0.8
            Quality level for lossy compression
        buffer_size : int, default=256
            Processing buffer size (optimized for EMG)
        power_mode : str, default="emg_optimized"
            Power mode optimized for EMG characteristics
        max_memory_mb : int, default=25
            Maximum memory usage for wearable devices
        emg_sampling_rate : float, default=2000.0
            EMG sampling rate in Hz
        activation_threshold : float, default=0.1
            Threshold for muscle activation detection
        """
        # Initialize parent with EMG-optimized parameters
        super().__init__(
            algorithm=algorithm,
            quality_level=quality_level,
            buffer_size=buffer_size,
            power_mode=power_mode,
            max_memory_mb=max_memory_mb
        )

        # EMG-specific parameters
        self.emg_sampling_rate = emg_sampling_rate
        self.activation_threshold = activation_threshold

        # EMG power optimization
        self.emg_power_optimizer = EMGPowerOptimizer(
            sampling_rate=emg_sampling_rate,
            activation_threshold=activation_threshold
        )

        # Muscle activation state tracking
        self.muscle_state = {
            'is_active': False,
            'activation_level': 0.0,
            'time_since_activation': 0.0
        }

        # EMG-specific performance tracking
        self.emg_metrics = {
            'activation_periods': 0,
            'rest_periods': 0,
            'power_savings': 0.0,
            'clinical_quality_maintained': True
        }

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress EMG data with mobile optimizations.

        Parameters
        ----------
        data : np.ndarray
            EMG data (channels, samples)

        Returns
        -------
        bytes
            Compressed EMG data
        """
        start_time = time.time()

        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Detect muscle activation state
        activation_state = self._detect_muscle_activation_state(data)
        self._update_muscle_state(activation_state)

        # Apply EMG power optimization
        optimized_params = self.emg_power_optimizer.optimize_for_state(
            self.muscle_state, self.power_mode
        )

        # Adjust compression parameters based on muscle state
        self._adjust_compression_for_muscle_state(optimized_params)

        # Perform compression with adjusted parameters
        if self.algorithm == "emg_mobile_lz":
            compressed = self._compress_emg_mobile_lz(data)
        elif self.algorithm == "emg_lightweight_quant":
            compressed = self._compress_emg_lightweight_quant(data)
        else:
            # Fall back to parent implementation
            compressed = super().compress(data)

        # Update EMG-specific metrics
        processing_time = time.time() - start_time
        self._update_emg_metrics(data, compressed, processing_time)

        return compressed

    def _detect_muscle_activation_state(self, data: np.ndarray) -> Dict[str, float]:
        """Detect current muscle activation state."""
        activation_levels = []

        for ch in range(data.shape[0]):
            # Calculate RMS envelope
            rms_envelope = self._calculate_rms_envelope(data[ch])

            # Normalize and check activation
            max_val = np.max(rms_envelope)
            activation_level = max_val / (self.activation_threshold + 1e-8)
            activation_levels.append(activation_level)

        # Overall activation state
        max_activation = np.max(activation_levels)
        mean_activation = np.mean(activation_levels)

        return {
            'max_activation': max_activation,
            'mean_activation': mean_activation,
            'is_active': max_activation > 1.0,
            'n_active_channels': np.sum(np.array(activation_levels) > 1.0)
        }

    def _calculate_rms_envelope(self, signal: np.ndarray, window_ms: float = 25) -> np.ndarray:
        """Calculate RMS envelope of EMG signal."""
        window_samples = int(window_ms * self.emg_sampling_rate / 1000)
        envelope = np.zeros(len(signal))

        for i in range(len(signal)):
            start_idx = max(0, i - window_samples // 2)
            end_idx = min(len(signal), i + window_samples // 2 + 1)
            envelope[i] = np.sqrt(np.mean(signal[start_idx:end_idx] ** 2))

        return envelope

    def _update_muscle_state(self, activation_state: Dict[str, float]):
        """Update muscle state tracking."""
        self.muscle_state.update({
            'is_active': activation_state['is_active'],
            'activation_level': activation_state['mean_activation'],
            'time_since_activation': 0.0 if activation_state['is_active']
                                   else self.muscle_state['time_since_activation'] + 0.1
        })

        # Update EMG metrics
        if activation_state['is_active']:
            self.emg_metrics['activation_periods'] += 1
        else:
            self.emg_metrics['rest_periods'] += 1

    def _adjust_compression_for_muscle_state(self, optimized_params: Dict[str, float]):
        """Adjust compression parameters based on muscle activation state."""
        if self.muscle_state['is_active']:
            # During muscle activation: prioritize quality
            self.quality_level = min(0.95, self.quality_level * 1.1)
            # Use smaller buffers for lower latency
            self.buffer_size = max(128, int(self.buffer_size * 0.8))
        else:
            # During rest: optimize for power
            self.quality_level = max(0.6, self.quality_level * 0.9)
            # Use larger buffers for efficiency
            self.buffer_size = min(512, int(self.buffer_size * 1.2))

        # Apply power optimizer suggestions
        if 'compression_aggressiveness' in optimized_params:
            aggressiveness = optimized_params['compression_aggressiveness']
            self.quality_level *= (2.0 - aggressiveness)  # Inverse relationship

    def _compress_emg_mobile_lz(self, data: np.ndarray) -> bytes:
        """EMG-optimized mobile LZ compression."""
        # Use EMG-aware LZ compressor with mobile optimizations
        emg_compressor = EMGLZCompressor(
            activation_threshold=self.activation_threshold,
            sampling_rate=self.emg_sampling_rate,
            pattern_buffer_size=self.buffer_size
        )

        return emg_compressor.compress(data)

    def _compress_emg_lightweight_quant(self, data: np.ndarray) -> bytes:
        """EMG-optimized lightweight quantization."""
        compressed_channels = []

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            # Apply EMG-specific preprocessing
            filtered_data = self._emg_preprocess(channel_data)

            # Adaptive quantization based on activation state
            if self.muscle_state['is_active']:
                # Higher precision during activation
                n_bits = 12
            else:
                # Lower precision during rest
                n_bits = 8

            # Quantize
            quantized = self._adaptive_quantize(filtered_data, n_bits)
            compressed_channels.append(quantized.tobytes())

        # Pack channels
        return self._pack_emg_channels(compressed_channels)

    def _emg_preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Apply EMG-specific preprocessing."""
        # Simple high-pass filter to remove DC and low-frequency artifacts
        from scipy import signal as sp_signal

        # High-pass filter at 20 Hz
        nyquist = self.emg_sampling_rate / 2
        high_cutoff = 20 / nyquist

        if high_cutoff < 1.0:
            b, a = sp_signal.butter(2, high_cutoff, btype='high')
            filtered = sp_signal.filtfilt(b, a, signal)
        else:
            filtered = signal

        return filtered

    def _adaptive_quantize(self, data: np.ndarray, n_bits: int) -> np.ndarray:
        """Adaptive quantization for EMG data."""
        # Robust quantization using percentiles to handle outliers
        data_min = np.percentile(data, 1)
        data_max = np.percentile(data, 99)
        data_range = data_max - data_min

        if data_range < 1e-8:
            return np.zeros(len(data), dtype=np.uint16)

        # Quantize
        levels = 2 ** n_bits
        normalized = np.clip((data - data_min) / data_range, 0, 1)
        quantized = np.round(normalized * (levels - 1)).astype(np.uint16)

        return quantized

    def _pack_emg_channels(self, compressed_channels: list) -> bytes:
        """Pack compressed EMG channels."""
        # Simple packing - in practice, use more efficient format
        import pickle
        return pickle.dumps({
            'channels': compressed_channels,
            'metadata': {
                'muscle_state': self.muscle_state,
                'shape': self._last_shape,
                'dtype': str(self._last_dtype)
            }
        })

    def _update_emg_metrics(
        self,
        original_data: np.ndarray,
        compressed_data: bytes,
        processing_time: float
    ):
        """Update EMG-specific performance metrics."""
        # Calculate power savings estimate
        if self.muscle_state['is_active']:
            power_multiplier = 1.2  # Higher power during processing
        else:
            power_multiplier = 0.8  # Power savings during rest

        self.emg_metrics['power_savings'] = (1.0 - power_multiplier) * 100

        # Update parent metrics
        self.processing_times.append(processing_time)

        original_size = original_data.nbytes
        compressed_size = len(compressed_data)
        self.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        self.compression_ratios.append(self.compression_ratio)

        self.memory_usage.append(compressed_size / (1024 * 1024))

    def get_emg_performance_stats(self) -> Dict[str, float]:
        """Get EMG-specific performance statistics."""
        base_stats = self.get_performance_stats()

        emg_stats = {
            'activation_detection_rate': (
                self.emg_metrics['activation_periods'] /
                max(1, self.emg_metrics['activation_periods'] + self.emg_metrics['rest_periods'])
            ),
            'estimated_power_savings_percent': self.emg_metrics['power_savings'],
            'muscle_activation_level': self.muscle_state['activation_level'],
            'clinical_quality_maintained': self.emg_metrics['clinical_quality_maintained']
        }

        # Combine with base stats
        base_stats.update(emg_stats)
        return base_stats


class EMGPowerOptimizer(PowerOptimizer):
    """
    Power optimizer specialized for EMG wearable devices.

    Optimizes power consumption based on muscle activation patterns
    and clinical data quality requirements.
    """

    def __init__(
        self,
        sampling_rate: float = 2000.0,
        activation_threshold: float = 0.1
    ):
        """
        Initialize EMG power optimizer.

        Parameters
        ----------
        sampling_rate : float, default=2000.0
            EMG sampling rate
        activation_threshold : float, default=0.1
            Muscle activation threshold
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.activation_threshold = activation_threshold

        # EMG-specific power profiles
        self.emg_power_profiles = {
            'clinical_priority': {
                'compression_aggressiveness': 0.3,
                'processing_frequency': 1.0,
                'quality_threshold': 0.9
            },
            'battery_save': {
                'compression_aggressiveness': 0.8,
                'processing_frequency': 0.5,
                'quality_threshold': 0.7
            },
            'balanced_emg': {
                'compression_aggressiveness': 0.5,
                'processing_frequency': 0.8,
                'quality_threshold': 0.8
            }
        }

    def optimize_for_state(
        self,
        muscle_state: Dict[str, float],
        power_mode: str
    ) -> Dict[str, float]:
        """
        Optimize parameters based on muscle activation state.

        Parameters
        ----------
        muscle_state : dict
            Current muscle activation state
        power_mode : str
            Power optimization mode

        Returns
        -------
        dict
            Optimized parameters
        """
        # Get base profile
        if power_mode == "emg_optimized":
            if muscle_state['is_active']:
                profile = self.emg_power_profiles['clinical_priority']
            else:
                profile = self.emg_power_profiles['battery_save']
        else:
            profile = self.emg_power_profiles.get(
                power_mode, self.emg_power_profiles['balanced_emg']
            )

        # Adjust based on activation level
        activation_factor = muscle_state.get('activation_level', 0.0)

        optimized_params = profile.copy()

        # Higher activation = lower compression aggressiveness
        optimized_params['compression_aggressiveness'] *= (2.0 - activation_factor) / 2.0

        # Adjust processing frequency based on activation
        if muscle_state['is_active']:
            optimized_params['processing_frequency'] = min(
                1.0, optimized_params['processing_frequency'] * 1.2
            )

        return optimized_params
