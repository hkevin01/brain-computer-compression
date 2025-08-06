"""
GPU-accelerated signal processing for neural data using CUDA.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cudf
    import cupy as cp
    import cusignal
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Create fallback implementations
    import numpy as cp  # Use numpy as cp fallback
    import scipy.signal as cusignal  # Use scipy as cusignal fallback
    warnings.warn("CUDA libraries not available. Using CPU fallback.")


class GPUNeuralProcessor:
    """
    GPU-accelerated neural signal processing pipeline.
    Provides CUDA-accelerated versions of common operations.
    Falls back to CPU if CUDA is not available.
    """

    def __init__(self, sampling_rate: float = 30000.0):
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        self.cuda_available = CUDA_AVAILABLE

        if self.cuda_available:
            # Initialize CUDA memory pool
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def to_gpu(self, data: np.ndarray) -> Any:
        """Transfer data to GPU memory."""
        return cp.asarray(data) if self.cuda_available else data

    def to_cpu(self, data: Any) -> np.ndarray:
        """Transfer data back to CPU memory."""
        return cp.asnumpy(data) if self.cuda_available else data

    def bandpass_filter(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float,
        order: int = 4,
        filter_type: str = 'butterworth'
    ) -> np.ndarray:
        """GPU-accelerated bandpass filter."""
        if not self.cuda_available:
            return self._cpu_bandpass_filter(data, low_freq, high_freq, order, filter_type)

        # Validate frequency range
        if low_freq >= high_freq:
            raise ValueError("Low frequency must be less than high frequency")
        if high_freq >= self.nyquist_freq:
            warnings.warn(f"High frequency {high_freq} Hz is close to Nyquist frequency")
            high_freq = self.nyquist_freq * 0.95

        # Transfer to GPU
        d_data = self.to_gpu(data)

        # Normalize frequencies
        low_norm = low_freq / self.nyquist_freq
        high_norm = high_freq / self.nyquist_freq

        # Design filter on CPU (negligible compute cost)
        if filter_type == 'butterworth':
            b, a = cusignal.butter(order, [low_norm, high_norm], btype='band')
        elif filter_type == 'elliptic':
            b, a = cusignal.ellip(order, 1, 40, [low_norm, high_norm], btype='band')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # Apply filter on GPU
        if d_data.ndim == 1:
            filtered_data = cusignal.filtfilt(b, a, d_data)
        else:
            filtered_data = cp.zeros_like(d_data)
            for ch in range(d_data.shape[0]):
                filtered_data[ch] = cusignal.filtfilt(b, a, d_data[ch])

        return self.to_cpu(filtered_data)

    def notch_filter(
        self,
        data: np.ndarray,
        notch_freq: float = 60.0,
        quality_factor: float = 30.0
    ) -> np.ndarray:
        """GPU-accelerated notch filter."""
        if not self.cuda_available:
            return self._cpu_notch_filter(data, notch_freq, quality_factor)

        # Transfer to GPU
        d_data = self.to_gpu(data)

        # Design notch filter
        nyq = self.sampling_rate / 2.0
        freq = notch_freq / nyq
        b, a = cusignal.iirnotch(freq, quality_factor)

        # Apply filter on GPU
        if d_data.ndim == 1:
            filtered_data = cusignal.filtfilt(b, a, d_data)
        else:
            filtered_data = cp.zeros_like(d_data)
            for ch in range(d_data.shape[0]):
                filtered_data[ch] = cusignal.filtfilt(b, a, d_data[ch])

        return self.to_cpu(filtered_data)

    def normalize_signals(
        self,
        data: np.ndarray,
        method: str = 'zscore',
        axis: Optional[int] = None
    ) -> np.ndarray:
        """GPU-accelerated signal normalization."""
        if not self.cuda_available:
            return self._cpu_normalize_signals(data, method, axis)

        d_data = self.to_gpu(data)

        if method == 'zscore':
            mean = cp.mean(d_data, axis=axis, keepdims=True)
            std = cp.std(d_data, axis=axis, keepdims=True)
            normalized = (d_data - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = cp.min(d_data, axis=axis, keepdims=True)
            max_val = cp.max(d_data, axis=axis, keepdims=True)
            normalized = (d_data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median = cp.median(d_data, axis=axis, keepdims=True)
            mad = cp.median(cp.abs(d_data - median), axis=axis, keepdims=True)
            normalized = (d_data - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return self.to_cpu(normalized)

    def preprocess_pipeline(
        self,
        data: np.ndarray,
        config: Dict = None,
        bandpass_range: Tuple[float, float] = (1.0, 500.0),
        notch_freq: float = 60.0,
        normalize: bool = True,
        remove_artifacts: bool = True
    ) -> np.ndarray:
        """GPU-accelerated preprocessing pipeline."""
        if not self.cuda_available:
            return self._cpu_preprocess_pipeline(
                data, config, bandpass_range, notch_freq, normalize, remove_artifacts)

        # Transfer data to GPU once
        d_data = self.to_gpu(data)
        steps = []

        if config is not None:
            # Apply configuration-based processing
            bandpass_cfg = config.get('bandpass', {})
            low = bandpass_cfg.get('low_freq', 1.0)
            high = bandpass_cfg.get('high_freq', 500.0)
            d_data = self.to_gpu(self.bandpass_filter(self.to_cpu(d_data), low, high))
            steps.append(f"Bandpass filter: {low}-{high} Hz")

            notch_cfg = config.get('notch', {})
            notch_freq = notch_cfg.get('notch_freq', 60.0)
            d_data = self.to_gpu(self.notch_filter(self.to_cpu(d_data), notch_freq))
            steps.append(f"Notch filter: {notch_freq} Hz")

            if config.get('normalize', True):
                d_data = self.to_gpu(self.normalize_signals(self.to_cpu(d_data)))
                steps.append("Z-score normalization")

        else:
            # Use default parameters
            if bandpass_range is not None:
                d_data = self.to_gpu(self.bandpass_filter(
                    self.to_cpu(d_data), bandpass_range[0], bandpass_range[1]))
                steps.append(f"Bandpass filter: {bandpass_range[0]}-{bandpass_range[1]} Hz")

            if notch_freq is not None:
                d_data = self.to_gpu(self.notch_filter(self.to_cpu(d_data), notch_freq))
                steps.append(f"Notch filter: {notch_freq} Hz")

            if normalize:
                d_data = self.to_gpu(self.normalize_signals(self.to_cpu(d_data)))
                steps.append("Z-score normalization")

        # Transfer final result back to CPU
        processed_data = self.to_cpu(d_data)

        # Ensure output shape matches input shape
        if processed_data.shape != data.shape:
            processed_data = processed_data.reshape(data.shape)

        return processed_data

    def _cpu_bandpass_filter(self, data, low_freq, high_freq, order, filter_type):
        """CPU fallback for bandpass filter."""
        from scipy import signal

        if low_freq >= high_freq:
            raise ValueError("Low frequency must be less than high frequency")

        low_norm = low_freq / self.nyquist_freq
        high_norm = high_freq / self.nyquist_freq

        if filter_type == 'butterworth':
            b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        elif filter_type == 'elliptic':
            b, a = signal.ellip(order, 1, 40, [low_norm, high_norm], btype='band')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        if data.ndim == 1:
            return signal.filtfilt(b, a, data)

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        return filtered_data

    def _cpu_notch_filter(self, data, notch_freq, quality_factor):
        """CPU fallback for notch filter."""
        from scipy import signal

        nyq = self.sampling_rate / 2.0
        freq = notch_freq / nyq
        b, a = signal.iirnotch(freq, quality_factor)

        if data.ndim == 1:
            return signal.filtfilt(b, a, data)

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        return filtered_data

    def _cpu_normalize_signals(self, data, method, axis):
        """CPU fallback for normalization."""
        if method == 'zscore':
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            return (data - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            return (data - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _cpu_preprocess_pipeline(self, data, config, bandpass_range, notch_freq, normalize, remove_artifacts):
        """CPU fallback for preprocessing pipeline."""
        processed_data = data.copy()

        if bandpass_range is not None:
            processed_data = self._cpu_bandpass_filter(
                processed_data, bandpass_range[0], bandpass_range[1], 4, 'butterworth')

        if notch_freq is not None:
            processed_data = self._cpu_notch_filter(
                processed_data, notch_freq, 30.0)

        if normalize:
            processed_data = self._cpu_normalize_signals(
                processed_data, 'zscore', None)

        return processed_data
