"""
Basic signal processing pipeline for neural data preprocessing.

This module provides fundamental signal processing operations for neural data,
including filtering, normalization, and preprocessing steps required for
brain-computer interface applications.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings


class NeuralSignalProcessor:
    """
    Core signal processing pipeline for neural data preprocessing.
    
    This class provides essential signal processing operations optimized
    for neural signals, including filtering, artifact removal, and
    feature extraction suitable for real-time BCI applications.
    """
    
    def __init__(self, sampling_rate: float = 30000.0):
        """
        Initialize the neural signal processor.
        
        Parameters
        ----------
        sampling_rate : float, default=30000.0
            Sampling rate of the neural signals in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        
    def bandpass_filter(
        self, 
        data: np.ndarray, 
        low_freq: float, 
        high_freq: float,
        order: int = 4,
        filter_type: str = 'butterworth'
    ) -> np.ndarray:
        """
        Apply bandpass filter to neural signals.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data with shape (channels, samples) or (samples,)
        low_freq : float
            Low cutoff frequency in Hz
        high_freq : float
            High cutoff frequency in Hz
        order : int, default=4
            Filter order
        filter_type : str, default='butterworth'
            Type of filter ('butterworth', 'elliptic', 'chebyshev1')
            
        Returns
        -------
        np.ndarray
            Filtered neural data
        """
        # Validate frequency range
        if low_freq >= high_freq:
            raise ValueError("Low frequency must be less than high frequency")
        if high_freq >= self.nyquist_freq:
            warnings.warn(f"High frequency {high_freq} Hz is close to Nyquist frequency {self.nyquist_freq} Hz")
            high_freq = self.nyquist_freq * 0.95
            
        # Normalize frequencies
        low_norm = low_freq / self.nyquist_freq
        high_norm = high_freq / self.nyquist_freq
        
        # Design filter
        if filter_type == 'butterworth':
            b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        elif filter_type == 'elliptic':
            b, a = signal.ellip(order, 1, 40, [low_norm, high_norm], btype='band')
        elif filter_type == 'chebyshev1':
            b, a = signal.cheby1(order, 1, [low_norm, high_norm], btype='band')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter
        if data.ndim == 1:
            # Single channel
            filtered_data = signal.filtfilt(b, a, data)
        else:
            # Multi-channel
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
        return filtered_data
    
    def notch_filter(
        self, 
        data: np.ndarray, 
        notch_freq: float = 60.0,
        quality_factor: float = 30.0
    ) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data
        notch_freq : float, default=60.0
            Frequency to notch out (typically 50Hz or 60Hz)
        quality_factor : float, default=30.0
            Quality factor of the notch filter
            
        Returns
        -------
        np.ndarray
            Filtered neural data
        """
        # Design notch filter
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Apply filter
        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
        return filtered_data
    
    def normalize_signals(
        self, 
        data: np.ndarray, 
        method: str = 'zscore',
        axis: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize neural signals.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data
        method : str, default='zscore'
            Normalization method ('zscore', 'minmax', 'robust')
        axis : int, optional
            Axis along which to normalize
            
        Returns
        -------
        tuple
            Normalized data and normalization parameters
        """
        if method == 'zscore':
            mean_vals = np.mean(data, axis=axis, keepdims=True)
            std_vals = np.std(data, axis=axis, keepdims=True)
            normalized_data = (data - mean_vals) / (std_vals + 1e-8)
            params = {'mean': mean_vals, 'std': std_vals, 'method': 'zscore'}
            
        elif method == 'minmax':
            min_vals = np.min(data, axis=axis, keepdims=True)
            max_vals = np.max(data, axis=axis, keepdims=True)
            normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)
            params = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
            
        elif method == 'robust':
            median_vals = np.median(data, axis=axis, keepdims=True)
            mad_vals = np.median(np.abs(data - median_vals), axis=axis, keepdims=True)
            normalized_data = (data - median_vals) / (mad_vals + 1e-8)
            params = {'median': median_vals, 'mad': mad_vals, 'method': 'robust'}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized_data, params
    
    def extract_features(
        self, 
        data: np.ndarray, 
        window_size: int = 1000,
        overlap: float = 0.5
    ) -> dict:
        """
        Extract basic features from neural signals for BCI applications.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data with shape (channels, samples)
        window_size : int, default=1000
            Window size for feature extraction
        overlap : float, default=0.5
            Overlap ratio between windows
            
        Returns
        -------
        dict
            Dictionary containing extracted features
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        step_size = int(window_size * (1 - overlap))
        n_windows = (n_samples - window_size) // step_size + 1
        
        features = {
            'power_spectral_density': [],
            'mean_power': [],
            'peak_frequency': [],
            'spectral_centroid': [],
            'rms_amplitude': [],
            'zero_crossings': []
        }
        
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            window_data = data[:, start_idx:end_idx]
            
            # Power spectral density
            freqs = fftfreq(window_size, 1/self.sampling_rate)
            positive_freqs = freqs[:window_size//2]
            
            window_features = {
                'power_spectral_density': [],
                'mean_power': [],
                'peak_frequency': [],
                'spectral_centroid': [],
                'rms_amplitude': [],
                'zero_crossings': []
            }
            
            for ch in range(n_channels):
                # FFT and power spectral density
                fft_vals = fft(window_data[ch])
                psd = np.abs(fft_vals[:window_size//2])**2
                
                # Mean power
                mean_power = np.mean(psd)
                
                # Peak frequency
                peak_freq = positive_freqs[np.argmax(psd)]
                
                # Spectral centroid
                spectral_centroid = np.sum(positive_freqs * psd) / np.sum(psd)
                
                # RMS amplitude
                rms_amplitude = np.sqrt(np.mean(window_data[ch]**2))
                
                # Zero crossings
                zero_crossings = np.sum(np.diff(np.sign(window_data[ch])) != 0)
                
                window_features['power_spectral_density'].append(psd)
                window_features['mean_power'].append(mean_power)
                window_features['peak_frequency'].append(peak_freq)
                window_features['spectral_centroid'].append(spectral_centroid)
                window_features['rms_amplitude'].append(rms_amplitude)
                window_features['zero_crossings'].append(zero_crossings)
            
            for key in features:
                features[key].append(window_features[key])
        
        # Convert to numpy arrays
        for key in features:
            features[key] = np.array(features[key])
        
        features['frequencies'] = positive_freqs
        features['time_windows'] = np.arange(n_windows) * step_size / self.sampling_rate
        
        return features
    
    def detect_artifacts(
        self, 
        data: np.ndarray, 
        threshold_std: float = 5.0,
        method: str = 'amplitude'
    ) -> np.ndarray:
        """
        Detect artifacts in neural signals.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data
        threshold_std : float, default=5.0
            Threshold in standard deviations for artifact detection
        method : str, default='amplitude'
            Artifact detection method ('amplitude', 'gradient')
            
        Returns
        -------
        np.ndarray
            Boolean array indicating artifact locations
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        artifacts = np.zeros((n_channels, n_samples), dtype=bool)
        
        for ch in range(n_channels):
            if method == 'amplitude':
                # Amplitude-based artifact detection
                signal_std = np.std(data[ch])
                signal_mean = np.mean(data[ch])
                threshold = threshold_std * signal_std
                artifacts[ch] = np.abs(data[ch] - signal_mean) > threshold
                
            elif method == 'gradient':
                # Gradient-based artifact detection
                gradient = np.gradient(data[ch])
                gradient_std = np.std(gradient)
                threshold = threshold_std * gradient_std
                artifacts[ch] = np.abs(gradient) > threshold
                
        return artifacts.squeeze() if artifacts.shape[0] == 1 else artifacts
    
    def preprocess_pipeline(
        self, 
        data: np.ndarray,
        bandpass_range: Tuple[float, float] = (1.0, 500.0),
        notch_freq: float = 60.0,
        normalize: bool = True,
        remove_artifacts: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline for neural signals.
        
        Parameters
        ----------
        data : np.ndarray
            Raw neural data
        bandpass_range : tuple, default=(1.0, 500.0)
            Frequency range for bandpass filtering
        notch_freq : float, default=60.0
            Frequency for notch filtering
        normalize : bool, default=True
            Whether to normalize signals
        remove_artifacts : bool, default=True
            Whether to detect and mark artifacts
            
        Returns
        -------
        tuple
            Preprocessed data and processing metadata
        """
        processed_data = data.copy()
        metadata = {'steps': []}
        
        # Bandpass filtering
        if bandpass_range is not None:
            processed_data = self.bandpass_filter(
                processed_data, bandpass_range[0], bandpass_range[1]
            )
            metadata['steps'].append(f"Bandpass filter: {bandpass_range[0]}-{bandpass_range[1]} Hz")
        
        # Notch filtering
        if notch_freq is not None:
            processed_data = self.notch_filter(processed_data, notch_freq)
            metadata['steps'].append(f"Notch filter: {notch_freq} Hz")
        
        # Normalization
        if normalize:
            processed_data, norm_params = self.normalize_signals(processed_data, method='zscore')
            metadata['normalization'] = norm_params
            metadata['steps'].append("Z-score normalization")
        
        # Artifact detection
        if remove_artifacts:
            artifacts = self.detect_artifacts(processed_data)
            metadata['artifacts'] = artifacts
            metadata['steps'].append("Artifact detection")
        
        metadata['sampling_rate'] = self.sampling_rate
        metadata['processed_shape'] = processed_data.shape
        
        return processed_data, metadata


def create_neural_processor(sampling_rate: float = 30000.0) -> NeuralSignalProcessor:
    """
    Factory function to create a neural signal processor.
    
    Parameters
    ----------
    sampling_rate : float, default=30000.0
        Sampling rate in Hz
        
    Returns
    -------
    NeuralSignalProcessor
        Configured signal processor instance
    """
    return NeuralSignalProcessor(sampling_rate=sampling_rate)
