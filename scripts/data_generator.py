#!/usr/bin/env python3
"""
Synthetic neural data generator for testing compression algorithms.

This script generates realistic synthetic neural data with configurable
parameters to simulate various recording conditions and neural activity patterns.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


class SyntheticNeuralGenerator:
    """Generate synthetic neural data with realistic characteristics."""
    
    def __init__(
        self,
        sampling_rate: int = 30000,
        noise_level: float = 0.1,
        spike_rate: float = 10.0,
        burst_probability: float = 0.1
    ):
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.spike_rate = spike_rate  # spikes per second
        self.burst_probability = burst_probability
    
    def generate_neural_signal(
        self, 
        n_channels: int, 
        duration: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic neural signals.
        
        Parameters
        ----------
        n_channels : int
            Number of electrode channels
        duration : float
            Duration in seconds
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Neural data with shape (n_channels, n_samples)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = int(duration * self.sampling_rate)
        data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Generate base signal with 1/f noise
            signal = self._generate_background_noise(n_samples)
            
            # Add spikes
            signal += self._generate_spikes(n_samples)
            
            # Add bursts occasionally
            if np.random.random() < self.burst_probability:
                signal += self._generate_burst(n_samples)
            
            # Add channel-specific characteristics
            signal *= np.random.uniform(0.5, 2.0)  # Amplitude variation
            
            data[ch] = signal
        
        # Add cross-channel correlations
        data = self._add_correlations(data)
        
        return data
    
    def _generate_background_noise(self, n_samples: int) -> np.ndarray:
        """Generate 1/f background noise characteristic of neural signals."""
        # Generate white noise
        white_noise = np.random.randn(n_samples)
        
        # Convert to frequency domain
        fft_noise = np.fft.fft(white_noise)
        
        # Create 1/f spectrum
        freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        freqs[0] = 1  # Avoid division by zero
        pink_spectrum = 1 / np.sqrt(np.abs(freqs))
        
        # Apply 1/f characteristic
        fft_pink = fft_noise * pink_spectrum
        
        # Convert back to time domain
        pink_noise = np.real(np.fft.ifft(fft_pink))
        
        return pink_noise * self.noise_level
    
    def _generate_spikes(self, n_samples: int) -> np.ndarray:
        """Generate spike events."""
        signal = np.zeros(n_samples)
        
        # Calculate number of spikes
        duration = n_samples / self.sampling_rate
        n_spikes = int(self.spike_rate * duration)
        
        # Generate spike times
        spike_times = np.random.randint(0, n_samples, n_spikes)
        
        for spike_time in spike_times:
            # Generate spike waveform
            spike_waveform = self._spike_waveform()
            
            # Add spike to signal
            start_idx = max(0, spike_time - len(spike_waveform) // 2)
            end_idx = min(n_samples, start_idx + len(spike_waveform))
            waveform_start = max(0, len(spike_waveform) // 2 - spike_time)
            waveform_end = waveform_start + (end_idx - start_idx)
            
            signal[start_idx:end_idx] += spike_waveform[waveform_start:waveform_end]
        
        return signal
    
    def _spike_waveform(self) -> np.ndarray:
        """Generate a realistic spike waveform."""
        # Simple biphasic spike
        duration_ms = 2.0  # 2ms spike duration
        n_points = int(duration_ms * self.sampling_rate / 1000)
        
        t = np.linspace(-1, 1, n_points)
        
        # Biphasic waveform
        waveform = -np.exp(-t**2 / 0.1) + 0.3 * np.exp(-(t-0.5)**2 / 0.05)
        
        # Amplitude variation
        amplitude = np.random.uniform(1.0, 5.0)
        
        return waveform * amplitude
    
    def _generate_burst(self, n_samples: int) -> np.ndarray:
        """Generate burst activity."""
        signal = np.zeros(n_samples)
        
        # Random burst location and duration
        burst_start = np.random.randint(0, n_samples // 2)
        burst_duration = np.random.randint(100, 1000)  # 100-1000 samples
        burst_end = min(n_samples, burst_start + burst_duration)
        
        # High-frequency oscillation during burst
        burst_freq = np.random.uniform(80, 200)  # 80-200 Hz
        t = np.arange(burst_end - burst_start) / self.sampling_rate
        burst_signal = np.sin(2 * np.pi * burst_freq * t)
        
        # Amplitude modulation
        envelope = np.exp(-((t - t.mean()) / (t.std() * 2))**2)
        burst_signal *= envelope * np.random.uniform(2.0, 10.0)
        
        signal[burst_start:burst_end] = burst_signal
        
        return signal
    
    def _add_correlations(self, data: np.ndarray) -> np.ndarray:
        """Add realistic cross-channel correlations."""
        n_channels, n_samples = data.shape
        
        # Create correlation matrix (nearby channels more correlated)
        correlation_matrix = np.eye(n_channels)
        for i in range(n_channels):
            for j in range(n_channels):
                distance = abs(i - j)
                correlation_matrix[i, j] = np.exp(-distance / 5.0) * 0.3
        
        # Apply correlations
        correlated_data = correlation_matrix @ data
        
        return correlated_data


def save_data(data: np.ndarray, filepath: str, metadata: dict = None):
    """Save neural data to HDF5 format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('neural_data', data=data)
        
        # Save metadata
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic neural data')
    parser.add_argument('--channels', type=int, default=64,
                       help='Number of channels (default: 64)')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Duration in seconds (default: 60)')
    parser.add_argument('--sampling-rate', type=int, default=30000,
                       help='Sampling rate in Hz (default: 30000)')
    parser.add_argument('--output', type=str, default='data/synthetic/',
                       help='Output directory (default: data/synthetic/)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Background noise level (default: 0.1)')
    parser.add_argument('--spike-rate', type=float, default=10.0,
                       help='Spike rate per second (default: 10.0)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticNeuralGenerator(
        sampling_rate=args.sampling_rate,
        noise_level=args.noise_level,
        spike_rate=args.spike_rate
    )
    
    # Generate data
    print(f"Generating {args.channels} channels, {args.duration}s duration...")
    data = generator.generate_neural_signal(
        n_channels=args.channels,
        duration=args.duration,
        seed=args.seed
    )
    
    # Prepare metadata
    metadata = {
        'channels': args.channels,
        'duration': args.duration,
        'sampling_rate': args.sampling_rate,
        'noise_level': args.noise_level,
        'spike_rate': args.spike_rate,
        'seed': args.seed
    }
    
    # Save data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"synthetic_{args.channels}ch_{args.duration}s_{args.sampling_rate}hz.h5"
    filepath = output_path / filename
    
    save_data(data, str(filepath), metadata)
    
    print(f"Synthetic neural data saved to: {filepath}")
    print(f"Data shape: {data.shape}")
    print(f"Data size: {data.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
