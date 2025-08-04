"""Utility functions for benchmarking."""
import time
from typing import Dict, Optional

import numpy as np

from src.benchmarking.config import BenchmarkDataset, HardwareBenchmark


def load_dataset(dataset: BenchmarkDataset, channels: int, duration_sec: float) -> np.ndarray:
    """Load a real dataset for benchmarking."""
    # Implementation of real dataset loading will be added in future updates
    # Currently using synthetic data with dataset-specific characteristics
    n_samples = int(19500 * duration_sec)  # Using max sample rate

    if dataset == BenchmarkDataset.OPENBCI_MOTOR:
        # Simulate motor-related neural activity
        data = np.random.normal(0, 0.5, (channels, n_samples))
        # Add motor-related oscillations (beta band 13-30 Hz)
        t = np.linspace(0, duration_sec, n_samples)
        for ch in range(channels):
            freq = np.random.uniform(13, 30)
            data[ch] += 0.3 * np.sin(2 * np.pi * freq * t)

    elif dataset == BenchmarkDataset.NEURALINK_MONKEY:
        # Simulate high-frequency spike data
        data = np.random.normal(0, 0.3, (channels, n_samples))
        # Add sparse spikes
        spike_prob = 0.001
        spikes = np.random.binomial(1, spike_prob, (channels, n_samples))
        data += spikes * np.random.uniform(1, 2, (channels, n_samples))

    elif dataset == BenchmarkDataset.KERNEL_VISUAL:
        # Simulate visual evoked potentials
        data = np.random.normal(0, 0.4, (channels, n_samples))
        # Add periodic visual responses
        t = np.linspace(0, duration_sec, n_samples)
        for ch in range(channels):
            freq = np.random.uniform(8, 12)  # Alpha band
            data[ch] += 0.25 * np.sin(2 * np.pi * freq * t)

    else:  # SYNTHETIC
        data = np.random.normal(0, 1, (channels, n_samples))

    return data


def setup_hardware(config: HardwareBenchmark) -> None:
    """Configure hardware for benchmarking."""
    if config.device == "cuda":
        try:
            import torch.cuda
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        except ImportError as exc:
            raise RuntimeError("CUDA benchmarking requires PyTorch") from exc
    elif config.device == "fpga":
        # FPGA setup will be implemented when hardware is available
        raise NotImplementedError("FPGA support is planned for future releases")


def cleanup_hardware(config: HardwareBenchmark) -> None:
    """Clean up hardware resources after benchmarking."""
    if config.device == "cuda":
        try:
            import torch.cuda
            torch.cuda.empty_cache()
        except ImportError:
            pass  # PyTorch not available, nothing to clean up


def get_next_chunk(data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    """Get next chunk of data for streaming simulation."""
    start = int(time.time() * 1000) % (data.shape[1] - chunk_size)
    return data[:, start:start + chunk_size]


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate signal-to-noise ratio in decibels."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    signal_power = np.mean(original ** 2)
    return 10 * np.log10(signal_power / mse)


def get_processing_time(start_time: Optional[float] = None) -> float:
    """Get processing time in milliseconds."""
    if start_time is None:
        return time.time()
    return (time.time() - start_time) * 1000


def get_hardware_metrics(config: HardwareBenchmark) -> Dict:
    """Get hardware-specific performance metrics."""
    metrics = {
        'device': config.device,
        'timestamp': time.time()
    }

    if config.device == "cuda":
        try:
            import torch.cuda
            metrics.update({
                'memory_used_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'memory_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            })

            if hasattr(torch.cuda, 'get_device_properties'):
                device = torch.cuda.get_device_properties(0)
                metrics['gpu_utilization'] = device.utilization_rate
        except ImportError:
            metrics.update({
                'memory_used_mb': 0,
                'memory_cached_mb': 0,
                'error': 'PyTorch CUDA not available'
            })
    elif config.device == "fpga":
        metrics.update({
            'status': 'Not implemented',
            'message': 'FPGA metrics collection planned for future release'
        })

    return metrics
