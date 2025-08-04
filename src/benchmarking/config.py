"""Configuration for automated benchmarking."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import time

import numpy as np


class BenchmarkDataset(Enum):
    """Available benchmark datasets."""
    OPENBCI_MOTOR = "openbci_motor"
    NEURALINK_MONKEY = "neuralink_monkey"
    KERNEL_VISUAL = "kernel_visual"
    SYNTHETIC = "synthetic"


@dataclass
class CompressionBenchmark:
    """Configuration for compression benchmarks."""
    name: str
    dataset: BenchmarkDataset
    channels: int
    sample_rate: int
    duration_sec: float
    noise_level: float = 0.1
    artifact_prob: float = 0.05
    min_compression: float = 5.0
    max_latency_ms: float = 2.0
    min_snr_db: float = 25.0


@dataclass
class HardwareBenchmark:
    """Configuration for hardware benchmarks."""
    name: str
    device: str  # "cpu", "cuda", "fpga"
    batch_size: int
    max_memory_mb: float
    power_target_w: Optional[float] = None
    thermal_limit_c: Optional[float] = None


STANDARD_BENCHMARKS = {
    "realtime_eeg": CompressionBenchmark(
        name="realtime_eeg",
        dataset=BenchmarkDataset.OPENBCI_MOTOR,
        channels=64,
        sample_rate=250,
        duration_sec=300,
        min_compression=8.0,
        max_latency_ms=2.0,
        min_snr_db=25.0
    ),
    "high_density": CompressionBenchmark(
        name="high_density",
        dataset=BenchmarkDataset.NEURALINK_MONKEY,
        channels=1024,
        sample_rate=19500,
        duration_sec=60,
        min_compression=15.0,
        max_latency_ms=1.0,
        min_snr_db=30.0
    ),
    "long_term": CompressionBenchmark(
        name="long_term",
        dataset=BenchmarkDataset.KERNEL_VISUAL,
        channels=256,
        sample_rate=1000,
        duration_sec=3600,
        min_compression=20.0,
        max_latency_ms=5.0,
        min_snr_db=20.0
    )
}

HARDWARE_BENCHMARKS = {
    "gpu_realtime": HardwareBenchmark(
        name="gpu_realtime",
        device="cuda",
        batch_size=32,
        max_memory_mb=4000,
        power_target_w=150
    ),
    "fpga_lowlatency": HardwareBenchmark(
        name="fpga_lowlatency",
        device="fpga",
        batch_size=1,
        max_memory_mb=512,
        power_target_w=10
    ),
    "cpu_fallback": HardwareBenchmark(
        name="cpu_fallback",
        device="cpu",
        batch_size=8,
        max_memory_mb=2000
    )
}


def generate_synthetic_data(config: CompressionBenchmark) -> np.ndarray:
    """Generate synthetic neural data for benchmarking."""
    # Calculate total samples
    n_samples = int(config.sample_rate * config.duration_sec)
    
    # Generate base signal
    signal = np.random.normal(0, 1, (config.channels, n_samples))
    
    # Add structured temporal patterns
    t = np.linspace(0, config.duration_sec, n_samples)
    for i in range(config.channels):
        # Add oscillations
        freq = np.random.uniform(5, 50)  # 5-50 Hz
        signal[i] += 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add sparse spikes
        if np.random.random() < config.artifact_prob:
            spike_times = np.random.choice(n_samples, size=int(0.001 * n_samples))
            signal[i, spike_times] += np.random.uniform(5, 10, len(spike_times))
    
    # Add noise
    signal += np.random.normal(0, config.noise_level, signal.shape)
    
    return signal


def run_benchmark(
    compressor: Any,
    config: CompressionBenchmark,
    hardware: Optional[HardwareBenchmark] = None
) -> Dict:
    """Run benchmark with given configuration."""
    # Generate or load data
    if config.dataset == BenchmarkDataset.SYNTHETIC:
        data = generate_synthetic_data(config)
    else:
        # Load real dataset
        data = load_dataset(config.dataset, config.channels, config.duration_sec)
    
    # Set up hardware if specified
    if hardware:
        setup_hardware(hardware)
    
    # Warm up
    warmup_data = data[:, :1000]
    _ = compressor.compress(warmup_data)
    
    # Run benchmark
    metrics = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < 60:  # 1 minute test
            # Get data chunk
            chunk = get_next_chunk(data)
            
            # Compress
            compressed = compressor.compress(chunk)
            decompressed = compressor.decompress(compressed)
            
            # Calculate metrics
            compression_ratio = len(chunk.tobytes()) / len(compressed)
            snr = calculate_snr(chunk, decompressed)
            latency = get_processing_time()
            
            metrics.append({
                'compression_ratio': compression_ratio,
                'snr_db': snr,
                'latency_ms': latency
            })
    
    finally:
        if hardware:
            cleanup_hardware(hardware)
    
    # Aggregate results
    results = {
        'compression_ratio': np.mean([m['compression_ratio'] for m in metrics]),
        'snr_db': np.mean([m['snr_db'] for m in metrics]),
        'latency_ms': {
            'mean': np.mean([m['latency_ms'] for m in metrics]),
            'p95': np.percentile([m['latency_ms'] for m in metrics], 95),
            'max': max(m['latency_ms'] for m in metrics)
        },
        'passed': all([
            np.mean([m['compression_ratio'] for m in metrics]) >= config.min_compression,
            np.mean([m['snr_db'] for m in metrics]) >= config.min_snr_db,
            np.percentile([m['latency_ms'] for m in metrics], 95) <= config.max_latency_ms
        ])
    }
    
    if hardware:
        results['hardware_metrics'] = get_hardware_metrics(hardware)
    
    return results
