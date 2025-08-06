"""
BCI Compression Benchmarking Module

This module provides benchmarking utilities for various BCI compression algorithms,
including specialized benchmarks for neural data, EMG data, and mobile/wearable devices.
"""

from .emg_benchmark import EMGBenchmarkSuite, create_synthetic_emg_datasets, run_emg_benchmark_example

__all__ = [
    'EMGBenchmarkSuite',
    'create_synthetic_emg_datasets',
    'run_emg_benchmark_example'
]
