"""
Unit and integration tests for benchmarking framework (Phase 4)
"""
import numpy as np
from src.benchmarking.metrics import BenchmarkMetrics
from src.benchmarking.profiler import PerformanceProfiler


class DummyCompressor:
    def compress(self, data):
        return data[:len(data) // 2]

    def decompress(self, data):
        return np.concatenate([data, data])


def test_compression_ratio():
    orig = np.arange(1000)
    comp = orig[:500]
    ratio = BenchmarkMetrics.compression_ratio(len(orig), len(comp))
    assert ratio == 2.0


def test_processing_latency():
    start, end = 0.0, 0.001
    latency = BenchmarkMetrics.processing_latency(start, end)
    assert latency == 1.0


def test_snr():
    orig = np.ones(1000)
    recon = np.ones(1000)
    snr = BenchmarkMetrics.snr(orig, recon)
    assert snr == float('inf')


def test_profiler_cpu_memory():
    profiler = PerformanceProfiler()
    cpu = profiler.cpu_usage()
    mem = profiler.memory_usage()
    assert isinstance(cpu, float)
    assert isinstance(mem, int)
