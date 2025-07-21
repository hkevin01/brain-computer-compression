"""
Performance monitoring utilities for BCI pipeline.
Tracks latency, throughput, and memory usage.

References:
- Real-time performance requirements
"""
import time
import psutil


def measure_latency(func, *args, **kwargs) -> float:
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return (end - start) * 1000.0


def get_throughput(num_samples: int, elapsed_ms: float) -> float:
    if elapsed_ms == 0:
        return 0.0
    return num_samples / (elapsed_ms / 1000.0)


def get_memory_usage() -> float:
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)
