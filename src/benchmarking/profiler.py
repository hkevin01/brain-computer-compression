"""
Performance Profiler for BCI Compression Toolkit

Profiles CPU, GPU, and memory usage during compression benchmarking.

References:
- NVIDIA CUDA Toolkit
- Python psutil
"""

import time
import psutil
from typing import Dict


class PerformanceProfiler:
    """
    Profiles system performance during benchmarking.
    """
    def __init__(self):
        self.process = psutil.Process()

    def cpu_usage(self) -> float:
        """Return current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def memory_usage(self) -> int:
        """Return current memory usage in bytes."""
        return self.process.memory_info().rss

    def gpu_usage(self) -> Dict[str, float]:
        """Return GPU usage stats (requires NVIDIA GPU and nvidia-smi)."""
        import subprocess
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            usage = result.stdout.strip().split(',')
            return {'gpu_utilization': float(usage[0]), 'gpu_memory_used': float(usage[1]) * 1024 * 1024}
        except Exception as e:
            return {'gpu_utilization': 0.0, 'gpu_memory_used': 0.0}
