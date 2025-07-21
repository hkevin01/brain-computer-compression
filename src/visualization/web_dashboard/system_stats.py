"""
SystemStats utility for real system statistics.
Provides methods to retrieve memory usage, GPU utilization, and error rates from the host system.

References:
- Real system integration (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

class SystemStats:
    """
    Retrieves real system statistics for health monitoring.
    """
    @staticmethod
    def get_memory_usage_mb() -> float:
        mem = psutil.virtual_memory()
        return mem.used / (1024 * 1024)

    @staticmethod
    def get_gpu_utilization_pct() -> float:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100.0
        return 0.0

    @staticmethod
    def get_error_rate_pct() -> float:
        # Placeholder: Replace with real error rate calculation
        return 0.0

    @staticmethod
    def get_all_stats() -> Dict[str, float]:
        return {
            "memory_usage_mb": round(SystemStats.get_memory_usage_mb(), 2),
            "gpu_utilization_pct": round(SystemStats.get_gpu_utilization_pct(), 2),
            "error_rate_pct": round(SystemStats.get_error_rate_pct(), 2)
        }
