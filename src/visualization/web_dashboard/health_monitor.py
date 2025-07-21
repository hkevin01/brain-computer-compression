"""
Dashboard health monitor for real system stats.
Provides memory, CPU, and GPU usage for dashboard display.

References:
- Real system health monitoring
"""
from typing import Dict
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def get_health_metrics() -> Dict[str, float]:
    """
    Returns real system health metrics.
    """
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    gpu = None
    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0].memoryUtil * 100
    return {
        "memory_percent": mem.percent,
        "cpu_percent": cpu,
        "gpu_percent": gpu if gpu is not None else "N/A"
    }
