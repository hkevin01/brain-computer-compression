"""
HealthMonitor for real-time system health metrics.
Provides methods to retrieve memory usage, GPU utilization, and error rates.

References:
- System health monitoring requirements (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict
import psutil
from src.visualization.web_dashboard.system_stats import SystemStats

class HealthMonitor:
    """
    Provides real-time system health metrics for dashboard.
    Now uses SystemStats and psutil for real system statistics.
    """
    def get_health_metrics(self) -> Dict[str, float]:
        """
        Returns real system health metrics.
        """
        mem = psutil.virtual_memory()
        memory_usage_mb = mem.used / (1024 * 1024)
        gpu_utilization_pct = SystemStats.get_gpu_utilization_pct()
        error_rate_pct = SystemStats.get_error_rate_pct()  # Replace with real error rate logic
        return {
            "memory_usage_mb": round(memory_usage_mb, 2),
            "gpu_utilization_pct": round(gpu_utilization_pct, 2),
            "error_rate_pct": round(error_rate_pct, 2)
        }
