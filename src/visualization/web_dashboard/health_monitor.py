"""
HealthMonitor for real-time system health metrics.
Provides methods to retrieve memory usage, GPU utilization, and error rates.

References:
- System health monitoring requirements (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict
from src.visualization.web_dashboard.system_stats import SystemStats

class HealthMonitor:
    """
    Provides real-time system health metrics for dashboard.
    Now uses SystemStats for real system statistics.
    """
    def get_health_metrics(self) -> Dict[str, float]:
        """
        Returns real system health metrics.
        """
        return SystemStats.get_all_stats()
