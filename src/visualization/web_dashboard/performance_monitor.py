"""
PerformanceMonitor for backend performance profiling and logging.
Tracks request latency, throughput, and resource usage.

References:
- Performance & monitoring (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
import time
from typing import Dict, List

class PerformanceMonitor:
    """
    Monitors backend performance metrics for dashboard.
    """
    def __init__(self):
        self.latency_history: List[float] = []
        self.throughput_history: List[float] = []
        self.last_request_time: float = time.time()

    def log_request(self, start: float, end: float) -> None:
        latency = end - start
        self.latency_history.append(latency)
        now = time.time()
        throughput = 1.0 / (now - self.last_request_time) if self.last_request_time != now else 0.0
        self.throughput_history.append(throughput)
        self.last_request_time = now

    def get_performance_metrics(self) -> Dict[str, float]:
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0.0
        return {
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "avg_throughput_rps": round(avg_throughput, 2)
        }
