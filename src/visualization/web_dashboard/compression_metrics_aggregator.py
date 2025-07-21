"""
CompressionMetricsAggregator for pipeline metrics aggregation.
Aggregates and computes metrics from compression pipeline modules.

References:
- Compression pipeline integration (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import List, Dict
import numpy as np

class CompressionMetricsAggregator:
    """
    Aggregates metrics from compression pipeline modules.
    """
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []

    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Adds a new set of metrics to the history.
        """
        self.metrics_history.append(metrics)

    def get_average_metrics(self) -> Dict[str, float]:
        """
        Returns average metrics over the history.
        """
        if not self.metrics_history:
            return {}
        keys = self.metrics_history[0].keys()
        avg = {k: float(np.mean([m[k] for m in self.metrics_history])) for k in keys}
        return avg

    def reset(self) -> None:
        """
        Clears the metrics history.
        """
        self.metrics_history.clear()
