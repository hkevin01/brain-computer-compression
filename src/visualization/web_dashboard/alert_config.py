"""
AlertConfig for alert thresholds and configuration.
Defines thresholds for metrics and alert severity levels.

References:
- Alert logic and automation (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict

class AlertConfig:
    """
    Stores alert thresholds and severity levels for system metrics.
    """
    thresholds: Dict[str, float] = {
        "compression_ratio": 2.0,
        "latency_ms": 2.0,
        "snr_db": 20.0,
        "power_mw": 250.0,
        "memory_usage_mb": 1400.0,
        "gpu_utilization_pct": 85.0,
        "error_rate_pct": 0.5
    }
    severity: Dict[str, str] = {
        "compression_ratio": "warning",
        "latency_ms": "critical",
        "snr_db": "warning",
        "power_mw": "warning",
        "memory_usage_mb": "critical",
        "gpu_utilization_pct": "warning",
        "error_rate_pct": "critical"
    }
