"""
Real-time visualization and monitoring for BCI compression toolkit.

This module provides web-based dashboard, live metrics, alert systems,
and system health monitoring for neural data compression.
"""

from .alert_system import AlertSystem
from .dashboard import BCICompressionDashboard
from .metrics_collector import MetricsCollector
from .system_monitor import SystemMonitor

__all__ = [
    'BCICompressionDashboard',
    'MetricsCollector',
    'AlertSystem',
    'SystemMonitor'
]
