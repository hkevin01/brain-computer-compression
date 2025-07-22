"""
AlertManager for managing and generating system alerts.
Provides methods to add, retrieve, and clear alerts.

References:
- Dashboard alert requirements (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import List, Dict
import time
from src.visualization.web_dashboard.alert_config import AlertConfig

class AlertManager:
    """
    Manages dashboard alerts, supports adding, checking, retrieving, and clearing alerts.
    """
    def __init__(self):
        self._alerts = []

    def add_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """
        Adds a new alert to the system.
        """
        self._alerts.append({
            "type": alert_type,
            "message": message,
            "severity": severity
        })

    def check_metrics_and_generate_alerts(self, metrics: Dict[str, float]) -> None:
        """
        Checks metrics against thresholds and generates alerts if exceeded.
        """
        if metrics.get("latency_ms", 0) > 1.5:
            self.add_alert("latency", "Latency exceeds threshold.", "high")
        if metrics.get("compression_ratio", 0) < 2.0:
            self.add_alert("compression", "Compression ratio below optimal.", "medium")
        if metrics.get("snr_db", 0) < 10.0:
            self.add_alert("snr", "SNR is too low.", "high")

    def get_alerts(self) -> List[Dict[str, str]]:
        """
        Returns all current alerts.
        """
        return self._alerts.copy()  # Return a copy of the alerts list

    def clear_alerts(self) -> None:
        """
        Clears all alerts.
        """
        self._alerts.clear()
