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
    Manages system alerts for dashboard visualization.
    Now supports automated alert generation based on metric thresholds and severity levels.
    """
    def __init__(self):
        self.alerts: List[Dict[str, str]] = []

    def add_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """
        Adds a new alert to the system.
        """
        self.alerts.append({
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        })

    def get_alerts(self) -> List[Dict[str, str]]:
        """
        Returns all current alerts.
        """
        return self.alerts[-10:]  # Return last 10 alerts

    def clear_alerts(self) -> None:
        """
        Clears all alerts.
        """
        self.alerts.clear()

    def check_metrics_and_generate_alerts(self, metrics: Dict[str, float]) -> None:
        """
        Checks metrics against thresholds and generates alerts if exceeded.
        """
        for key, value in metrics.items():
            threshold = AlertConfig.thresholds.get(key)
            severity = AlertConfig.severity.get(key, "info")
            if threshold is not None and value > threshold:
                self.add_alert(key, f"{key} exceeded threshold: {value} > {threshold}", severity)
