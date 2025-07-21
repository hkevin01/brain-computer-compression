"""
Dashboard backend API stub for metrics and alerts.
Provides endpoints for metrics, alerts, and configuration.

References:
- Dashboard integration
- Backend/frontend communication
"""
from typing import Dict, Any
from .alert_manager import check_metrics_and_generate_alerts
from .health_monitor import get_health_metrics
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector
from src.config.config_loader import load_config, load_dashboard_config

# Stub functions for backend API

def get_metrics() -> Dict[str, Any]:
    config = load_config()
    connector = PipelineConnector()
    metrics = connector.get_live_metrics(num_channels=config["num_channels"], sample_size=config["sample_size"])
    metrics.update(get_health_metrics())
    return metrics


def get_alerts() -> Dict[str, Any]:
    metrics = get_metrics()
    dashboard_cfg = load_dashboard_config()
    alerts = check_metrics_and_generate_alerts(metrics)
    # Optionally filter or format alerts based on dashboard config
    return {"alerts": alerts}


def get_config() -> Dict[str, Any]:
    return load_dashboard_config()
