"""
Basic tests for dashboard API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.visualization.web_dashboard.api import app

client = TestClient(app)

def test_get_live_metrics():
    response = client.get("/metrics/live")
    assert response.status_code == 200
    data = response.json()
    assert "compression_ratio" in data
    assert "latency_ms" in data
    assert "snr_db" in data
    assert "power_mw" in data

def test_get_alerts():
    response = client.get("/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    assert all("type" in alert and "message" in alert for alert in alerts)

def test_get_system_health():
    response = client.get("/health")
    assert response.status_code == 200
    health = response.json()
    assert "memory_usage_mb" in health
    assert "gpu_utilization_pct" in health
    assert "error_rate_pct" in health

def test_get_recent_logs():
    response = client.get("/logs")
    assert response.status_code == 200
    logs = response.json()
    assert isinstance(logs, list)
    assert all("timestamp" in log and "level" in log and "message" in log for log in logs)
