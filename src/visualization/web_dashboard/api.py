"""
FastAPI backend for real-time dashboard metrics.
Provides endpoints for live compression metrics, system health, and alerts.

References:
- https://fastapi.tiangolo.com/
- Real-time BCI dashboard requirements (see project_plan.md)

Usage Example:
    uvicorn api:app --reload
"""
from fastapi import FastAPI, HTTPException
from typing import Dict, List
import random
from src.visualization.web_dashboard.logging_utils import log_event, log_error, log_metrics
from src.visualization.web_dashboard.audit_utils import log_audit
from src.visualization.web_dashboard.user_manager import UserManager
from src.visualization.web_dashboard.permission_checker import PermissionChecker
from src.visualization.web_dashboard.metrics_streamer import MetricsStreamer
from src.visualization.web_dashboard.compression_metrics_aggregator import CompressionMetricsAggregator
from src.visualization.web_dashboard.alert_manager import AlertManager
from src.visualization.web_dashboard.health_monitor import HealthMonitor
from src.visualization.web_dashboard.auth_manager import AuthManager
from src.visualization.web_dashboard.performance_monitor import PerformanceMonitor
import time

app = FastAPI(title="BCI Compression Dashboard API")

user_manager = UserManager()
permission_checker = PermissionChecker("src/visualization/web_dashboard/access_control.yaml")
metrics_streamer = MetricsStreamer()
metrics_aggregator = CompressionMetricsAggregator()
alert_manager = AlertManager()
health_monitor = HealthMonitor()
auth_manager = AuthManager()
performance_monitor = PerformanceMonitor()

@app.get("/performance", response_model=Dict[str, float])
def get_performance_metrics() -> Dict[str, float]:
    """
    Returns real-time backend performance metrics (latency, throughput).
    """
    return performance_monitor.get_performance_metrics()

@app.get("/metrics/live", response_model=Dict[str, float])
def get_live_metrics() -> Dict[str, float]:
    """
    Returns current live metrics for compression pipeline.
    Uses MetricsStreamer and CompressionMetricsAggregator.
    Triggers automated alerts if metrics exceed thresholds.
    Logs request performance.
    """
    start = time.time()
    try:
        metrics = metrics_streamer.get_metrics()
        metrics_aggregator.add_metrics(metrics)
        alert_manager.check_metrics_and_generate_alerts(metrics)
        log_metrics(metrics)
        return metrics
    except Exception as e:
        log_error("Failed to get live metrics", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving live metrics")
    finally:
        end = time.time()
        performance_monitor.log_request(start, end)

@app.get("/metrics/average", response_model=Dict[str, float])
def get_average_metrics() -> Dict[str, float]:
    """
    Returns average metrics over recent history.
    """
    try:
        avg_metrics = metrics_aggregator.get_average_metrics()
        log_metrics(avg_metrics)
        return avg_metrics
    except Exception as e:
        log_error("Failed to get average metrics", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving average metrics")

@app.get("/alerts", response_model=List[Dict[str, str]])
def get_alerts() -> List[Dict[str, str]]:
    """
    Returns current system alerts from AlertManager.
    """
    try:
        alerts = alert_manager.get_alerts()
        log_event("Alerts retrieved", alerts)
        return alerts
    except Exception as e:
        log_error("Failed to get alerts", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alerts")

@app.get("/health", response_model=Dict[str, float])
def get_system_health() -> Dict[str, float]:
    """
    Returns system health metrics from HealthMonitor.
    """
    try:
        health = health_monitor.get_health_metrics()
        log_metrics(health)
        return health
    except Exception as e:
        log_error("Failed to get system health", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving system health")

@app.post("/login")
def login(username: str, password: str) -> Dict[str, str]:
    """
    Authenticates user and returns session ID.
    """
    session_id = auth_manager.login(username, password)
    if session_id:
        log_event("User logged in", username)
        return {"session_id": session_id}
    else:
        log_error("Login failed", username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/logout")
def logout(session_id: str) -> Dict[str, str]:
    """
    Logs out user and invalidates session.
    """
    auth_manager.logout(session_id)
    log_event("User logged out", session_id)
    return {"status": "logged out"}

# Example: Add authentication check to /logs endpoint
@app.get("/logs", response_model=List[Dict[str, str]])
def get_recent_logs(session_id: str) -> List[Dict[str, str]]:
    """
    Returns recent dashboard log events (simulated).
    Requires authentication.
    """
    if not auth_manager.validate_session(session_id):
        log_error("Unauthorized access to logs", session_id)
        raise HTTPException(status_code=401, detail="Unauthorized")
    user = auth_manager.get_user(session_id)
    try:
        role = user_manager.get_role(user)
        if not permission_checker.has_permission(role, "view_logs"):
            log_audit("Unauthorized access to logs", user)
            raise HTTPException(status_code=403, detail="Access denied")
        logs = [
            {"timestamp": "2025-07-21T15:20:00", "level": "INFO", "message": "Metrics updated"},
            {"timestamp": "2025-07-21T15:21:00", "level": "ERROR", "message": "Failed to retrieve metrics"}
        ]
        log_event("Logs retrieved", logs)
        log_audit("Logs accessed", user, logs)
        return logs
    except HTTPException as e:
        log_error("Access denied to logs", str(e))
        raise
    except Exception as e:
        log_error("Failed to get logs", str(e))
        log_audit("Error retrieving logs", user, str(e))
        raise HTTPException(status_code=500, detail="Error retrieving logs")
