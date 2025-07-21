"""
Metrics logging utility for dashboard and reproducibility.
Logs pipeline metrics, alerts, and performance data.

References:
- Dashboard integration
- Research reproducibility
"""
import logging
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = os.path.join(LOG_DIR, f"dashboard_metrics_{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_dashboard_metrics(metrics: dict):
    logging.info("Dashboard Metrics: %s", metrics)


def log_alert(alert: dict):
    logging.info("Alert: %s", alert)
