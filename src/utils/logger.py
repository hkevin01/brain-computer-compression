"""
Logging utility for BCI compression toolkit.
Logs performance metrics, parameters, and intermediate results.

References:
- Research reproducibility
- Performance metric logging
"""
import logging
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = os.path.join(LOG_DIR, f"pipeline_log_{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def log_metrics(metrics: dict):
    """
    Logs pipeline metrics.
    """
    logging.info("Metrics: %s", metrics)


def log_parameters(params: dict):
    """
    Logs pipeline parameters.
    """
    logging.info("Parameters: %s", params)
