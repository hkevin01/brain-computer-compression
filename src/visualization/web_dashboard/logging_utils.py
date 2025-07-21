"""
Logging utilities for dashboard backend.
Provides functions for event, error, and metrics logging.

References:
- Python logging documentation
- BCI dashboard logging requirements (see project_plan.md)
"""
import logging
from typing import Any

logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_event(event: str, details: Any = None) -> None:
    """Log a dashboard event with optional details."""
    logger.info(f"EVENT: {event} | Details: {details}")

def log_error(error: str, details: Any = None) -> None:
    """Log a dashboard error with optional details."""
    logger.error(f"ERROR: {error} | Details: {details}")

def log_metrics(metrics: dict) -> None:
    """Log dashboard metrics."""
    logger.info(f"METRICS: {metrics}")
