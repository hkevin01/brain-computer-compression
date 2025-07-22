"""
Pipeline logging and audit trail utility for BCI toolkit.
Logs pipeline events, errors, and configuration changes.

References:
- Research reproducibility
- Debugging and audit trails
"""
import logging
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = os.path.join(LOG_DIR, f"pipeline_audit_{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_event(event: str):
    logging.info("Event: %s", event)


def log_error(error: str):
    logging.error("Error: %s", error)


def log_config_change(config: dict):
    logging.info("Config Change: %s", config)
