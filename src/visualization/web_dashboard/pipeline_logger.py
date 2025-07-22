"""
PipelineLogger: Logging hooks for dashboard and pipeline events.
References:
- Pipeline audit and dashboard metrics logging
"""
from typing import Dict, Any, Optional
import logging

class PipelineLogger:
    """
    Logs pipeline events for dashboard and audit trail.
    """
    def __init__(self, log_file: str = "pipeline_audit.log"):
        self.logger = logging.getLogger("PipelineLogger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs a pipeline event.
        Args:
            event: Event description
            details: Optional event details
        """
        msg = f"EVENT: {event}"
        if details:
            msg += f" | DETAILS: {details}"
        self.logger.info(msg)
