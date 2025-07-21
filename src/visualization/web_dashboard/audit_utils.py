"""
Audit logging utilities for dashboard and API events.
Logs access, modification, and error events for compliance.

References:
- Security & compliance requirements (see project_plan.md)
- Python logging best practices
"""
import logging
from typing import Any

auditor = logging.getLogger("audit")
auditor.setLevel(logging.INFO)
audit_handler = logging.StreamHandler()
audit_formatter = logging.Formatter('[%(asctime)s] AUDIT: %(message)s')
audit_handler.setFormatter(audit_formatter)
auditor.addHandler(audit_handler)

def log_audit(event: str, user: str = "anonymous", details: Any = None) -> None:
    """Log an audit event with user and optional details."""
    auditor.info(f"User: {user} | Event: {event} | Details: {details}")
