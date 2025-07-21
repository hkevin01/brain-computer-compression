"""
DashboardConfig for centralizing dashboard settings.
Stores configuration for API endpoints, refresh intervals, and UI options.

References:
- Modular and maintainable code (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict

class DashboardConfig:
    """
    Central configuration for dashboard backend and frontend.
    """
    API_ENDPOINTS: Dict[str, str] = {
        "metrics_live": "/metrics/live",
        "metrics_average": "/metrics/average",
        "alerts": "/alerts",
        "health": "/health",
        "logs": "/logs",
        "login": "/login",
        "logout": "/logout"
    }
    REFRESH_INTERVALS: Dict[str, int] = {
        "metrics": 1000,
        "average_metrics": 5000,
        "alerts": 5000,
        "health": 5000,
        "logs": 10000
    }
    UI_OPTIONS: Dict[str, str] = {
        "theme": "light",
        "language": "en"
    }
