# Dashboard Backend Integration

## Overview
The dashboard backend provides API endpoints for metrics, alerts, health monitoring, and configuration. It integrates with the pipeline and system monitoring modules to deliver real-time data to the frontend.

## Key Modules
- `dashboard_api.py`: Main API endpoints for metrics, alerts, and config.
- `alert_manager.py`: Automated alert generation based on metrics and thresholds.
- `health_monitor.py`: Real system stats (memory, CPU, GPU).
- `metrics_logger.py`: Logging for dashboard metrics and alerts.
- `config_loader.py`: Loads pipeline and dashboard configuration.

## Configuration
Dashboard settings are defined in `config/dashboard_config.json` and loaded via `config_loader.py`.

## Usage Example
```python
from visualization.web_dashboard.dashboard_api import get_metrics, get_alerts, get_config
metrics = get_metrics()
alerts = get_alerts()
config = get_config()
```

## References
- See `project_plan.md` for integration details.
- See `alert_manager.py` for alert logic and severity levels.
