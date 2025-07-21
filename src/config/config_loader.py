"""
Configuration loader for user-defined pipeline settings.
Loads and validates config from file or environment.

References:
- Configurable pipeline parameters
"""
import os
import json
from typing import Dict, Any

CONFIG_PATH = os.getenv("BCI_CONFIG_PATH", "config/pipeline_config.json")
DASHBOARD_CONFIG_PATH = os.getenv("DASHBOARD_CONFIG_PATH", "config/dashboard_config.json")


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        # Fallback to default config
        return {
            "num_channels": 64,
            "sample_size": 1000,
            "compression_type": "lossless",
            "quality_level": 1.0,
            "artifact_severity": 0.5,
            "random_seed": 42,
        }


def load_dashboard_config() -> Dict[str, Any]:
    if os.path.exists(DASHBOARD_CONFIG_PATH):
        with open(DASHBOARD_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "refresh_interval_ms": 1000,
            "alert_thresholds": {
                "latency_ms": 1.5,
                "compression_ratio": 2.0,
                "snr_db": 10.0
            },
            "display_panels": ["metrics", "alerts", "health"]
        }
