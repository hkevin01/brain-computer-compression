"""
Advanced pipeline configuration manager for BCI toolkit.
Supports profiles, presets, and runtime config updates.

References:
- Configurable pipeline parameters
- Runtime updates
"""
from typing import Dict, Any
import json
import os

CONFIG_DIR = "config"

class PipelineConfigManager:
    def __init__(self, config_name: str = "default.json"):
        self.config_path = os.path.join(CONFIG_DIR, config_name)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "num_channels": 64,
                "sample_size": 1000,
                "compression_type": "lossless",
                "quality_level": 1.0,
                "artifact_severity": 0.5,
                "random_seed": 42,
            }

    def update_config(self, updates: Dict[str, Any]) -> None:
        self.config.update(updates)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def get_config(self) -> Dict[str, Any]:
        return self.config
