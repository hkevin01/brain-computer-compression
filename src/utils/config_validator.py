"""
Configuration validation and error reporting for BCI pipeline.
Validates config dicts and reports issues.

References:
- Robust error handling
- Config validation
"""
from typing import Dict, Any

def validate_config(config: Dict[str, Any]) -> bool:
    required_keys = ["num_channels", "sample_size", "compression_type", "quality_level", "artifact_severity", "random_seed"]
    for key in required_keys:
        if key not in config:
            print(f"Missing config key: {key}")
            return False
    return True
