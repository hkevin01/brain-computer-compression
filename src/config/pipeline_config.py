"""
Pipeline and compression configuration for BCI toolkit.
Supports reproducible experiments and parameter logging.

References:
- Research reproducibility
- Configurable compression parameters
"""
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "num_channels": 64,
    "sample_size": 1000,
    "compression_type": "lossless",
    "quality_level": 1.0,
    "artifact_severity": 0.5,
    "random_seed": 42,
}


def get_config() -> Dict[str, Any]:
    """
    Returns default pipeline configuration.
    """
    return DEFAULT_CONFIG
