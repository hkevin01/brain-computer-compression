"""
Metrics module for BCI compression evaluation.

This module provides quality metrics for evaluating compression algorithms,
including specialized metrics for neural data, EMG signals, and other
biomedical signal types.
"""

from .emg_quality import (
    EMGQualityMetrics,
    evaluate_emg_compression_quality,
    quick_emg_quality_check
)

__all__ = [
    'EMGQualityMetrics',
    'evaluate_emg_compression_quality',
    'quick_emg_quality_check'
]
