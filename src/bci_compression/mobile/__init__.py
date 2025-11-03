"""
Mobile-optimized real-time compression library for BCI devices.

This module provides lightweight, power-efficient compression algorithms
specifically designed for mobile and embedded BCI applications.
"""

from .adaptive_quality import AdaptiveQualityController
from .emg_mobile import MobileEMGCompressor
from .mobile_compressor import MobileBCICompressor
from .mobile_metrics import MobileMetrics
from .power_optimizer import PowerOptimizer
from .streaming_pipeline import MobileStreamingPipeline

__all__ = [
    'MobileBCICompressor',
    'MobileEMGCompressor',
    'MobileStreamingPipeline',
    'PowerOptimizer',
    'MobileMetrics',
    'AdaptiveQualityController'
]
