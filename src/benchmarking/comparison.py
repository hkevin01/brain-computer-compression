"""
Comparison Framework for BCI Compression Toolkit

Compares new and existing compression algorithms using standardized metrics.

References:
- Neuralink Compression Challenge
- IEEE benchmarking standards
"""

import time
from typing import Any, Dict, List

from src.bci_compression.benchmarking.metrics import BenchmarkMetrics


class CompressionComparison:
    """
    Framework for comparing compression algorithms.
    """

    def __init__(self, methods: List[Any]):
        self.methods = methods

    def run_comparison(self, data: Any) -> Dict[str, Dict[str, float]]:
        """
        Run all methods on data and return metrics.
        """
        results = {}
        for method in self.methods:
            start = time.time()
            compressed = method.compress(data)
            end = time.time()
            decompressed = method.decompress(compressed)
            metrics = {
                'compression_ratio': BenchmarkMetrics.compression_ratio(len(data), len(compressed)),
                'latency_ms': BenchmarkMetrics.processing_latency(start, end),
                'snr_db': BenchmarkMetrics.snr(data, decompressed)
            }
            results[method.__class__.__name__] = metrics
        return results
