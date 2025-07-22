"""
Pipeline benchmarking and performance reporting for BCI toolkit.
Benchmarks compression, latency, and throughput.

References:
- Benchmarking methodologies
- Performance reporting
"""
import time
import numpy as np
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector

def benchmark_pipeline(num_channels: int = 64, sample_size: int = 1000, iterations: int = 10):
    connector = PipelineConnector()
    results = []
    for i in range(iterations):
        data = np.random.normal(0, 1, (num_channels, sample_size))
        start = time.time()
        metrics = connector.get_live_metrics(num_channels=num_channels, sample_size=sample_size)
        end = time.time()
        metrics['iteration'] = i
        metrics['elapsed_ms'] = (end - start) * 1000.0
        results.append(metrics)
        print(f"Iteration {i}: {metrics}")
    return results
