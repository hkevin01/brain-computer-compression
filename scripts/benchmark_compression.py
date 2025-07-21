"""
Benchmarking script for compression algorithms in BCI toolkit.
Runs tests and logs results for reproducibility.

References:
- Benchmarking methodologies
- Synthetic and real neural data
"""
import numpy as np
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector
from src.config.pipeline_config import get_config
from src.utils.logger import log_metrics, log_parameters


def run_benchmark():
    config = get_config()
    np.random.seed(config["random_seed"])
    connector = PipelineConnector()
    _ = np.random.normal(0, 1, (config["num_channels"], config["sample_size"]))
    metrics = connector.get_live_metrics(num_channels=config["num_channels"], sample_size=config["sample_size"])
    log_parameters(config)
    log_metrics(metrics)
    print("Benchmark metrics:", metrics)


if __name__ == "__main__":
    run_benchmark()
