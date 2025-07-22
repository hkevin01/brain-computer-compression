"""
StreamingPipeline orchestrates real-time neural data processing.
Handles buffering, artifact detection, compression, and metrics logging.

References:
- Real-time streaming, artifact detection, compression
"""
import numpy as np
from src.utils import artifact_detector, stream_processor
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector
from src.visualization.web_dashboard.metrics_logger import log_dashboard_metrics

class StreamingPipeline:
    def __init__(self, buffer_size: int = 256, overlap: int = 64):
        self.connector = PipelineConnector()
        self.buffer_size = buffer_size
        self.overlap = overlap

    def process_stream(self, data: np.ndarray):
        for window, idx in stream_processor.stream_processor(data, buffer_size=self.buffer_size, overlap=self.overlap):
            metrics = self.connector.get_live_metrics(num_channels=window.shape[0], sample_size=window.shape[1])
            artifacts = artifact_detector.detect_artifacts(window)
            metrics.update(artifacts)
            log_dashboard_metrics(metrics)
            print(f"Window {idx}: Metrics {metrics}")
