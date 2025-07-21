import numpy as np
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector
from src.utils import stream_processor

def test_streaming_pipeline():
    connector = PipelineConnector()
    data = np.random.normal(0, 1, (4, 128))
    results = []
    for window, idx in stream_processor.stream_processor(data, buffer_size=32, overlap=8):
        metrics = connector.get_live_metrics(num_channels=window.shape[0], sample_size=window.shape[1])
        results.append(metrics)
    assert len(results) > 0
    assert all('compression_ratio' in m for m in results)
