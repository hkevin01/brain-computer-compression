"""
Test for PipelineConnector multi-channel metrics and error handling.
"""
import pytest
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector

def test_get_live_metrics_default():
    connector = PipelineConnector()
    metrics = connector.get_live_metrics()
    assert "compression_ratio" in metrics
    assert "snr_db" in metrics
    assert metrics["compression_ratio"] > 0
    assert metrics["snr_db"] > 0

def test_get_live_metrics_multichannel():
    connector = PipelineConnector()
    metrics = connector.get_live_metrics(num_channels=128, sample_size=500)
    assert "compression_ratio" in metrics
    assert "snr_db" in metrics
    assert metrics["compression_ratio"] > 0
    assert metrics["snr_db"] > 0

def test_get_live_metrics_error_handling():
    connector = PipelineConnector()
    # Simulate error by passing invalid arguments
    metrics = connector.get_live_metrics(num_channels=-1, sample_size=-1)
    assert "error" in metrics
