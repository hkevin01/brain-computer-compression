"""
Test for PipelineConnector multi-channel metrics and error handling.
"""
import numpy as np
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

def test_get_live_metrics_basic():
    connector = PipelineConnector()
    metrics = connector.get_live_metrics(num_channels=8, sample_size=100)
    assert "compression_ratio" in metrics
    assert "snr_db" in metrics
    assert "artifacts" in metrics
    assert metrics["compression_ratio"] > 0
    assert isinstance(metrics["fft_freqs"], np.ndarray) or isinstance(metrics["fft_freqs"], list)

def test_get_live_metrics_reproducibility():
    connector = PipelineConnector()
    m1 = connector.get_live_metrics(num_channels=4, sample_size=50, seed=42)
    m2 = connector.get_live_metrics(num_channels=4, sample_size=50, seed=42)
    assert np.allclose(m1["snr_db"], m2["snr_db"])

def test_simulate_compression():
    connector = PipelineConnector()
    ratio = connector.simulate_compression(1000, 500)
    assert ratio == 2.0
    assert connector.simulate_compression(1000, 0) == 0.0

def test_inject_artifacts_spike():
    connector = PipelineConnector()
    signal = np.zeros((2, 10))
    noisy = connector.inject_artifacts(signal, artifact_type="spike", severity=1.0)
    assert noisy.shape == signal.shape
    assert np.any(noisy != signal)

def test_inject_artifacts_noise():
    connector = PipelineConnector()
    signal = np.zeros((2, 10))
    noisy = connector.inject_artifacts(signal, artifact_type="noise", severity=1.0)
    assert noisy.shape == signal.shape
    assert np.any(noisy != signal)

def test_inject_artifacts_drift():
    connector = PipelineConnector()
    signal = np.zeros((2, 10))
    drifted = connector.inject_artifacts(signal, artifact_type="drift", severity=1.0)
    assert drifted.shape == signal.shape
    assert np.any(drifted != signal)

def test_simulate_multimodal_fusion():
    connector = PipelineConnector()
    eeg = np.ones((2, 10))
    fmri = np.ones((2, 10)) * 2
    fused = connector.simulate_multimodal_fusion(eeg, fmri)
    assert fused.shape == eeg.shape
    assert np.allclose(fused, 0.7 * eeg + 0.3 * fmri)

def test_error_handling():
    connector = PipelineConnector()
    # Simulate error by passing invalid input
    metrics = connector.get_live_metrics(num_channels=-1, sample_size=-1)
    assert "error" in metrics
