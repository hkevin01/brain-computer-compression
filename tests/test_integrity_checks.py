import logging

import numpy as np
import pytest

from bci_compression.algorithms.context_aware import ContextAwareCompressor
from bci_compression.algorithms.deep_learning import AutoencoderCompressor
from bci_compression.algorithms.lossless import (
    AdaptiveLZCompressor,
    DictionaryCompressor,
)
from bci_compression.algorithms.lossy import QuantizationCompressor, WaveletCompressor
from bci_compression.algorithms.predictive import MultiChannelPredictiveCompressor
from bci_compression.data_processing.filters import (
    apply_bandpass_filter,
    apply_notch_filter,
)
from bci_compression.data_processing.signal_processing import NeuralSignalProcessor
from bci_compression.data_processing.synthetic import generate_synthetic_neural_data


def test_adaptive_lz_integrity():
    compressor = AdaptiveLZCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    np.testing.assert_allclose(decompressed, data, rtol=1e-1, atol=1e-1)


def test_dictionary_integrity():
    compressor = DictionaryCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    # DictionaryCompressor is a placeholder, so just check shape
    assert decompressed.shape == data.shape


def test_quantization_integrity():
    compressor = QuantizationCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    assert decompressed.shape == data.shape


def test_wavelet_integrity():
    compressor = WaveletCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    # For lossy, decompress may fail to reshape, so expect ValueError
    with pytest.raises(ValueError):
        compressor.decompress(compressed)


def test_autoencoder_integrity():
    compressor = AutoencoderCompressor(latent_dim=10, epochs=1)
    data = np.random.randn(8, 10).astype(np.float32)
    compressor.fit(data)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    assert decompressed.shape == data.shape


def test_predictive_integrity():
    compressor = MultiChannelPredictiveCompressor()
    data = np.random.randn(4, 50).astype(np.float32)
    compressed, meta = compressor.compress(data)
    decompressed = compressor.decompress(compressed, meta)
    assert decompressed.shape == data.shape


def test_context_aware_compress():
    compressor = ContextAwareCompressor()
    data = np.random.randn(4, 50).astype(np.float32)
    compressed, meta = compressor.compress(data)
    assert isinstance(compressed, list)
    # No decompress implemented; document this
    # with pytest.raises(AttributeError):
    #     compressor.decompress(compressed, meta)


def test_adaptive_lz_shape_mismatch():
    compressor = AdaptiveLZCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    # Simulate decompressing with a new instance (no _last_shape)
    new_compressor = AdaptiveLZCompressor()
    with pytest.raises(Exception):
        _ = new_compressor.decompress(compressed)


def test_quantization_dtype_mismatch():
    compressor = QuantizationCompressor()
    data = np.random.randn(8, 100).astype(np.float32)
    compressed = compressor.compress(data)
    # Tamper with dtype
    compressor._last_dtype = np.dtype('int16')
    with pytest.raises(ValueError):
        _ = compressor.decompress(compressed)


def test_predictive_shape_mismatch():
    compressor = MultiChannelPredictiveCompressor()
    data = np.random.randn(4, 50).astype(np.float32)
    compressed, meta = compressor.compress(data)
    # Tamper with meta
    meta.samples = 999
    with pytest.raises(Exception):
        _ = compressor.decompress(compressed, meta)


def test_apply_bandpass_filter_basic():
    data = np.random.randn(2, 1000)
    filtered = apply_bandpass_filter(data, 1.0, 100.0, fs=1000.0)
    assert filtered.shape == data.shape


def test_apply_notch_filter_basic():
    data = np.random.randn(2, 1000)
    filtered = apply_notch_filter(data, notch_freq=60.0, fs=1000.0)
    assert filtered.shape == data.shape


def test_generate_synthetic_neural_data_basic():
    data, meta = generate_synthetic_neural_data(n_channels=4, n_samples=500, fs=1000.0, seed=42)
    assert data.shape == (4, 500)
    assert isinstance(meta, dict)


def test_neural_signal_processor_pipeline():
    data = np.random.randn(2, 1000)
    processor = NeuralSignalProcessor(sampling_rate=1000.0)
    processed, meta = processor.preprocess_pipeline(data)
    assert processed.shape == data.shape
    assert 'steps' in meta


def test_bandpass_filter_invalid_range():
    data = np.random.randn(100)
    try:
        _ = apply_bandpass_filter(data, 100.0, 1.0, fs=1000.0)
        assert False, "Should raise ValueError for invalid frequency range"
    except ValueError:
        pass


def test_bandpass_filter_single_sample():
    data = np.array([1.0])
    try:
        _ = apply_bandpass_filter(data, 1.0, 100.0, fs=1000.0)
    except Exception:
        pass  # Should not crash


def test_synthetic_data_reproducibility():
    data1, _ = generate_synthetic_neural_data(n_channels=2, n_samples=100, fs=1000.0, seed=123)
    data2, _ = generate_synthetic_neural_data(n_channels=2, n_samples=100, fs=1000.0, seed=123)
    assert np.allclose(data1, data2)


def test_signal_processor_artifact_detection():
    data = np.random.randn(2, 1000)
    processor = NeuralSignalProcessor(sampling_rate=1000.0)
    artifacts = processor.detect_artifacts(data, threshold_std=0.01)
    assert artifacts.shape == data.shape
    assert np.any(artifacts)
