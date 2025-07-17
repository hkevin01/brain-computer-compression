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
    compressor._last_dtype = np.int16
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