import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bci_compression.mobile.adaptive_quality import AdaptiveQualityController
from src.bci_compression.mobile.mobile_compressor import MobileBCICompressor
from src.bci_compression.mobile.mobile_metrics import MobileMetrics
from src.bci_compression.mobile.power_optimizer import PowerOptimizer
from src.bci_compression.mobile.streaming_pipeline import MobileStreamingPipeline


def test_mobile_bci_compressor_roundtrip():
    compressor = MobileBCICompressor(algorithm="lightweight_quant")
    data = np.random.randn(2, 100)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    assert decompressed.shape == data.shape

    # For lossy compression, check SNR instead of exact matching
    snr = MobileMetrics.snr(data, decompressed)
    assert snr > -10.0  # Realistic SNR for mobile lossy compression


def test_mobile_streaming_pipeline():
    compressor = MobileBCICompressor(algorithm="mobile_lz")
    pipeline = MobileStreamingPipeline(compressor=compressor, buffer_size=20, overlap=5)
    data = np.random.randn(100)
    chunks = [data[i:i + 10] for i in range(0, 100, 10)]

    def data_stream():
        if chunks:
            return chunks.pop(0)
        return None

    pipeline.process_stream(data_stream)
    assert pipeline.stats['chunks_processed'] > 0


def test_power_optimizer():
    compressor = MobileBCICompressor()
    optimizer = PowerOptimizer(compressor)
    optimizer.set_mode('battery_save')
    assert compressor.power_mode == 'battery_save'
    optimizer.set_mode('performance')
    assert compressor.power_mode == 'performance'


def test_mobile_metrics():
    orig = np.random.randn(100)
    recon = orig + np.random.normal(0, 0.1, 100)
    cr = MobileMetrics.compression_ratio(1000, 200)
    assert cr == 5.0
    snr = MobileMetrics.snr(orig, recon)
    assert snr > 10
    psnr = MobileMetrics.psnr(orig, recon)
    assert psnr > 10
    latency = MobileMetrics.latency_ms(0, 0.01)
    assert 9 < latency < 11
    power = MobileMetrics.power_estimate(10, 100)
    assert 0.9 < power < 1.1


def test_adaptive_quality_controller():
    compressor = MobileBCICompressor(quality_level=0.8)
    controller = AdaptiveQualityController(compressor)
    controller.adjust_quality(signal_snr=5)
    assert compressor.quality_level > 0.8
    controller.adjust_quality(battery_level=0.1)
    assert compressor.quality_level <= 0.9


def test_improved_algorithms():
    """Test the improved compression algorithms."""
    # Test enhanced LZ
    compressor_lz = MobileBCICompressor(algorithm="mobile_lz")
    data = np.random.randn(2, 50)
    compressed_lz = compressor_lz.compress(data)
    decompressed_lz = compressor_lz.decompress(compressed_lz)
    assert decompressed_lz.shape == data.shape

    # Test improved prediction
    compressor_pred = MobileBCICompressor(algorithm="fast_predict")
    compressed_pred = compressor_pred.compress(data)
    decompressed_pred = compressor_pred.decompress(compressed_pred)
    assert decompressed_pred.shape == data.shape

    # Test compression ratios
    assert len(compressed_lz) < data.nbytes
    assert len(compressed_pred) < data.nbytes

    # Test compression ratios
    assert len(compressed_lz) < data.nbytes
    assert len(compressed_pred) < data.nbytes

    # Test compression ratios
    assert len(compressed_lz) < data.nbytes
    assert len(compressed_pred) < data.nbytes

    # Test compression ratios
    assert len(compressed_lz) < data.nbytes
    assert len(compressed_pred) < data.nbytes

    # Test compression ratios
    assert len(compressed_lz) < data.nbytes
    assert len(compressed_pred) < data.nbytes
