"""
Unit tests for AdaptiveCompressionPipeline (edge-cloud adaptation).
"""
import numpy as np
from src.compression.adaptive_pipeline import AdaptiveCompressionPipeline

def test_network_analysis():
    pipeline = AdaptiveCompressionPipeline({'mode': 'auto'})
    assert pipeline.analyze_network({'latency': 5, 'bandwidth': 200}) == 'cloud'
    assert pipeline.analyze_network({'latency': 100, 'bandwidth': 5}) == 'edge'
    assert pipeline.analyze_network({'latency': 20, 'bandwidth': 50}) == 'hybrid'

def test_compression_modes():
    data = np.random.randn(32, 1000)
    pipeline = AdaptiveCompressionPipeline({'mode': 'auto'})
    pipeline.mode = 'edge'
    compressed = pipeline.compress(data)
    decompressed = pipeline.decompress(compressed)
    assert decompressed.shape == data.shape
    pipeline.mode = 'cloud'
    compressed = pipeline.compress(data)
    decompressed = pipeline.decompress(compressed)
    assert decompressed.shape == data.shape
    pipeline.mode = 'hybrid'
    compressed = pipeline.compress(data)
    decompressed = pipeline.decompress(compressed)
    assert decompressed.shape == data.shape
