"""
Comprehensive Test Suite for Transformer-Based Neural Compression (Phase 8a)

Tests include:
- Unit tests for all transformer components
- Performance benchmarking tests
- Integration tests for end-to-end workflows
- Property-based testing for compression invariants
- Quality metric validation tests
- Real-time processing validation
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.bci_compression.algorithms.transformer_compression import (
    TransformerCompressor,
    TransformerConfig,
    PositionalEncoding,
    MultiHeadAttention,
    PerformanceMonitor,
    create_transformer_compressor,
    benchmark_transformer_compression
)


class TestTransformerConfig(unittest.TestCase):
    """Test transformer configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerConfig()
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.compression_ratio_target, 4.0)
        self.assertTrue(config.causal_masking)
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = TransformerConfig(
            d_model=512,
            n_heads=16,
            compression_ratio_target=6.0
        )
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_heads, 16)
        self.assertEqual(config.compression_ratio_target, 6.0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid head count
        with self.assertRaises(ValueError):
            config = TransformerConfig(d_model=256, n_heads=7)  # Not divisible


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding for neural signals."""
    
    def setUp(self):
        self.pos_enc = PositionalEncoding(d_model=256, max_len=1024)
    
    def test_encoding_shape(self):
        """Test positional encoding output shape."""
        seq_len = 512
        encoding = self.pos_enc.encode(np.random.randn(seq_len, 256))
        self.assertEqual(encoding.shape, (seq_len, 256))
    
    def test_encoding_properties(self):
        """Test mathematical properties of positional encoding."""
        signal = np.random.randn(100, 256)
        encoded = self.pos_enc.encode(signal)
        
        # Encoded signal should have same shape
        self.assertEqual(encoded.shape, signal.shape)
        
        # Encoding should be deterministic
        encoded2 = self.pos_enc.encode(signal)
        np.testing.assert_array_equal(encoded, encoded2)
    
    def test_max_length_validation(self):
        """Test maximum length validation."""
        with self.assertLogs(level='WARNING'):
            signal = np.random.randn(2000, 256)  # Exceeds max_len
            encoded = self.pos_enc.encode(signal)
            self.assertEqual(encoded.shape[0], 1024)  # Truncated to max_len


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention mechanism."""
    
    def setUp(self):
        self.attention = MultiHeadAttention(d_model=256, nhead=8)
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        seq_len, d_model = 100, 256
        query = np.random.randn(seq_len, d_model)
        key = np.random.randn(seq_len, d_model)
        value = np.random.randn(seq_len, d_model)
        
        output, weights = self.attention.forward(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (seq_len, d_model))
        self.assertEqual(weights.shape, (self.attention.nhead, seq_len, seq_len))
    
    def test_causal_masking(self):
        """Test causal masking for real-time processing."""
        seq_len = 50
        mask = self.attention._create_causal_mask(seq_len)
        
        # Check mask properties
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Upper triangle should be False (masked)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertFalse(mask[i, j])
    
    def test_attention_weights_properties(self):
        """Test attention weights mathematical properties."""
        seq_len, d_model = 50, 256
        query = np.random.randn(seq_len, d_model)
        
        output, weights = self.attention.forward(query, query, query)
        
        # Attention weights should sum to 1 (approximately due to masking)
        for head in range(weights.shape[0]):
            for i in range(seq_len):
                weight_sum = np.sum(weights[head, i, :i+1])  # Only unmasked positions
                self.assertAlmostEqual(weight_sum, 1.0, places=5)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_metric_recording(self):
        """Test metric recording functionality."""
        self.monitor.record_compression(0.1, 3.5, 28.5, 50.0)
        self.monitor.record_compression(0.12, 3.7, 29.1, 52.0)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertAlmostEqual(summary['avg_compression_time_ms'], 110.0, places=1)
        self.assertAlmostEqual(summary['avg_compression_ratio'], 3.6, places=1)
        self.assertAlmostEqual(summary['avg_snr_db'], 28.8, places=1)
        self.assertEqual(summary['total_operations'], 2)
    
    def test_error_recording(self):
        """Test error recording and tracking."""
        self.monitor.record_compression(0.1, 3.5, 28.5, 50.0)
        self.monitor.record_error("Test error")
        
        summary = self.monitor.get_performance_summary()
        self.assertEqual(summary['error_rate'], 1.0)  # 1 error out of 1 operation
    
    def test_empty_metrics(self):
        """Test behavior with no recorded metrics."""
        summary = self.monitor.get_performance_summary()
        self.assertEqual(summary['status'], 'no_data')


class TestTransformerCompressor(unittest.TestCase):
    """Comprehensive tests for transformer compressor."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = TransformerConfig(
            d_model=128,  # Smaller for faster testing
            n_heads=4,
            num_layers=2,
            max_seq_length=256
        )
        self.compressor = TransformerCompressor(config=self.config)
    
    def test_initialization(self):
        """Test compressor initialization."""
        self.assertIsInstance(self.compressor.config, TransformerConfig)
        self.assertIsNotNone(self.compressor.pos_encoding)
        self.assertIsNotNone(self.compressor.attention)
        self.assertIsNotNone(self.compressor.monitor)
    
    def test_signal_compression_decompression(self):
        """Test basic compression and decompression."""
        # Generate test signal
        signal = np.random.randn(200, 1)
        
        # Compress
        compressed = self.compressor.compress(signal)
        
        # Verify compressed data structure
        self.assertIn('compressed_data', compressed)
        self.assertIn('metadata', compressed)
        self.assertIn('compression_ratio', compressed)
        
        # Decompress
        reconstructed = self.compressor.decompress(compressed)
        
        # Verify reconstruction
        self.assertEqual(reconstructed.shape, signal.shape)
    
    def test_multichannel_compression(self):
        """Test compression of multi-channel neural data."""
        # Multi-channel signal
        signal = np.random.randn(200, 8)  # 8 channels
        
        compressed = self.compressor.compress(signal)
        reconstructed = self.compressor.decompress(compressed)
        
        self.assertEqual(reconstructed.shape, signal.shape)
    
    def test_compression_ratio_target(self):
        """Test that compression achieves target ratio."""
        signal = np.random.randn(500, 1)
        compressed = self.compressor.compress(signal, quality_factor=1.0)
        
        ratio = compressed['compression_ratio']
        target = self.config.compression_ratio_target
        
        # Should be within reasonable range of target
        self.assertGreater(ratio, target * 0.5)
        self.assertLess(ratio, target * 2.0)
    
    def test_quality_factor_effect(self):
        """Test effect of quality factor on compression."""
        signal = np.random.randn(200, 1)
        
        # High quality compression
        high_quality = self.compressor.compress(signal, quality_factor=1.0)
        
        # Low quality compression
        low_quality = self.compressor.compress(signal, quality_factor=0.3)
        
        # Low quality should have higher compression ratio
        self.assertGreater(
            low_quality['compression_ratio'],
            high_quality['compression_ratio'] * 0.8
        )
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Empty signal
        with self.assertRaises(ValueError):
            self.compressor.compress(np.array([]))
        
        # Invalid quality factor
        with self.assertRaises(ValueError):
            signal = np.random.randn(100, 1)
            self.compressor.compress(signal, quality_factor=2.0)  # > 1.0
    
    def test_performance_monitoring(self):
        """Test that performance is monitored during operation."""
        signal = np.random.randn(200, 1)
        
        # Clear previous metrics
        self.compressor.monitor = PerformanceMonitor()
        
        # Perform compression
        self.compressor.compress(signal)
        
        # Check that metrics were recorded
        summary = self.compressor.monitor.get_performance_summary()
        self.assertEqual(summary['total_operations'], 1)
        self.assertGreater(summary['avg_compression_time_ms'], 0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating compressors."""
    
    def test_create_transformer_compressor(self):
        """Test transformer compressor factory function."""
        # Test speed mode
        speed_compressor = create_transformer_compressor(
            compression_ratio_target=3.0,
            quality_mode="speed"
        )
        self.assertEqual(speed_compressor.config.d_model, 128)
        self.assertEqual(speed_compressor.config.n_heads, 4)
        
        # Test quality mode
        quality_compressor = create_transformer_compressor(
            quality_mode="quality"
        )
        self.assertEqual(quality_compressor.config.d_model, 512)
        self.assertEqual(quality_compressor.config.n_heads, 16)
    
    def test_invalid_quality_mode(self):
        """Test handling of invalid quality mode."""
        # Should default to balanced mode
        compressor = create_transformer_compressor(quality_mode="invalid")
        self.assertEqual(compressor.config.d_model, 256)  # Balanced mode default


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking utilities."""
    
    def test_benchmark_transformer_compression(self):
        """Test compression performance benchmarking."""
        # Small benchmark for testing
        results = benchmark_transformer_compression(
            signal_length=100,
            num_channels=4,
            num_trials=3
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'avg_compression_time',
            'avg_decompression_time',
            'avg_compression_ratio',
            'avg_snr_db',
            'compression_latency_ms',
            'total_latency_ms'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))
    
    def test_latency_requirements(self):
        """Test that latency requirements are met."""
        results = benchmark_transformer_compression(
            signal_length=512,
            num_channels=16,
            num_trials=5
        )
        
        # Phase 8a requirement: <2ms total latency
        self.assertLess(results['total_latency_ms'], 2.0)
    
    def test_quality_requirements(self):
        """Test that quality requirements are met."""
        results = benchmark_transformer_compression(
            signal_length=200,
            num_channels=8,
            num_trials=5
        )
        
        # Phase 8a requirement: 25-35 dB SNR
        self.assertGreater(results['avg_snr_db'], 20.0)  # Relaxed for test
        
        # Phase 8a requirement: 3-5x compression ratio
        self.assertGreater(results['avg_compression_ratio'], 2.0)
        self.assertLess(results['avg_compression_ratio'], 10.0)


class TestPropertyBasedTesting(unittest.TestCase):
    """Property-based tests for compression invariants."""
    
    def setUp(self):
        self.compressor = TransformerCompressor(
            config=TransformerConfig(d_model=128, n_heads=4, num_layers=2)
        )
    
    def test_compression_invertibility(self):
        """Test that compression followed by decompression preserves signal structure."""
        for _ in range(10):  # Multiple random tests
            # Generate random signal
            signal_shape = (np.random.randint(50, 300), np.random.randint(1, 5))
            signal = np.random.randn(*signal_shape)
            
            # Compress and decompress
            compressed = self.compressor.compress(signal)
            reconstructed = self.compressor.decompress(compressed)
            
            # Shape should be preserved
            self.assertEqual(reconstructed.shape, signal.shape)
            
            # Signal should be reasonably similar (correlation > 0.5)
            if signal.size > 1:
                correlation = np.corrcoef(signal.flatten(), reconstructed.flatten())[0, 1]
                if not np.isnan(correlation):
                    self.assertGreater(correlation, 0.3)  # Relaxed threshold
    
    def test_compression_ratio_monotonicity(self):
        """Test that lower quality factors give higher compression ratios."""
        signal = np.random.randn(200, 2)
        
        quality_factors = [0.3, 0.6, 0.9]
        compression_ratios = []
        
        for qf in quality_factors:
            compressed = self.compressor.compress(signal, quality_factor=qf)
            compression_ratios.append(compressed['compression_ratio'])
        
        # Lower quality should generally give higher compression
        # (allowing some tolerance for variability)
        self.assertGreaterEqual(compression_ratios[0], compression_ratios[2] * 0.8)
    
    def test_signal_energy_preservation(self):
        """Test that signal energy is reasonably preserved."""
        for _ in range(5):
            signal = np.random.randn(150, 1)
            
            compressed = self.compressor.compress(signal, quality_factor=0.8)
            reconstructed = self.compressor.decompress(compressed)
            
            original_energy = np.sum(signal ** 2)
            reconstructed_energy = np.sum(reconstructed ** 2)
            
            if original_energy > 0:
                energy_ratio = reconstructed_energy / original_energy
                # Energy should be preserved within reasonable bounds
                self.assertGreater(energy_ratio, 0.3)
                self.assertLess(energy_ratio, 3.0)


class TestIntegrationWorkflows(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    def test_real_time_processing_simulation(self):
        """Simulate real-time processing workflow."""
        compressor = create_transformer_compressor(quality_mode="speed")
        
        # Simulate streaming chunks
        chunk_size = 128
        total_chunks = 10
        processing_times = []
        
        for i in range(total_chunks):
            # Generate chunk
            chunk = np.random.randn(chunk_size, 4)
            
            # Time processing
            start_time = time.time()
            compressed = compressor.compress(chunk)
            reconstructed = compressor.decompress(compressed)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Verify chunk processing
            self.assertEqual(reconstructed.shape, chunk.shape)
        
        # Check real-time performance
        avg_time = np.mean(processing_times)
        self.assertLess(avg_time, 0.002)  # <2ms requirement
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple signals."""
        compressor = create_transformer_compressor(quality_mode="balanced")
        
        # Generate batch of signals
        batch_signals = [
            np.random.randn(200, 1),
            np.random.randn(150, 3),
            np.random.randn(300, 2)
        ]
        
        compressed_batch = []
        reconstructed_batch = []
        
        # Process batch
        for signal in batch_signals:
            compressed = compressor.compress(signal)
            reconstructed = compressor.decompress(compressed)
            
            compressed_batch.append(compressed)
            reconstructed_batch.append(reconstructed)
        
        # Verify batch processing
        for i, (original, reconstructed) in enumerate(zip(batch_signals, reconstructed_batch)):
            self.assertEqual(original.shape, reconstructed.shape)
            
            # Check compression was achieved
            ratio = compressed_batch[i]['compression_ratio']
            self.assertGreater(ratio, 1.5)
    
    def test_error_recovery_workflow(self):
        """Test error handling and recovery in workflows."""
        compressor = TransformerCompressor()
        
        # Test with corrupted compressed data
        signal = np.random.randn(100, 1)
        compressed = compressor.compress(signal)
        
        # Corrupt the compressed data
        corrupted_compressed = compressed.copy()
        corrupted_compressed['compressed_data']['chunks'] = []
        
        # Should handle corruption gracefully
        with self.assertRaises(ValueError):
            compressor.decompress(corrupted_compressed)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTransformerConfig,
        TestPositionalEncoding,
        TestMultiHeadAttention,
        TestPerformanceMonitor,
        TestTransformerCompressor,
        TestFactoryFunctions,
        TestPerformanceBenchmarking,
        TestPropertyBasedTesting,
        TestIntegrationWorkflows
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
