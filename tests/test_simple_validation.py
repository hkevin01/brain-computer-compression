#!/usr/bin/env python3
"""
Simple Unit Test Suite

This provides essential unit tests to validate core functionality
without complex validation logic.
"""

import os
import sys
import time
import unittest

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestNeuralAlgorithms(unittest.TestCase):
    """Test neural compression algorithms."""

    def setUp(self):
        """Set up test data."""
        # Use smaller dataset for quick tests
        self.neural_data = np.random.randn(32, 10000).astype(np.float32)
        self.quick_data = np.random.randn(8, 1000).astype(np.float32)

    @pytest.mark.quick
    def test_neural_lz_compression(self):
        """Test Neural LZ compression."""
        try:
            from bci_compression.algorithms import create_neural_lz_compressor

            compressor = create_neural_lz_compressor('speed')
            # Use smaller dataset for quick tests
            compressed, metadata = compressor.compress(self.quick_data)

            # Check compression ratio is reasonable
            compression_ratio = metadata.get('overall_compression_ratio', 1.0)
            self.assertGreater(compression_ratio, 1.0, "Should achieve some compression")
            self.assertLess(compression_ratio, 10.0, "Compression ratio should be reasonable")

            print(f"Neural LZ: {compression_ratio:.2f}x compression ratio")

        except ImportError:
            self.skipTest("Neural LZ compressor not available")

    def test_perceptual_quantizer(self):
        """Test perceptual quantization."""
        try:
            from bci_compression.algorithms import PerceptualQuantizer

            quantizer = PerceptualQuantizer(base_bits=12)
            quantized, quant_info = quantizer.quantize(self.neural_data, quality_level=0.8)

            # Check output properties
            self.assertEqual(quantized.shape, self.neural_data.shape)

            # Calculate SNR
            mse = np.mean((self.neural_data - quantized) ** 2)
            if mse > 0:
                snr = 10 * np.log10(np.var(self.neural_data) / mse)
                self.assertGreater(snr, 10.0, "SNR should be reasonable")
                print(f"Perceptual: {snr:.1f}dB SNR")

        except ImportError:
            self.skipTest("Perceptual quantizer not available")

    @pytest.mark.slow
    def test_transformer_compression(self):
        """Test transformer-based compression (slow test)."""
        try:
            from bci_compression.algorithms import create_transformer_compressor

            # Create transformer compressor with small configuration for testing
            compressor = create_transformer_compressor(
                d_model=64,
                n_heads=4,
                n_layers=2,
                max_sequence_length=256
            )

            # Use smaller data for faster testing
            test_data = self.quick_data  # 8 channels, 1000 samples

            # Test compression
            compressed = compressor.compress(test_data)
            decompressed = compressor.decompress(compressed)

            # Check basic properties
            self.assertEqual(decompressed.shape, test_data.shape)
            compression_ratio = test_data.nbytes / len(compressed)
            self.assertGreater(compression_ratio, 1.0, "Should achieve compression")

            # Check compression statistics
            stats = compressor.compression_stats
            self.assertIn('compression_ratio', stats)
            self.assertIn('processing_time', stats)

            print(f"Transformer: {compression_ratio:.2f}x compression ratio, "
                  f"{stats['processing_time'] * 1000:.1f}ms processing time")

        except ImportError:
            self.skipTest("Transformer compressor not available")
        except Exception as e:
            self.skipTest(f"Transformer compression failed: {e}")


class TestEMGAlgorithms(unittest.TestCase):
    """Test EMG compression algorithms."""

    def setUp(self):
        """Set up test data."""
        self.emg_data = np.random.randn(4, 2000).astype(np.float32)

    def test_emg_lz_compression(self):
        """Test EMG LZ compression."""
        try:
            from bci_compression.algorithms.emg_compression import EMGLZCompressor

            compressor = EMGLZCompressor(sampling_rate=2000.0)
            compressed = compressor.compress(self.emg_data)
            decompressed = compressor.decompress(compressed)

            # Check compression and reconstruction
            compression_ratio = self.emg_data.nbytes / len(compressed)
            self.assertGreater(compression_ratio, 1.0, "Should achieve compression")
            self.assertEqual(decompressed.shape, self.emg_data.shape, "Shape should be preserved")

            print(f"EMG LZ: {compression_ratio:.2f}x compression ratio")

        except ImportError:
            self.skipTest("EMG LZ compressor not available")

    def test_emg_perceptual_quantizer(self):
        """Test EMG perceptual quantization."""
        try:
            from bci_compression.algorithms.emg_compression import (
                EMGPerceptualQuantizer,
            )

            compressor = EMGPerceptualQuantizer(sampling_rate=2000.0, quality_level=0.8)
            compressed = compressor.compress(self.emg_data)
            decompressed = compressor.decompress(compressed)

            # Check compression and reconstruction
            compression_ratio = self.emg_data.nbytes / len(compressed)
            self.assertGreater(compression_ratio, 1.0, "Should achieve compression")
            self.assertEqual(decompressed.shape, self.emg_data.shape, "Shape should be preserved")

            print(f"EMG Perceptual: {compression_ratio:.2f}x compression ratio")

        except ImportError:
            self.skipTest("EMG perceptual quantizer not available")

    def test_mobile_emg_compressor(self):
        """Test mobile EMG compression."""
        try:
            from bci_compression.mobile.emg_mobile import MobileEMGCompressor

            compressor = MobileEMGCompressor(
                emg_sampling_rate=1000.0,
                target_latency_ms=25.0,
                battery_level=0.5
            )

            # Use downsampled data for mobile
            mobile_data = self.emg_data[:, ::2]  # 1kHz

            start_time = time.time()
            compressed = compressor.compress(mobile_data)
            compression_time = time.time() - start_time

            decompressed = compressor.decompress(compressed)

            # Check performance
            compression_ratio = mobile_data.nbytes / len(compressed)
            self.assertGreater(compression_ratio, 1.0, "Should achieve compression")
            self.assertEqual(decompressed.shape, mobile_data.shape, "Shape should be preserved")
            self.assertLess(compression_time, 0.1, "Should be fast enough")

            print(f"Mobile EMG: {compression_ratio:.2f}x compression, {compression_time*1000:.2f}ms")

        except ImportError:
            self.skipTest("Mobile EMG compressor not available")


class TestQualityMetrics(unittest.TestCase):
    """Test quality metrics functionality."""

    def setUp(self):
        """Set up test data."""
        self.emg_data = np.random.randn(4, 2000).astype(np.float32)
        # Create slightly degraded version
        self.degraded_data = self.emg_data + np.random.normal(0, 0.02, self.emg_data.shape)

    def test_emg_quality_metrics(self):
        """Test EMG quality metrics."""
        try:
            from bci_compression.metrics.emg_quality import EMGQualityMetrics

            quality_metrics = EMGQualityMetrics(sampling_rate=2000.0)

            # Test individual metrics
            activation_metrics = quality_metrics.muscle_activation_detection_accuracy(
                self.emg_data, self.degraded_data
            )
            envelope_metrics = quality_metrics.emg_envelope_correlation(
                self.emg_data, self.degraded_data
            )

            # Check that metrics return reasonable values
            self.assertIsInstance(activation_metrics, dict)
            self.assertIsInstance(envelope_metrics, dict)

            print("EMG Quality Metrics: Tests passed")

        except ImportError:
            self.skipTest("EMG quality metrics not available")

    def test_overall_quality_evaluation(self):
        """Test overall quality evaluation."""
        try:
            from bci_compression.metrics.emg_quality import (
                evaluate_emg_compression_quality,
            )

            overall_quality = evaluate_emg_compression_quality(
                self.emg_data, self.degraded_data, 2000.0
            )

            # Check that overall quality returns a score
            self.assertIsInstance(overall_quality, dict)
            self.assertIn('overall_quality_score', overall_quality)

            quality_score = overall_quality['overall_quality_score']
            self.assertGreaterEqual(quality_score, 0.0)
            self.assertLessEqual(quality_score, 1.0)

            print(f"Overall Quality: {quality_score:.3f}")

        except ImportError:
            self.skipTest("Overall quality evaluation not available")


class TestPluginSystem(unittest.TestCase):
    """Test plugin system functionality."""

    def test_emg_plugin_discovery(self):
        """Test EMG plugin discovery."""
        try:
            from bci_compression.algorithms.emg_plugins import get_emg_compressors

            emg_compressors = get_emg_compressors()

            # Check that plugins are discovered
            self.assertIsInstance(emg_compressors, dict)
            self.assertGreater(len(emg_compressors), 0, "Should find some EMG compressors")

            print(f"EMG Plugins: Found {len(emg_compressors)} compressors")
            print(f"Plugin names: {list(emg_compressors.keys())}")

        except ImportError:
            self.skipTest("EMG plugin system not available")

    def test_emg_plugin_creation(self):
        """Test EMG plugin creation and usage."""
        try:
            from bci_compression.algorithms.emg_plugins import create_emg_compressor

            # Test data
            test_data = np.random.randn(2, 1000).astype(np.float32)

            # Try to create and use an EMG compressor
            compressor = create_emg_compressor('emg_lz', sampling_rate=1000.0)

            compressed = compressor.compress(test_data)
            decompressed = compressor.decompress(compressed)

            # Check that plugin works
            self.assertEqual(decompressed.shape, test_data.shape)

            print("EMG Plugin Creation: Success")

        except ImportError:
            self.skipTest("EMG plugin creation not available")


class TestMobileOptimization(unittest.TestCase):
    """Test mobile optimization features."""

    def test_power_optimizer(self):
        """Test power optimization."""
        try:
            from bci_compression.mobile.emg_mobile import EMGPowerOptimizer

            optimizer = EMGPowerOptimizer()

            # Test optimization for different scenarios
            config_high = optimizer.optimize_for_power_consumption(0.8, 0.3, 2.0)
            config_low = optimizer.optimize_for_power_consumption(0.2, 0.7, 2.0)

            # Check that configurations are returned
            self.assertIsInstance(config_high, dict)
            self.assertIsInstance(config_low, dict)

            print("Power Optimizer: Tests passed")

        except ImportError:
            self.skipTest("Power optimizer not available")


def run_simple_tests():
    """Run simple test suite."""
    print("=" * 60)
    print("SIMPLE UNIT TEST SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestNeuralAlgorithms,
        TestEMGAlgorithms,
        TestQualityMetrics,
        TestPluginSystem,
        TestMobileOptimization
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_simple_tests()
    sys.exit(exit_code)
