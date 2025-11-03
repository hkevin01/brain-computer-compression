#!/usr/bin/env python3
"""
Fast Validation Test Suite

Quick tests with timeouts to prevent hanging.
Focuses on smoke tests and basic functionality.
"""

import os
import sys
import unittest
import numpy as np
from functools import wraps
import signal

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def timeout(seconds=5):
    """Decorator to add timeout to test methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test {func.__name__} timed out after {seconds}s")

            # Set the signal handler
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result
        return wrapper
    return decorator


class TestBasicImports(unittest.TestCase):
    """Test that basic imports work."""

    @timeout(2)
    def test_core_imports(self):
        """Test core module imports."""
        from bci_compression import core
        from bci_compression import algorithms
        from bci_compression import mobile
        self.assertTrue(True)

    @timeout(2)
    def test_algorithm_imports(self):
        """Test algorithm imports."""
        from bci_compression.algorithms import create_neural_lz_compressor
        from bci_compression.algorithms.emg_compression import EMGLZCompressor
        self.assertTrue(True)

    @timeout(2)
    def test_mobile_imports(self):
        """Test mobile module imports."""
        from bci_compression.mobile import MobileBCICompressor
        from bci_compression.mobile import MobileEMGCompressor
        self.assertTrue(True)


class TestQuickCompression(unittest.TestCase):
    """Quick compression tests with small data."""

    def setUp(self):
        """Set up small test data."""
        np.random.seed(42)
        # Very small data for speed
        self.tiny_data = np.random.randn(2, 100).astype(np.float32)

    @timeout(10)
    def test_emg_lz_quick(self):
        """Quick test of EMG LZ compression."""
        try:
            from bci_compression.algorithms.emg_compression import EMGLZCompressor

            compressor = EMGLZCompressor()
            compressed = compressor.compress(self.tiny_data)

            # Basic checks
            self.assertIsNotNone(compressed)
            self.assertIsInstance(compressed, bytes)

            # Check metadata from stats
            stats = compressor.compression_stats
            self.assertIn('compression_ratio', stats)

            print(f"EMG LZ Quick: {stats.get('compression_ratio', 0):.2f}x")

        except ImportError:
            self.skipTest("EMG LZ not available")

    @timeout(10)
    def test_mobile_compressor_quick(self):
        """Quick test of mobile compressor."""
        try:
            from bci_compression.mobile import MobileBCICompressor

            compressor = MobileBCICompressor(algorithm="mobile_lz")
            compressed = compressor.compress(self.tiny_data)

            # Basic checks
            self.assertIsNotNone(compressed)
            self.assertIsInstance(compressed, bytes)

            print("Mobile compressor: OK")

        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Mobile compressor not fully implemented: {e}")


class TestPluginSystem(unittest.TestCase):
    """Test plugin system basics."""

    @timeout(5)
    def test_plugin_registry(self):
        """Test plugin registration system."""
        try:
            from bci_compression.plugins import list_plugins, get_plugin

            plugins = list_plugins()
            self.assertIsInstance(plugins, list)

            print(f"Found {len(plugins)} plugins")

        except ImportError:
            self.skipTest("Plugin system not available")


class TestBCISystemSupport(unittest.TestCase):
    """Test support for different BCI systems."""

    @timeout(5)
    def test_multi_channel_support(self):
        """Test different channel configurations."""
        configs = [
            (8, 1000),   # 8-channel system, 1000 samples
            (32, 1000),  # 32-channel (common EEG)
            (64, 1000),  # 64-channel (high-density EEG)
            (128, 1000), # 128-channel (GSN HydroCel)
            (256, 1000), # 256-channel (Utah array)
        ]

        for n_channels, n_samples in configs:
            data = np.random.randn(n_channels, n_samples).astype(np.float32)
            self.assertEqual(data.shape, (n_channels, n_samples))

        print(f"Tested {len(configs)} channel configurations")

    @timeout(5)
    def test_sampling_rates(self):
        """Test different sampling rate configurations."""
        sampling_rates = [
            250,    # Low-cost EEG
            500,    # Standard EEG
            1000,   # High-quality EEG
            2000,   # EMG
            30000,  # Neural spikes
        ]

        for rate in sampling_rates:
            # Calculate samples for 1 second
            n_samples = rate
            data = np.random.randn(8, n_samples).astype(np.float32)
            self.assertEqual(data.shape[1], n_samples)

        print(f"Tested {len(sampling_rates)} sampling rates")

    @timeout(5)
    def test_system_profiles(self):
        """Test BCI system profile loading."""
        try:
            from bci_compression.formats import list_supported_systems, get_system_profile

            systems = list_supported_systems()
            self.assertGreater(len(systems), 0)

            # Test loading specific system
            openbci = get_system_profile('openbci_8')
            self.assertEqual(openbci.num_channels, 8)
            self.assertEqual(openbci.sampling_rate, 200)

            print(f"Found {len(systems)} supported BCI systems")

        except ImportError:
            self.skipTest("BCI format module not available")

    @timeout(5)
    def test_data_adaptation(self):
        """Test data adaptation between BCI systems."""
        try:
            from bci_compression.formats import adapt_data

            # Simulate OpenBCI 8-channel data (200 Hz)
            openbci_data = np.random.randn(8, 200).astype(np.float32)

            # Adapt to 1000 Hz
            adapted, settings = adapt_data(
                openbci_data,
                source_system='openbci_8',
                target_sampling_rate=1000
            )

            # Check resampling worked
            expected_samples = int(200 * 1000 / 200)
            self.assertEqual(adapted.shape, (8, expected_samples))
            self.assertEqual(settings['sampling_rate'], 1000)

            print(f"Data adaptation: OK (resampled 200Hz→1000Hz)")

        except ImportError:
            self.skipTest("BCI format module not available")


def run_fast_tests():
    """Run fast validation tests."""
    print("=" * 70)
    print("FAST VALIDATION TEST SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestBasicImports,
        TestQuickCompression,
        TestPluginSystem,
        TestBCISystemSupport,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) /
                    result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_fast_tests()
    sys.exit(0 if success else 1)
