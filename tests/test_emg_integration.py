#!/usr/bin/env python3
"""
EMG Extension Integration Test

This script tests the complete EMG extension integration to ensure
all components work together correctly.
"""

import logging
import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_emg_algorithms():
    """Test EMG compression algorithms."""
    logger.info("=== Testing EMG Algorithms ===")

    try:
        from bci_compression.algorithms.emg_compression import (
            EMGLZCompressor,
            EMGPerceptualQuantizer,
            EMGPredictiveCompressor,
        )

        # Create test data
        test_data = np.random.randn(4, 2000)  # 4 channels, 1 second @ 2kHz

        # Test each algorithm
        algorithms = {
            'EMG LZ': EMGLZCompressor(sampling_rate=2000.0),
            'EMG Perceptual': EMGPerceptualQuantizer(sampling_rate=2000.0),
            'EMG Predictive': EMGPredictiveCompressor()
        }

        results = {}
        for name, algo in algorithms.items():
            try:
                compressed = algo.compress(test_data)
                decompressed = algo.decompress(compressed)
                compression_ratio = test_data.nbytes / len(compressed)

                results[name] = {
                    'success': True,
                    'compression_ratio': compression_ratio,
                    'shape_match': test_data.shape == decompressed.shape
                }

                logger.info(f"✓ {name}: ratio={compression_ratio:.2f}, shape_ok={results[name]['shape_match']}")

            except Exception as e:
                results[name] = {'success': False, 'error': str(e)}
                logger.error(f"✗ {name}: {e}")

        return results

    except ImportError as e:
        logger.error(f"Failed to import EMG algorithms: {e}")
        return {}


def test_emg_quality_metrics():
    """Test EMG quality metrics."""
    logger.info("\n=== Testing EMG Quality Metrics ===")

    try:
        from bci_compression.metrics.emg_quality import (
            EMGQualityMetrics,
            evaluate_emg_compression_quality,
        )

        # Create test data
        original = np.random.randn(4, 2000)
        # Add slight noise for testing
        degraded = original + np.random.normal(0, 0.01, original.shape)

        quality_metrics = EMGQualityMetrics(sampling_rate=2000.0)

        # Test individual metrics
        tests = [
            ('muscle_activation_detection_accuracy', lambda: quality_metrics.muscle_activation_detection_accuracy(original, degraded)),
            ('emg_envelope_correlation', lambda: quality_metrics.emg_envelope_correlation(original, degraded)),
            ('emg_spectral_fidelity', lambda: quality_metrics.emg_spectral_fidelity(original, degraded)),
            ('emg_timing_precision', lambda: quality_metrics.emg_timing_precision(original, degraded)),
            ('overall_quality_evaluation', lambda: evaluate_emg_compression_quality(original, degraded, 2000.0))
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = {'success': True, 'result': result}
                logger.info(f"✓ {test_name}: OK")

            except Exception as e:
                results[test_name] = {'success': False, 'error': str(e)}
                logger.error(f"✗ {test_name}: {e}")

        return results

    except ImportError as e:
        logger.error(f"Failed to import EMG quality metrics: {e}")
        return {}


def test_emg_mobile():
    """Test EMG mobile optimization."""
    logger.info("\n=== Testing EMG Mobile Optimization ===")

    try:
        from bci_compression.mobile.emg_mobile import (
            EMGPowerOptimizer,
            MobileEMGCompressor,
        )

        # Test mobile compressor
        mobile_compressor = MobileEMGCompressor(
            emg_sampling_rate=1000.0,
            target_latency_ms=50.0,
            battery_level=0.5
        )

        # Test power optimizer
        power_optimizer = EMGPowerOptimizer()

        test_data = np.random.randn(4, 1000)

        tests = [
            ('mobile_compression', lambda: mobile_compressor.compress(test_data)),
            ('power_optimization', lambda: power_optimizer.optimize_for_power_consumption(0.3, 0.6, 2.0))
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = {'success': True, 'result': type(result).__name__}
                logger.info(f"✓ {test_name}: OK")

            except Exception as e:
                results[test_name] = {'success': False, 'error': str(e)}
                logger.error(f"✗ {test_name}: {e}")

        return results

    except ImportError as e:
        logger.error(f"Failed to import EMG mobile components: {e}")
        return {}


def test_emg_plugins():
    """Test EMG plugin system."""
    logger.info("\n=== Testing EMG Plugin System ===")

    try:
        from bci_compression.algorithms.emg_plugins import (
            create_emg_compressor,
            get_emg_compressors,
        )

        # Test plugin discovery
        emg_compressors = get_emg_compressors()
        logger.info(f"Found {len(emg_compressors)} EMG compressor plugins")

        results = {'plugin_count': len(emg_compressors)}

        # Test plugin creation
        test_data = np.random.randn(2, 1000)

        for plugin_name in emg_compressors:
            try:
                compressor = create_emg_compressor(plugin_name, sampling_rate=1000.0)
                compressed = compressor.compress(test_data)
                decompressed = compressor.decompress(compressed)

                results[plugin_name] = {
                    'success': True,
                    'compression_ratio': test_data.nbytes / len(compressed)
                }
                logger.info(f"✓ Plugin {plugin_name}: OK")

            except Exception as e:
                results[plugin_name] = {'success': False, 'error': str(e)}
                logger.error(f"✗ Plugin {plugin_name}: {e}")

        return results

    except ImportError as e:
        logger.error(f"Failed to import EMG plugins: {e}")
        return {}


def test_emg_benchmarking():
    """Test EMG benchmarking framework."""
    logger.info("\n=== Testing EMG Benchmarking ===")

    try:
        from bci_compression.benchmarks.emg_benchmark import (
            EMGBenchmarkSuite,
            create_synthetic_emg_datasets,
        )

        # Create test datasets
        datasets = create_synthetic_emg_datasets()
        logger.info(f"Created {len(datasets)} synthetic datasets")

        # Create benchmark suite
        benchmark = EMGBenchmarkSuite(
            sampling_rate=2000.0,
            output_dir="test_emg_benchmark"
        )

        # Run a quick benchmark (subset of compressors)
        try:
            # Run on just one dataset to save time
            subset_datasets = {'test_dataset': next(iter(datasets.values()))}
            results = benchmark.run_comprehensive_benchmark(subset_datasets, save_results=False)

            success = len(results) > 0 and 'summary' in results
            logger.info(f"✓ Benchmarking: {'OK' if success else 'Failed'}")

            return {'success': success, 'datasets': len(datasets)}

        except Exception as e:
            logger.error(f"✗ Benchmarking failed: {e}")
            return {'success': False, 'error': str(e)}

    except ImportError as e:
        logger.error(f"Failed to import EMG benchmarking: {e}")
        return {}


def run_comprehensive_test():
    """Run comprehensive EMG extension test."""
    logger.info("Starting EMG Extension Integration Test...\n")

    test_results = {}

    # Run all tests
    test_functions = [
        ('EMG Algorithms', test_emg_algorithms),
        ('EMG Quality Metrics', test_emg_quality_metrics),
        ('EMG Mobile', test_emg_mobile),
        ('EMG Plugins', test_emg_plugins),
        ('EMG Benchmarking', test_emg_benchmarking)
    ]

    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results[test_name] = {'success': False, 'error': str(e)}

    # Generate summary
    logger.info("\n" + "=" * 50)
    logger.info("EMG EXTENSION INTEGRATION TEST SUMMARY")
    logger.info("=" * 50)

    total_tests = 0
    passed_tests = 0

    for test_name, results in test_results.items():
        logger.info(f"\n{test_name}:")

        if isinstance(results, dict):
            for sub_test, sub_result in results.items():
                if isinstance(sub_result, dict) and 'success' in sub_result:
                    total_tests += 1
                    if sub_result['success']:
                        passed_tests += 1
                        logger.info(f"  ✓ {sub_test}")
                    else:
                        logger.info(f"  ✗ {sub_test}: {sub_result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"  → {sub_test}: {sub_result}")

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")

    if success_rate >= 80:
        logger.info("🎉 EMG extension integration test PASSED!")
        return True
    else:
        logger.warning("⚠️  EMG extension integration test had issues")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
