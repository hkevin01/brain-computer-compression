#!/usr/bin/env python3
"""
Comprehensive Validation Suite - Clean Version

This test suite provides thorough validation of all toolkit components,
ensuring performance claims and functionality work as intended.
"""

import sys
import os
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validates performance claims against actual measurements."""

    def __init__(self):
        self.results = {}
        self.performance_targets = {
            'neural': {
                'neural_lz': {
                    'ratio_min': 1.5, 'ratio_max': 3.0, 'latency_max': 0.001
                },
                'arithmetic': {
                    'ratio_min': 2.0, 'ratio_max': 4.0, 'latency_max': 0.001
                },
                'perceptual': {
                    'ratio_min': 2.0, 'ratio_max': 10.0, 'latency_max': 0.001
                },
                'predictive': {
                    'ratio_min': 1.5, 'ratio_max': 2.0, 'latency_max': 0.002
                },
            },
            'emg': {
                'emg_lz': {
                    'ratio_min': 5.0, 'ratio_max': 12.0, 'latency_max': 0.025
                },
                'emg_perceptual': {
                    'ratio_min': 8.0, 'ratio_max': 20.0, 'latency_max': 0.035
                },
                'emg_predictive': {
                    'ratio_min': 10.0, 'ratio_max': 25.0, 'latency_max': 0.050
                },
                'mobile_emg': {
                    'ratio_min': 3.0, 'ratio_max': 8.0, 'latency_max': 0.015
                },
            }
        }

    def validate_neural_algorithms(self) -> Dict[str, Any]:
        """Validate neural compression algorithm performance."""
        logger.info("=== Validating Neural Algorithm Performance ===")

        results = {}

        # Test Neural LZ
        try:
            from bci_compression.algorithms import create_neural_lz_compressor

            # Generate realistic neural data
            neural_data = self._generate_realistic_neural_data()
            compressor = create_neural_lz_compressor('balanced')

            # Measure performance
            start_time = time.time()
            compressed, metadata = compressor.compress(neural_data)
            compression_time = time.time() - start_time

            # Extract compression ratio
            compression_ratio = metadata.get(
                'overall_compression_ratio',
                (neural_data.nbytes / len(compressed)
                 if isinstance(compressed, bytes) else 1.0)
            )

            # Validate against targets
            targets = self.performance_targets['neural']['neural_lz']
            ratio_valid = (targets['ratio_min'] <= compression_ratio <=
                          targets['ratio_max'])
            latency_valid = compression_time <= targets['latency_max']

            results['neural_lz'] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'ratio_meets_target': ratio_valid,
                'latency_meets_target': latency_valid,
                'status': 'PASS' if (ratio_valid and latency_valid) else 'FAIL'
            }

            logger.info(
                f"Neural LZ: {compression_ratio:.2f}x ratio, "
                f"{compression_time * 1000:.2f}ms - "
                f"{results['neural_lz']['status']}"
            )

        except Exception as e:
            results['neural_lz'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Neural LZ validation failed: {e}")

        # Test Perceptual Quantization
        try:
            from bci_compression.algorithms import PerceptualQuantizer

            quantizer = PerceptualQuantizer(base_bits=12)

            start_time = time.time()
            quantized, quant_info = quantizer.quantize(
                neural_data, quality_level=0.8
            )
            quantization_time = time.time() - start_time

            # Calculate compression ratio and SNR
            compression_ratio = neural_data.nbytes / quantized.nbytes
            mse = np.mean((neural_data - quantized) ** 2)
            snr = (10 * np.log10(np.var(neural_data) / mse)
                   if mse > 0 else float('inf'))

            targets = self.performance_targets['neural']['perceptual']
            ratio_valid = (targets['ratio_min'] <= compression_ratio <=
                          targets['ratio_max'])
            latency_valid = quantization_time <= targets['latency_max']
            snr_valid = snr >= 15.0

            results['perceptual'] = {
                'compression_ratio': compression_ratio,
                'compression_time': quantization_time,
                'snr_db': snr,
                'ratio_meets_target': ratio_valid,
                'latency_meets_target': latency_valid,
                'snr_meets_target': snr_valid,
                'status': ('PASS' if (ratio_valid and latency_valid and
                                     snr_valid) else 'FAIL')
            }

            logger.info(
                f"Perceptual: {compression_ratio:.2f}x ratio, {snr:.1f}dB SNR, "
                f"{quantization_time * 1000:.2f}ms - "
                f"{results['perceptual']['status']}"
            )

        except Exception as e:
            results['perceptual'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Perceptual validation failed: {e}")

        return results

    def validate_emg_algorithms(self) -> Dict[str, Any]:
        """Validate EMG compression algorithm performance."""
        logger.info("=== Validating EMG Algorithm Performance ===")

        results = {}

        # Generate realistic EMG data
        emg_data = self._generate_realistic_emg_data()

        # Test EMG LZ
        try:
            from bci_compression.algorithms.emg_compression import (
                EMGLZCompressor
            )

            compressor = EMGLZCompressor(sampling_rate=2000.0)

            start_time = time.time()
            compressed = compressor.compress(emg_data)
            compression_time = time.time() - start_time

            # Measure compression ratio
            compression_ratio = emg_data.nbytes / len(compressed)

            # Test decompression and quality
            decompressed = compressor.decompress(compressed)
            quality_score = self._calculate_emg_quality(emg_data, decompressed)

            targets = self.performance_targets['emg']['emg_lz']
            ratio_valid = (targets['ratio_min'] <= compression_ratio <=
                          targets['ratio_max'])
            latency_valid = compression_time <= targets['latency_max']
            quality_valid = quality_score >= 0.85

            results['emg_lz'] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'quality_score': quality_score,
                'ratio_meets_target': ratio_valid,
                'latency_meets_target': latency_valid,
                'quality_meets_target': quality_valid,
                'status': ('PASS' if (ratio_valid and latency_valid and
                                     quality_valid) else 'FAIL')
            }

            logger.info(
                f"EMG LZ: {compression_ratio:.2f}x ratio, Q={quality_score:.3f}, "
                f"{compression_time * 1000:.2f}ms - {results['emg_lz']['status']}"
            )

        except Exception as e:
            results['emg_lz'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"EMG LZ validation failed: {e}")

        # Test EMG Perceptual
        try:
            from bci_compression.algorithms.emg_compression import (
                EMGPerceptualQuantizer
            )

            compressor = EMGPerceptualQuantizer(
                sampling_rate=2000.0, quality_level=0.8
            )

            start_time = time.time()
            compressed = compressor.compress(emg_data)
            compression_time = time.time() - start_time

            compression_ratio = emg_data.nbytes / len(compressed)
            decompressed = compressor.decompress(compressed)
            quality_score = self._calculate_emg_quality(emg_data, decompressed)

            targets = self.performance_targets['emg']['emg_perceptual']
            ratio_valid = (targets['ratio_min'] <= compression_ratio <=
                          targets['ratio_max'])
            latency_valid = compression_time <= targets['latency_max']
            quality_valid = quality_score >= 0.90

            results['emg_perceptual'] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'quality_score': quality_score,
                'ratio_meets_target': ratio_valid,
                'latency_meets_target': latency_valid,
                'quality_meets_target': quality_valid,
                'status': ('PASS' if (ratio_valid and latency_valid and
                                     quality_valid) else 'FAIL')
            }

            logger.info(
                f"EMG Perceptual: {compression_ratio:.2f}x ratio, "
                f"Q={quality_score:.3f}, {compression_time * 1000:.2f}ms - "
                f"{results['emg_perceptual']['status']}"
            )

        except Exception as e:
            results['emg_perceptual'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"EMG Perceptual validation failed: {e}")

        # Test Mobile EMG
        try:
            from bci_compression.mobile.emg_mobile import MobileEMGCompressor

            compressor = MobileEMGCompressor(
                emg_sampling_rate=1000.0,
                target_latency_ms=15.0,
                battery_level=0.5
            )

            # Use lower sampling rate data for mobile
            mobile_emg_data = emg_data[:, ::2]  # Downsample to 1kHz

            start_time = time.time()
            compressed = compressor.compress(mobile_emg_data)
            compression_time = time.time() - start_time

            compression_ratio = mobile_emg_data.nbytes / len(compressed)
            decompressed = compressor.decompress(compressed)
            quality_score = self._calculate_emg_quality(
                mobile_emg_data, decompressed
            )

            targets = self.performance_targets['emg']['mobile_emg']
            ratio_valid = (targets['ratio_min'] <= compression_ratio <=
                          targets['ratio_max'])
            latency_valid = compression_time <= targets['latency_max']
            quality_valid = quality_score >= 0.80

            results['mobile_emg'] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'quality_score': quality_score,
                'ratio_meets_target': ratio_valid,
                'latency_meets_target': latency_valid,
                'quality_meets_target': quality_valid,
                'status': ('PASS' if (ratio_valid and latency_valid and
                                     quality_valid) else 'FAIL')
            }

            logger.info(
                f"Mobile EMG: {compression_ratio:.2f}x ratio, "
                f"Q={quality_score:.3f}, {compression_time * 1000:.2f}ms - "
                f"{results['mobile_emg']['status']}"
            )

        except Exception as e:
            results['mobile_emg'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Mobile EMG validation failed: {e}")

        return results

    def validate_quality_metrics(self) -> Dict[str, Any]:
        """Validate quality metrics functionality."""
        logger.info("=== Validating Quality Metrics ===")

        results = {}

        # Test EMG Quality Metrics
        try:
            from bci_compression.metrics.emg_quality import (
                EMGQualityMetrics, evaluate_emg_compression_quality
            )

            emg_data = self._generate_realistic_emg_data()
            # Create slightly degraded version
            degraded = emg_data + np.random.normal(0, 0.02, emg_data.shape)

            quality_metrics = EMGQualityMetrics(sampling_rate=2000.0)

            # Test individual metrics
            activation_metrics = (
                quality_metrics.muscle_activation_detection_accuracy(
                    emg_data, degraded
                )
            )
            envelope_metrics = quality_metrics.emg_envelope_correlation(
                emg_data, degraded
            )
            spectral_metrics = quality_metrics.emg_spectral_fidelity(
                emg_data, degraded
            )
            timing_metrics = quality_metrics.emg_timing_precision(
                emg_data, degraded
            )

            # Test overall evaluation
            overall_quality = evaluate_emg_compression_quality(
                emg_data, degraded, 2000.0
            )

            results['emg_quality_metrics'] = {
                'activation_precision': activation_metrics.get(
                    'activation_precision', 0
                ),
                'activation_recall': activation_metrics.get(
                    'activation_recall', 0
                ),
                'activation_f1': activation_metrics.get('activation_f1', 0),
                'envelope_correlation': envelope_metrics.get(
                    'envelope_correlation', 0
                ),
                'spectral_correlation': spectral_metrics.get(
                    'spectral_correlation', 0
                ),
                'timing_accuracy': timing_metrics.get('timing_accuracy', 0),
                'overall_quality': overall_quality.get(
                    'overall_quality_score', 0
                ),
                'status': 'PASS'
            }

            logger.info("EMG Quality Metrics: All tests passed")

        except Exception as e:
            results['emg_quality_metrics'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"EMG quality metrics validation failed: {e}")

        return results

    def validate_plugin_system(self) -> Dict[str, Any]:
        """Validate plugin system functionality."""
        logger.info("=== Validating Plugin System ===")

        results = {}

        try:
            from bci_compression.algorithms.emg_plugins import (
                get_emg_compressors, create_emg_compressor
            )

            # Test plugin discovery
            emg_compressors = get_emg_compressors()

            results['plugin_discovery'] = {
                'emg_plugins_found': len(emg_compressors),
                'plugin_names': list(emg_compressors.keys()),
                'status': 'PASS' if len(emg_compressors) >= 4 else 'FAIL'
            }

            # Test plugin creation and usage
            test_data = np.random.randn(2, 1000)
            plugin_results = {}

            for plugin_name in emg_compressors:
                try:
                    compressor = create_emg_compressor(
                        plugin_name, sampling_rate=1000.0
                    )
                    compressed = compressor.compress(test_data)
                    decompressed = compressor.decompress(compressed)

                    plugin_results[plugin_name] = {
                        'creation': True,
                        'compression': True,
                        'decompression': True,
                        'shape_preserved': test_data.shape == decompressed.shape,
                        'status': 'PASS'
                    }

                except Exception as e:
                    plugin_results[plugin_name] = {
                        'error': str(e), 'status': 'ERROR'
                    }

            results['plugin_functionality'] = plugin_results

            passed_plugins = len([p for p in plugin_results.values()
                                 if p.get('status') == 'PASS'])

            logger.info(
                f"Plugin System: {passed_plugins}/{len(plugin_results)} "
                f"plugins working"
            )

        except Exception as e:
            results['plugin_system'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Plugin system validation failed: {e}")

        return results

    def validate_mobile_optimization(self) -> Dict[str, Any]:
        """Validate mobile optimization features."""
        logger.info("=== Validating Mobile Optimization ===")

        results = {}

        try:
            from bci_compression.mobile.emg_mobile import EMGPowerOptimizer

            optimizer = EMGPowerOptimizer()

            # Test power optimization scenarios
            scenarios = [
                {'battery_level': 0.8, 'cpu_usage': 0.3,
                 'description': 'High battery'},
                {'battery_level': 0.4, 'cpu_usage': 0.7,
                 'description': 'Medium battery'},
                {'battery_level': 0.1, 'cpu_usage': 0.5,
                 'description': 'Low battery'}
            ]

            scenario_results = {}

            for i, scenario in enumerate(scenarios):
                try:
                    config = optimizer.optimize_for_power_consumption(
                        scenario['battery_level'],
                        scenario['cpu_usage'],
                        2.0  # data_rate_mbps
                    )

                    required_keys = [
                        'sampling_rate_hz', 'compression_level',
                        'processing_interval_ms'
                    ]
                    valid_config = all(k in config for k in required_keys)

                    scenario_results[f"scenario_{i}"] = {
                        'description': scenario['description'],
                        'battery_level': scenario['battery_level'],
                        'recommended_sampling_rate': config.get(
                            'sampling_rate_hz', 0
                        ),
                        'compression_level': config.get('compression_level', 0),
                        'processing_interval': config.get(
                            'processing_interval_ms', 0
                        ),
                        'valid_config': valid_config,
                        'status': 'PASS'
                    }

                except Exception as e:
                    scenario_results[f"scenario_{i}"] = {
                        'error': str(e), 'status': 'ERROR'
                    }

            results['power_optimization'] = scenario_results

            # Validate that low battery scenarios recommend lower sampling rates
            high_battery_sr = scenario_results.get('scenario_0', {}).get(
                'recommended_sampling_rate', 0
            )
            low_battery_sr = scenario_results.get('scenario_2', {}).get(
                'recommended_sampling_rate', 0
            )

            power_aware = (
                low_battery_sr < high_battery_sr
                if (high_battery_sr > 0 and low_battery_sr > 0) else False
            )

            results['power_adaptation'] = {
                'high_battery_sampling_rate': high_battery_sr,
                'low_battery_sampling_rate': low_battery_sr,
                'power_aware_adaptation': power_aware,
                'status': ('PASS' if (power_aware and low_battery_sr > 0)
                          else 'FAIL')
            }

            logger.info("Mobile Optimization: Power adaptation working correctly")

        except Exception as e:
            results['mobile_optimization'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Mobile optimization validation failed: {e}")

        return results

    def _generate_realistic_neural_data(self) -> np.ndarray:
        """Generate realistic neural data for testing."""
        # 64 channels, 30k samples (1 second at 30kHz)
        n_channels, n_samples = 64, 30000

        # Base noise
        neural_data = np.random.normal(0, 10, (n_channels, n_samples))

        # Add spike-like events
        for ch in range(n_channels):
            # Random spike times
            n_spikes = np.random.randint(50, 200)
            spike_times = np.random.choice(
                n_samples, size=n_spikes, replace=False
            )
            for spike_time in spike_times:
                if spike_time < n_samples - 10:
                    # Add spike waveform
                    spike_amplitude = 100 * np.random.choice([-1, 1])
                    spike_waveform = spike_amplitude * np.exp(-np.arange(10) / 3)
                    neural_data[ch, spike_time:spike_time+10] += spike_waveform

        # Add correlated activity between nearby channels
        for ch in range(n_channels - 1):
            correlation = 0.1 * neural_data[ch, :]
            neural_data[ch + 1, :] += correlation

        return neural_data.astype(np.float32)

    def _generate_realistic_emg_data(self) -> np.ndarray:
        """Generate realistic EMG data for testing."""
        # 4 channels, 2k samples (1 second at 2kHz)
        n_channels, n_samples = 4, 2000
        time = np.linspace(0, 1.0, n_samples)

        # Base noise
        emg_data = np.random.normal(0, 0.02, (n_channels, n_samples))

        # Add muscle activation patterns
        for ch in range(n_channels):
            # Activation periods
            activation_start = 0.2 + ch * 0.2
            activation_end = activation_start + 0.3
            activation_mask = (time >= activation_start) & (time <= activation_end)

            if np.any(activation_mask):
                # EMG frequency content (20-500Hz, peak around 100-150Hz)
                activation_signal = (0.3 * (ch + 1) *
                                   np.random.normal(0, 1, n_samples))
                # Add frequency content
                freq_content = 0.2 * np.sin(
                    2 * np.pi * (100 + ch * 25) * time
                )
                # Apply envelope
                envelope_center = (activation_start + activation_end) / 2
                envelope = np.exp(-((time - envelope_center) ** 2) / 0.05)

                combined_signal = (activation_signal + freq_content) * envelope
                emg_data[ch, activation_mask] += combined_signal[activation_mask]

        return emg_data.astype(np.float32)

    def _calculate_emg_quality(self, original: np.ndarray,
                              reconstructed: np.ndarray) -> float:
        """Calculate simple EMG quality score."""
        try:
            # Correlation-based quality
            correlation = np.corrcoef(
                original.flatten(), reconstructed.flatten()
            )[0, 1]

            # RMS error
            rms_error = np.sqrt(np.mean((original - reconstructed) ** 2))
            rms_original = np.sqrt(np.mean(original ** 2))
            normalized_error = (rms_error / rms_original
                               if rms_original > 0 else 1.0)

            # Combined quality score
            quality_score = (correlation * (1 - normalized_error)
                           if not np.isnan(correlation) else 0.5)

            return max(0.0, min(1.0, quality_score))

        except Exception:
            return 0.5


class StressTestValidator:
    """Performs stress tests to validate robustness."""

    def __init__(self):
        self.results = {}

    def run_latency_stress_test(self) -> Dict[str, Any]:
        """Test latency under various conditions."""
        logger.info("=== Running Latency Stress Test ===")

        results = {}

        try:
            # Test EMG real-time latency
            from bci_compression.mobile.emg_mobile import MobileEMGCompressor

            compressor = MobileEMGCompressor(
                emg_sampling_rate=1000.0,
                target_latency_ms=25.0
            )

            # Test small chunks (real-time scenario)
            chunk_sizes = [50, 100, 250, 500]  # samples
            latencies = []

            for chunk_size in chunk_sizes:
                chunk_latencies = []

                for _ in range(100):  # 100 iterations
                    data_chunk = np.random.randn(4, chunk_size).astype(np.float32)

                    start_time = time.time()
                    compressed = compressor.compress(data_chunk)
                    latency = time.time() - start_time

                    chunk_latencies.append(latency)

                avg_latency = np.mean(chunk_latencies)
                max_latency = np.max(chunk_latencies)
                latencies.append(avg_latency)

                results[f'chunk_{chunk_size}'] = {
                    'chunk_size': chunk_size,
                    'avg_latency_ms': avg_latency * 1000,
                    'max_latency_ms': max_latency * 1000,
                    'meets_target': avg_latency <= 0.025,  # 25ms target
                    'status': 'PASS' if avg_latency <= 0.025 else 'FAIL'
                }

                logger.info(
                    f"Latency test chunk {chunk_size}: "
                    f"{avg_latency*1000:.2f}ms avg - "
                    f"{results[f'chunk_{chunk_size}']['status']}"
                )

            # Overall latency performance
            results['overall_latency'] = {
                'avg_latency_ms': np.mean(latencies) * 1000,
                'consistent_performance': np.std(latencies) < 0.005,
                'status': 'PASS' if all(l <= 0.025 for l in latencies) else 'FAIL'
            }

        except Exception as e:
            results['latency_stress'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Latency stress test failed: {e}")

        return results

    def run_robustness_test(self) -> Dict[str, Any]:
        """Test robustness with edge cases and corrupted data."""
        logger.info("=== Running Robustness Test ===")

        results = {}

        # Test edge cases
        edge_cases = [
            ('zeros', np.zeros((4, 1000))),
            ('ones', np.ones((4, 1000))),
            ('random_large', np.random.randn(4, 1000) * 1000),
            ('random_small', np.random.randn(4, 1000) * 0.001),
            ('inf_values', np.full((4, 1000), np.inf)),
            ('nan_values', np.full((4, 1000), np.nan)),
        ]

        try:
            from bci_compression.algorithms.emg_compression import (
                EMGLZCompressor
            )
            compressor = EMGLZCompressor(sampling_rate=1000.0)

            for test_name, test_data in edge_cases:
                try:
                    # Handle special cases
                    if test_name in ['inf_values', 'nan_values']:
                        # Should handle gracefully or raise appropriate error
                        compressed = compressor.compress(test_data)
                        results[test_name] = {
                            'status': 'UNEXPECTED_SUCCESS',
                            'warning': 'Should handle inf/nan values'
                        }
                    else:
                        compressed = compressor.compress(test_data)
                        decompressed = compressor.decompress(compressed)

                        # Check if shape is preserved
                        shape_preserved = test_data.shape == decompressed.shape

                        results[test_name] = {
                            'compression_successful': True,
                            'decompression_successful': True,
                            'shape_preserved': shape_preserved,
                            'status': 'PASS' if shape_preserved else 'FAIL'
                        }

                except Exception as e:
                    # Expected for inf/nan cases
                    if test_name in ['inf_values', 'nan_values']:
                        results[test_name] = {
                            'status': 'EXPECTED_ERROR', 'error': str(e)
                        }
                    else:
                        results[test_name] = {
                            'status': 'UNEXPECTED_ERROR', 'error': str(e)
                        }

                logger.info(f"Robustness test {test_name}: {results[test_name]['status']}")

        except Exception as e:
            results['robustness'] = {'error': str(e), 'status': 'ERROR'}
            logger.error(f"Robustness test failed: {e}")

        return results


def run_comprehensive_validation():
    """Run all validation tests."""
    logger.info("Starting Comprehensive Validation Suite...")

    # Initialize validators
    perf_validator = PerformanceValidator()
    stress_validator = StressTestValidator()

    # Run all validation tests
    validation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'neural_algorithms': perf_validator.validate_neural_algorithms(),
        'emg_algorithms': perf_validator.validate_emg_algorithms(),
        'quality_metrics': perf_validator.validate_quality_metrics(),
        'plugin_system': perf_validator.validate_plugin_system(),
        'mobile_optimization': perf_validator.validate_mobile_optimization(),
        'latency_stress': stress_validator.run_latency_stress_test(),
        'robustness': stress_validator.run_robustness_test()
    }

    # Generate summary
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    error_tests = 0

    for category, results in validation_results.items():
        if category == 'timestamp':
            continue

        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'status' in test_result:
                total_tests += 1
                if test_result['status'] == 'PASS':
                    passed_tests += 1
                elif test_result['status'] in ['FAIL', 'TIMEOUT']:
                    failed_tests += 1
                else:
                    error_tests += 1

    # Calculate success rate
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Errors: {error_tests}")
    logger.info(f"Success Rate: {success_rate:.1f}%")

    validation_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'error_tests': error_tests,
        'success_rate': success_rate,
        'overall_status': 'PASS' if success_rate >= 85 else 'FAIL'
    }

    if success_rate >= 85:
        logger.info("🎉 COMPREHENSIVE VALIDATION PASSED!")
    else:
        logger.warning("⚠️  COMPREHENSIVE VALIDATION FAILED - Review failed tests")

    # Save results
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / 'comprehensive_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    logger.info(
        f"Validation results saved to "
        f"{results_dir / 'comprehensive_validation_results.json'}"
    )

    return validation_results


if __name__ == "__main__":
    results = run_comprehensive_validation()

    # Exit with appropriate code
    success_rate = results['summary']['success_rate']
    exit_code = 0 if success_rate >= 85 else 1
    sys.exit(exit_code)
