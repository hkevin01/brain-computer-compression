#!/usr/bin/env python3
"""
EMG Compression Toolkit Demo

This script demonstrates the complete EMG compression functionality including:
- EMG data loading and preprocessing
- Various EMG compression algorithms
- Quality assessment and mobile optimization
- Benchmarking and performance evaluation

Usage:
    python emg_demo.py
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_compression.algorithms.emg_compression import (
    EMGLZCompressor,
    EMGPerceptualQuantizer,
    EMGPredictiveCompressor,
)
from bci_compression.benchmarks.emg_benchmark import (
    EMGBenchmarkSuite,
    create_synthetic_emg_datasets,
)
from bci_compression.metrics.emg_quality import (
    EMGQualityMetrics,
    evaluate_emg_compression_quality,
)
from bci_compression.mobile.emg_mobile import EMGPowerOptimizer, MobileEMGCompressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_emg_data(sampling_rate: float = 2000.0, duration: float = 10.0) -> np.ndarray:
    """
    Create realistic EMG demo data with multiple muscle activations.

    Parameters
    ----------
    sampling_rate : float, default=2000.0
        EMG sampling rate in Hz
    duration : float, default=10.0
        Signal duration in seconds

    Returns
    -------
    np.ndarray
        Multi-channel EMG data (channels x samples)
    """
    n_samples = int(duration * sampling_rate)
    n_channels = 4
    time = np.linspace(0, duration, n_samples)

    # Initialize with noise
    emg_data = np.random.normal(0, 0.02, (n_channels, n_samples))

    # Add realistic muscle activation patterns
    activation_patterns = [
        {'start': 1.0, 'end': 3.0, 'amplitude': 0.8, 'frequency': 120},  # Strong activation
        {'start': 4.0, 'end': 5.5, 'amplitude': 0.4, 'frequency': 150},  # Moderate activation
        {'start': 6.5, 'end': 8.0, 'amplitude': 0.6, 'frequency': 100},  # Fatigue pattern
        {'start': 8.5, 'end': 9.5, 'amplitude': 0.3, 'frequency': 180}   # Brief activation
    ]

    for ch in range(n_channels):
        for i, pattern in enumerate(activation_patterns):
            if ch == i:  # Each channel gets one primary activation
                mask = (time >= pattern['start']) & (time <= pattern['end'])

                # Create activation signal with realistic characteristics
                activation_signal = pattern['amplitude'] * np.random.normal(0, 1, n_samples)

                # Add frequency content
                freq_signal = 0.3 * np.sin(2 * np.pi * pattern['frequency'] * time)

                # Apply envelope
                envelope_center = (pattern['start'] + pattern['end']) / 2
                envelope_width = (pattern['end'] - pattern['start']) / 4
                envelope = np.exp(-((time - envelope_center) ** 2) / (2 * envelope_width ** 2))

                combined_signal = (activation_signal + freq_signal) * envelope
                emg_data[ch, mask] += combined_signal[mask]

        # Add some crosstalk between channels
        if ch > 0:
            crosstalk = 0.1 * emg_data[ch - 1, :]
            emg_data[ch, :] += crosstalk

    return emg_data


def demo_emg_compression_algorithms():
    """Demonstrate EMG compression algorithms."""
    logger.info("=== EMG Compression Algorithms Demo ===")

    # Create demo data
    sampling_rate = 2000.0
    emg_data = create_demo_emg_data(sampling_rate=sampling_rate)

    logger.info(f"Created EMG data: {emg_data.shape} ({emg_data.nbytes} bytes)")

    # Initialize compressors
    compressors = {
        'LZ': EMGLZCompressor(sampling_rate=sampling_rate),
        'Perceptual': EMGPerceptualQuantizer(sampling_rate=sampling_rate),
        'Predictive': EMGPredictiveCompressor(),
        'Mobile': MobileEMGCompressor(emg_sampling_rate=sampling_rate)
    }

    results = {}

    for name, compressor in compressors.items():
        logger.info(f"\nTesting {name} compressor...")

        try:
            # Compress
            compressed = compressor.compress(emg_data)
            compression_ratio = emg_data.nbytes / len(compressed)

            # Decompress
            decompressed = compressor.decompress(compressed)

            # Calculate quality
            quality = evaluate_emg_compression_quality(emg_data, decompressed, sampling_rate)

            results[name] = {
                'compression_ratio': compression_ratio,
                'compressed_size': len(compressed),
                'quality_score': quality.get('overall_quality_score', 0),
                'decompressed_data': decompressed
            }

            logger.info(f"  Compression ratio: {compression_ratio:.2f}")
            logger.info(f"  Quality score: {quality.get('overall_quality_score', 0):.3f}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results[name] = {'error': str(e)}

    return emg_data, results


def demo_emg_quality_metrics():
    """Demonstrate EMG quality metrics."""
    logger.info("\n=== EMG Quality Metrics Demo ===")

    sampling_rate = 2000.0
    quality_metrics = EMGQualityMetrics(sampling_rate)

    # Create test data
    original = create_demo_emg_data(sampling_rate=sampling_rate, duration=5.0)

    # Create slightly degraded version for comparison
    noise_level = 0.05
    degraded = original + np.random.normal(0, noise_level, original.shape)

    logger.info("Calculating EMG quality metrics...")

    # Test individual metrics
    activation_metrics = quality_metrics.muscle_activation_detection_accuracy(original, degraded)
    logger.info(f"Activation detection - Precision: {activation_metrics['activation_precision']:.3f}, "
                f"Recall: {activation_metrics['activation_recall']:.3f}")

    envelope_metrics = quality_metrics.emg_envelope_correlation(original, degraded)
    logger.info(f"Envelope correlation: {envelope_metrics['envelope_correlation']:.3f}")

    spectral_metrics = quality_metrics.emg_spectral_fidelity(original, degraded)
    logger.info(f"Spectral fidelity: {spectral_metrics['spectral_correlation']:.3f}")

    timing_metrics = quality_metrics.emg_timing_precision(original, degraded)
    logger.info(f"Timing precision: {timing_metrics['timing_accuracy']:.3f}")

    # Overall quality assessment
    overall_quality = evaluate_emg_compression_quality(original, degraded, sampling_rate)
    logger.info(f"Overall quality score: {overall_quality['overall_quality_score']:.3f}")


def demo_mobile_emg_optimization():
    """Demonstrate mobile EMG optimization."""
    logger.info("\n=== Mobile EMG Optimization Demo ===")

    # Create mobile compressor
    mobile_compressor = MobileEMGCompressor(
        emg_sampling_rate=1000.0,  # Lower sampling rate for mobile
        target_latency_ms=50.0,
        battery_level=0.3  # Low battery scenario
    )

    # Create EMG data stream
    emg_data = create_demo_emg_data(sampling_rate=1000.0, duration=2.0)

    logger.info(f"Mobile EMG data: {emg_data.shape}")

    # Test power optimization
    power_optimizer = EMGPowerOptimizer()

    # Simulate different power scenarios
    power_scenarios = [
        {'battery_level': 0.8, 'cpu_usage': 0.3, 'description': 'High battery, low CPU'},
        {'battery_level': 0.4, 'cpu_usage': 0.7, 'description': 'Medium battery, high CPU'},
        {'battery_level': 0.1, 'cpu_usage': 0.5, 'description': 'Low battery, medium CPU'}
    ]

    for scenario in power_scenarios:
        logger.info(f"\nTesting scenario: {scenario['description']}")

        power_config = power_optimizer.optimize_for_power_consumption(
            battery_level=scenario['battery_level'],
            cpu_usage=scenario['cpu_usage'],
            data_rate_mbps=2.0
        )

        logger.info(f"  Recommended sampling rate: {power_config['sampling_rate_hz']:.0f} Hz")
        logger.info(f"  Compression level: {power_config['compression_level']}")
        logger.info(f"  Processing interval: {power_config['processing_interval_ms']:.1f} ms")

        # Apply power-optimized compression
        mobile_compressor.config.update(power_config)

        try:
            compressed = mobile_compressor.compress(emg_data)
            compression_ratio = emg_data.nbytes / len(compressed)
            logger.info(f"  Compression ratio: {compression_ratio:.2f}")

        except Exception as e:
            logger.error(f"  Compression error: {e}")


def demo_emg_benchmarking():
    """Demonstrate EMG benchmarking suite."""
    logger.info("\n=== EMG Benchmarking Demo ===")

    # Create test datasets
    datasets = create_synthetic_emg_datasets()
    logger.info(f"Created {len(datasets)} test datasets")

    # Run benchmark
    benchmark = EMGBenchmarkSuite(sampling_rate=2000.0, output_dir="demo_results")
    results = benchmark.run_comprehensive_benchmark(datasets, save_results=True)

    # Display summary
    if 'summary' in results and 'best_performers' in results['summary']:
        best = results['summary']['best_performers']
        logger.info("\nBenchmark Summary:")
        logger.info(f"  Best compression: {best.get('compression_ratio', 'N/A')}")
        logger.info(f"  Best quality: {best.get('quality', 'N/A')}")
        logger.info(f"  Fastest: {best.get('speed', 'N/A')}")

    # Display average metrics
    if 'summary' in results and 'average_metrics' in results['summary']:
        logger.info("\nAverage Performance:")
        for compressor, metrics in results['summary']['average_metrics'].items():
            logger.info(f"  {compressor}:")
            logger.info(f"    Compression ratio: {metrics['avg_compression_ratio']:.2f}")
            logger.info(f"    Time: {metrics['avg_compression_time_ms']:.2f} ms")
            logger.info(f"    Quality: {metrics['avg_quality_score']:.3f}")

    return results


def create_demo_visualization(original_data, compression_results):
    """Create visualization of compression results."""
    logger.info("\n=== Creating Visualization ===")

    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)

    # Plot original data and compression results
    fig, axes = plt.subplots(len(compression_results) + 1, 1, figsize=(12, 8))

    # Plot original data
    time = np.linspace(0, original_data.shape[1] / 2000.0, original_data.shape[1])
    axes[0].plot(time, original_data[0, :])
    axes[0].set_title('Original EMG Signal (Channel 1)')
    axes[0].set_ylabel('Amplitude')

    # Plot compression results
    for i, (name, result) in enumerate(compression_results.items()):
        if 'error' not in result and 'decompressed_data' in result:
            decompressed = result['decompressed_data']
            axes[i + 1].plot(time, decompressed[0, :])
            axes[i + 1].set_title(f'{name} Compression (Ratio: {result["compression_ratio"]:.2f})')
            axes[i + 1].set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_dir / 'emg_compression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_dir / 'emg_compression_comparison.png'}")


def main():
    """Run complete EMG compression demo."""
    logger.info("Starting EMG Compression Toolkit Demo...")

    try:
        # Demo 1: Compression algorithms
        original_data, compression_results = demo_emg_compression_algorithms()

        # Demo 2: Quality metrics
        demo_emg_quality_metrics()

        # Demo 3: Mobile optimization
        demo_mobile_emg_optimization()

        # Demo 4: Benchmarking
        benchmark_results = demo_emg_benchmarking()

        # Demo 5: Visualization
        create_demo_visualization(original_data, compression_results)

        logger.info("\n=== Demo Complete ===")
        logger.info("All EMG compression features demonstrated successfully!")
        logger.info("Check the 'demo_results' directory for output files and visualizations.")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
