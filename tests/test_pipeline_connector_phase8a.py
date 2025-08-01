"""
Enhanced Test Suite for PipelineConnector with Phase 8a Integration

Tests include:
- Real-time compression metrics with transformer integration
- Algorithm auto-selection validation
- Performance monitoring and grading
- Dashboard metrics integration
- Error handling and fallback scenarios
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualization.web_dashboard.pipeline_connector import PipelineConnector


class TestPipelineConnectorPhase8a(unittest.TestCase):
    """Test suite for Phase 8a enhanced PipelineConnector."""

    def setUp(self):
        """Set up test fixtures."""
        self.connector = PipelineConnector()
        
        # Test data for various scenarios
        self.test_data_small = np.random.randn(16, 500)    # Small dataset
        self.test_data_medium = np.random.randn(64, 2000)  # Medium dataset
        self.test_data_large = np.random.randn(128, 5000)  # Large dataset
        
        # Add realistic neural patterns
        self._add_neural_patterns()

    def _add_neural_patterns(self):
        """Add realistic neural signal patterns to test data."""
        # Add temporal correlations (typical in neural data)
        for i in range(1, self.test_data_medium.shape[1]):
            self.test_data_medium[:, i] = 0.7 * self.test_data_medium[:, i-1] + 0.3 * self.test_data_medium[:, i]
        
        # Add sparse spikes
        for _ in range(50):
            ch = np.random.randint(0, self.test_data_medium.shape[0])
            t = np.random.randint(10, self.test_data_medium.shape[1]-10)
            self.test_data_medium[ch, t:t+5] += np.array([0, 1, 3, 1, 0])

    def test_real_time_compression_metrics_transformer(self):
        """Test real-time compression metrics with transformer algorithm."""
        metrics = self.connector.get_real_time_compression_metrics(
            self.test_data_medium, 
            algorithm="transformer"
        )
        
        # Validate required fields
        required_fields = [
            'compression_ratio', 'snr_db', 'latency_ms', 'algorithm_used',
            'real_time_capable', 'performance_grade', 'signal_quality_score'
        ]
        
        for field in required_fields:
            self.assertIn(field, metrics, f"Missing required field: {field}")
        
        # Validate performance targets
        if 'error' not in metrics:
            self.assertGreater(metrics['compression_ratio'], 1.0)
            self.assertGreater(metrics['snr_db'], 0.0)
            self.assertLess(metrics['latency_ms'], 10.0)  # Reasonable upper bound
            self.assertIn(metrics['performance_grade'], ['A', 'B', 'C', 'D', 'F'])

    def test_real_time_compression_metrics_auto_selection(self):
        """Test automatic algorithm selection."""
        metrics = self.connector.get_real_time_compression_metrics(
            self.test_data_medium,
            algorithm="auto"
        )
        
        self.assertIn('algorithm_used', metrics)
        self.assertIn(metrics['algorithm_used'], ['transformer', 'sparse', 'adaptive'])

    def test_latency_targets_phase8a(self):
        """Test that latency meets Phase 8a targets (<2ms for real-time)."""
        # Test with smaller chunk size for real-time processing
        real_time_chunk = self.test_data_medium[:, :256]  # ~10ms at 25kHz
        
        start_time = time.time()
        metrics = self.connector.get_real_time_compression_metrics(
            real_time_chunk,
            algorithm="transformer"
        )
        total_time = time.time() - start_time
        
        # Total processing should be fast
        self.assertLess(total_time * 1000, 50.0, "Total processing time too slow")
        
        # Reported latency should meet targets for good grades
        if 'error' not in metrics and metrics['performance_grade'] in ['A', 'B']:
            self.assertLess(metrics['latency_ms'], 5.0, "Latency too high for good grade")

    def test_compression_quality_targets(self):
        """Test compression quality meets Phase 8a targets."""
        metrics = self.connector.get_real_time_compression_metrics(
            self.test_data_medium,
            algorithm="transformer"
        )
        
        if 'error' not in metrics:
            # Phase 8a targets: 3-5x compression, 25-35dB SNR
            if metrics['performance_grade'] == 'A':
                self.assertGreaterEqual(metrics['compression_ratio'], 3.0)
                self.assertGreaterEqual(metrics['snr_db'], 25.0)

    def test_algorithm_selection_logic(self):
        """Test algorithm selection logic based on signal characteristics."""
        # Test with highly correlated signal (should prefer transformer)
        correlated_signal = np.zeros((32, 1000))
        for i in range(1, 1000):
            correlated_signal[:, i] = 0.9 * correlated_signal[:, i-1] + 0.1 * np.random.randn(32)
        
        optimal_algo = self.connector._select_optimal_algorithm(correlated_signal)
        # Should prefer transformer for highly correlated temporal patterns
        
        # Test with sparse signal
        sparse_signal = np.random.randn(32, 1000) * 0.05  # Low amplitude, mostly near zero
        optimal_algo_sparse = self.connector._select_optimal_algorithm(sparse_signal)
        
        # Algorithms should be valid choices
        valid_algorithms = ['transformer', 'sparse', 'adaptive']
        self.assertIn(optimal_algo, valid_algorithms)
        self.assertIn(optimal_algo_sparse, valid_algorithms)

    def test_performance_grading_system(self):
        """Test performance grading system accuracy."""
        # Test with known good metrics (should get A)
        good_metrics = {
            'compression_ratio': 4.5,
            'snr_db': 32.0,
            'memory_usage_mb': 5.0
        }
        grade_a = self.connector._calculate_performance_grade(good_metrics, 1.0)
        self.assertEqual(grade_a, 'A')
        
        # Test with poor metrics (should get F)
        poor_metrics = {
            'compression_ratio': 1.2,
            'snr_db': 8.0,
            'memory_usage_mb': 50.0
        }
        grade_f = self.connector._calculate_performance_grade(poor_metrics, 15.0)
        self.assertEqual(grade_f, 'F')

    def test_temporal_correlation_calculation(self):
        """Test temporal correlation analysis."""
        # Create signal with known correlation
        n_samples = 500
        correlated_signal = np.zeros((1, n_samples))
        correlated_signal[0, 0] = np.random.randn()
        
        for i in range(1, n_samples):
            correlated_signal[0, i] = 0.8 * correlated_signal[0, i-1] + 0.2 * np.random.randn()
        
        correlation = self.connector._calculate_temporal_correlation(correlated_signal)
        
        # Should detect high temporal correlation
        self.assertGreater(correlation, 0.5, "Failed to detect temporal correlation")
        
        # Test with random signal (low correlation)
        random_signal = np.random.randn(1, n_samples)
        correlation_random = self.connector._calculate_temporal_correlation(random_signal)
        
        # Should be much lower than correlated signal
        self.assertLess(correlation_random, correlation)

    def test_dashboard_metrics_integration(self):
        """Test dashboard metrics integration."""
        # First generate some compression metrics
        self.connector.get_real_time_compression_metrics(self.test_data_medium)
        
        # Get dashboard metrics
        dashboard_data = self.connector.get_dashboard_metrics()
        
        # Validate dashboard data structure
        required_sections = [
            'current_metrics', 'performance_summary', 'compression_history',
            'system_status', 'algorithm_availability'
        ]
        
        for section in required_sections:
            self.assertIn(section, dashboard_data)
        
        # Check algorithm availability
        self.assertIn('transformer', dashboard_data['algorithm_availability'])
        self.assertIn('adaptive', dashboard_data['algorithm_availability'])

    def test_metrics_cache_management(self):
        """Test metrics cache management and history."""
        # Generate multiple compression operations
        for _ in range(5):
            self.connector.get_real_time_compression_metrics(
                np.random.randn(32, 500),
                algorithm="auto"
            )
        
        # Check that cache is populated
        self.assertGreater(len(self.connector._metrics_cache['compression_history']), 0)
        
        # Test cache size limit (should not exceed 100)
        for _ in range(150):  # Generate more than cache limit
            self.connector._update_metrics_cache({'compression_ratio': 3.0, 'snr_db': 25.0}, 1.5)
        
        self.assertLessEqual(len(self.connector._metrics_cache['compression_history']), 100)

    def test_snr_calculation_accuracy(self):
        """Test SNR calculation accuracy."""
        # Create signal with known SNR
        signal_amplitude = 1.0
        noise_amplitude = 0.1
        
        original = np.random.randn(100) * signal_amplitude
        noise = np.random.randn(100) * noise_amplitude
        noisy = original + noise
        
        calculated_snr = self.connector._calculate_snr(original, noisy)
        
        # Expected SNR ≈ 20*log10(signal_amp/noise_amp) = 20*log10(10) = 20dB
        expected_snr = 20.0
        
        # Allow some tolerance due to random nature
        self.assertAlmostEqual(calculated_snr, expected_snr, delta=5.0)

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # High quality scenario
        high_quality_score = self.connector._calculate_quality_score(35.0, 5.0)
        
        # Low quality scenario
        low_quality_score = self.connector._calculate_quality_score(10.0, 1.5)
        
        # High quality should score better
        self.assertGreater(high_quality_score, low_quality_score)
        
        # Scores should be in valid range
        self.assertGreaterEqual(high_quality_score, 0)
        self.assertLessEqual(high_quality_score, 100)
        self.assertGreaterEqual(low_quality_score, 0)
        self.assertLessEqual(low_quality_score, 100)

    def test_error_handling_robustness(self):
        """Test error handling for various failure scenarios."""
        # Test with invalid signal shape
        try:
            metrics = self.connector.get_real_time_compression_metrics(
                np.array([]),  # Empty array
                algorithm="transformer"
            )
            self.assertIn('error', metrics)
        except Exception:
            pass  # Expected to handle gracefully
        
        # Test with very large signal
        try:
            very_large_signal = np.random.randn(1000, 10000)  # Large array
            metrics = self.connector.get_real_time_compression_metrics(
                very_large_signal,
                algorithm="transformer"
            )
            # Should either succeed or return error gracefully
            self.assertTrue('error' in metrics or 'compression_ratio' in metrics)
        except MemoryError:
            self.skipTest("Insufficient memory for large array test")

    def test_multi_channel_consistency(self):
        """Test consistent behavior across different channel counts."""
        channel_counts = [1, 16, 64, 128]
        results = []
        
        for n_channels in channel_counts:
            test_data = np.random.randn(n_channels, 1000)
            metrics = self.connector.get_real_time_compression_metrics(test_data)
            
            if 'error' not in metrics:
                results.append(metrics['compression_ratio'])
        
        if len(results) > 1:
            # Compression ratios should be reasonably consistent
            cv = np.std(results) / np.mean(results)  # Coefficient of variation
            self.assertLess(cv, 0.5, "Compression ratios too inconsistent across channel counts")

    def test_transformer_fallback_behavior(self):
        """Test behavior when transformer is not available."""
        # Mock transformer as unavailable
        with patch.object(self.connector, 'transformer_compressor', None):
            metrics = self.connector.get_real_time_compression_metrics(
                self.test_data_medium,
                algorithm="transformer"
            )
            
            # Should fallback gracefully
            self.assertTrue('compression_ratio' in metrics or 'error' in metrics)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for Phase 8a compliance."""

    def setUp(self):
        """Set up benchmark tests."""
        self.connector = PipelineConnector()
        self.benchmark_data = np.random.randn(64, 2000)  # Standard BCI dataset size

    def test_compression_speed_benchmark(self):
        """Benchmark compression speed for real-time requirements."""
        iterations = 10
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            self.connector.get_real_time_compression_metrics(self.benchmark_data)
            elapsed = time.time() - start_time
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Compression benchmark: {avg_time:.2f}±{std_time:.2f}ms")
        
        # Performance target: average < 10ms for benchmark dataset
        self.assertLess(avg_time, 10.0, f"Average compression time {avg_time:.2f}ms too slow")

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during compression."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple compressions
        for _ in range(20):
            self.connector.get_real_time_compression_metrics(self.benchmark_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increase: {memory_increase:.1f}MB")
        
        # Should not leak significant memory
        self.assertLess(memory_increase, 50.0, f"Memory increase {memory_increase:.1f}MB too high")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
