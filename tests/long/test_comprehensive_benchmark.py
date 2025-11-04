"""
Comprehensive Benchmark Test Suite

Full-scale benchmarking with large datasets. These tests are excluded
from quick test runs.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.mark.slow
class TestComprehensiveBenchmark:
    """Comprehensive benchmarking tests."""
    
    def test_full_benchmark_suite(self):
        """Run full benchmark suite with large datasets."""
        # Import the performance benchmark
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from test_performance_benchmark import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark()
        
        # Run all benchmarks
        neural_results = benchmark.benchmark_neural_algorithms()
        emg_results = benchmark.benchmark_emg_algorithms()
        scalability_results = benchmark.benchmark_scalability()
        
        # Validate results
        assert neural_results is not None
        assert emg_results is not None
        assert scalability_results is not None
        
    def test_stress_test(self):
        """Stress test with very large datasets."""
        import numpy as np
        from bci_compression.algorithms import create_neural_lz_compressor
        
        # Large dataset: 128 channels, 100k samples
        large_data = np.random.randn(128, 100000).astype(np.float32)
        
        compressor = create_neural_lz_compressor('speed')
        compressed, metadata = compressor.compress(large_data)
        
        # Verify compression succeeded
        compression_ratio = metadata.get('overall_compression_ratio', 0)
        assert compression_ratio > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
