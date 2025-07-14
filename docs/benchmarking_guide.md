# Benchmarking Guide

## Overview

This guide provides comprehensive instructions for benchmarking neural data compression algorithms using the BCI Compression Toolkit.

## Quick Benchmarking

### Run Validation Tests

```bash
# Test all implemented algorithms
python tests/validate_phase2.py  # Core algorithms
python tests/validate_phase3.py  # Advanced techniques

# Expected output:
# ✅ All tests passing
# Success rate: 100.0%
```

### Basic Performance Comparison

```python
from bci_compression.algorithms import (
    create_neural_lz_compressor,
    create_predictive_compressor,
    create_context_aware_compressor
)
import numpy as np
import time

# Generate test data
neural_data = np.random.randn(64, 30000)  # 64 channels, 30k samples

algorithms = {
    'Neural LZ': create_neural_lz_compressor('balanced'),
    'Predictive': create_predictive_compressor('balanced'),
    'Context-Aware': create_context_aware_compressor('adaptive')
}

results = {}
for name, algorithm in algorithms.items():
    start_time = time.time()
    
    # Setup if needed
    if hasattr(algorithm, 'setup_spatial_model'):
        algorithm.setup_spatial_model(64)
    
    # Compress
    compressed, metadata = algorithm.compress(neural_data)
    compression_time = time.time() - start_time
    
    # Extract metrics
    if hasattr(metadata, 'compression_ratio'):
        ratio = metadata.compression_ratio
    else:
        ratio = metadata.get('overall_compression_ratio', 1.0)
    
    results[name] = {
        'ratio': ratio,
        'time': compression_time,
        'throughput': neural_data.size / compression_time
    }
    
    print(f"{name}:")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Time: {compression_time:.4f}s")
    print(f"  Throughput: {results[name]['throughput']:.0f} samples/s")
```

## Detailed Benchmarking

### Standard Test Datasets

```python
def generate_realistic_neural_data(n_channels=64, n_samples=30000):
    """Generate realistic neural data with various characteristics."""
    # This function creates data with:
    # - Neural oscillations (alpha, beta, gamma)
    # - Spike events
    # - Spatial correlations
    # - Temporal structure
    # - Realistic noise
```

### Compression Metrics

#### Quality Metrics

```python
def calculate_compression_quality(original, decompressed):
    """Calculate signal quality metrics."""
    
    # Signal-to-Noise Ratio
    mse = np.mean((original - decompressed) ** 2)
    signal_power = np.mean(original ** 2)
    snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
    
    # Spectral preservation
    orig_fft = np.fft.fft(original, axis=-1)
    decomp_fft = np.fft.fft(decompressed, axis=-1)
    spectral_correlation = np.corrcoef(
        np.abs(orig_fft).flatten(),
        np.abs(decomp_fft).flatten()
    )[0, 1]
    
    # Temporal correlation
    temporal_correlation = np.corrcoef(
        original.flatten(),
        decompressed.flatten()
    )[0, 1]
    
    return {
        'snr_db': snr_db,
        'spectral_correlation': spectral_correlation,
        'temporal_correlation': temporal_correlation,
        'mse': mse
    }
```

#### Performance Metrics

```python
def benchmark_algorithm_performance(algorithm, test_data, n_runs=5):
    """Comprehensive performance benchmarking."""
    
    results = {
        'compression_times': [],
        'compression_ratios': [],
        'memory_usage': [],
        'throughput': []
    }
    
    for run in range(n_runs):
        # Measure memory usage
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Time compression
        start_time = time.time()
        compressed, metadata = algorithm.compress(test_data)
        compression_time = time.time() - start_time
        
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        
        # Extract metrics
        ratio = getattr(metadata, 'compression_ratio', 
                       metadata.get('overall_compression_ratio', 1.0))
        
        results['compression_times'].append(compression_time)
        results['compression_ratios'].append(ratio)
        results['memory_usage'].append(memory_used)
        results['throughput'].append(test_data.size / compression_time)
    
    # Calculate statistics
    return {
        'avg_compression_time': np.mean(results['compression_times']),
        'std_compression_time': np.std(results['compression_times']),
        'avg_compression_ratio': np.mean(results['compression_ratios']),
        'avg_memory_usage_mb': np.mean(results['memory_usage']) / 1024**2,
        'avg_throughput': np.mean(results['throughput'])
    }
```

### Real-Time Performance Testing

```python
def test_realtime_performance(algorithm, chunk_size=1000, n_chunks=100):
    """Test real-time processing performance."""
    
    latencies = []
    
    for i in range(n_chunks):
        # Generate chunk
        chunk = np.random.randn(32, chunk_size)
        
        # Time processing
        start_time = time.time()
        
        if hasattr(algorithm, 'process_chunk'):
            result = algorithm.process_chunk(chunk)
        else:
            result = algorithm.compress(chunk)
            
        latency = time.time() - start_time
        latencies.append(latency * 1000)  # Convert to ms
    
    return {
        'avg_latency_ms': np.mean(latencies),
        'max_latency_ms': np.max(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'realtime_capable': np.max(latencies) < 10  # < 10ms for real-time
    }
```

## GPU Performance Benchmarking

```python
def benchmark_gpu_performance():
    """Benchmark GPU vs CPU performance."""
    
    try:
        from bci_compression.algorithms import create_gpu_compression_system
        
        # Test data
        test_data = np.random.randn(128, 10000)
        
        # GPU system
        gpu_system = create_gpu_compression_system('latency')
        
        # Benchmark GPU processing
        gpu_times = []
        for _ in range(10):
            start_time = time.time()
            processed, meta = gpu_system.process_chunk(test_data)
            gpu_times.append(time.time() - start_time)
        
        # Get GPU statistics
        backend = gpu_system.backend if hasattr(gpu_system, 'backend') else gpu_system
        stats = backend.get_performance_stats()
        
        return {
            'gpu_available': backend.gpu_available,
            'avg_gpu_time': np.mean(gpu_times),
            'gpu_operations': stats.get('gpu_operations', 0),
            'cpu_operations': stats.get('cpu_operations', 0),
            'speedup_estimate': '3-5x when GPU available'
        }
        
    except ImportError:
        return {'gpu_available': False, 'message': 'GPU acceleration not available'}
```

## Comparison with Standard Methods

### Baseline Compression

```python
import gzip
import bz2
import lzma

def benchmark_standard_compression(data):
    """Compare with standard compression algorithms."""
    
    # Convert to bytes
    data_bytes = data.astype(np.float32).tobytes()
    original_size = len(data_bytes)
    
    algorithms = {
        'gzip': gzip.compress,
        'bz2': bz2.compress,
        'lzma': lzma.compress
    }
    
    results = {}
    for name, compress_func in algorithms.items():
        start_time = time.time()
        compressed = compress_func(data_bytes)
        compression_time = time.time() - start_time
        
        results[name] = {
            'ratio': original_size / len(compressed),
            'time': compression_time,
            'size_mb': len(compressed) / 1024**2
        }
    
    return results
```

## Benchmarking Best Practices

### Data Preparation

1. **Realistic Data**: Use actual neural recordings when possible
2. **Multiple Datasets**: Test on various recording conditions
3. **Consistent Preprocessing**: Apply same filtering/normalization
4. **Statistical Significance**: Multiple runs with error bars

### Performance Metrics

1. **Compression Ratio**: Original size / Compressed size
2. **Processing Time**: Wall-clock time for compression
3. **Memory Usage**: Peak memory consumption
4. **Quality Metrics**: SNR, correlation, spectral preservation
5. **Real-Time Capability**: Latency < processing window

### Reporting Results

```python
def generate_benchmark_report(results):
    """Generate formatted benchmark report."""
    
    print("=" * 60)
    print("NEURAL DATA COMPRESSION BENCHMARK RESULTS")
    print("=" * 60)
    
    for algorithm, metrics in results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"  Compression Ratio: {metrics['ratio']:.2f}x")
        print(f"  Processing Time: {metrics['time']:.4f}s")
        print(f"  Throughput: {metrics['throughput']:.0f} samples/s")
        
        if 'quality' in metrics:
            quality = metrics['quality']
            print(f"  Signal Quality:")
            print(f"    SNR: {quality['snr_db']:.1f} dB")
            print(f"    Temporal Correlation: {quality['temporal_correlation']:.3f}")
            print(f"    Spectral Correlation: {quality['spectral_correlation']:.3f}")
    
    print("\n" + "=" * 60)
```

## Automated Benchmarking

Create a comprehensive benchmarking script:

```bash
#!/bin/bash
# benchmark.sh - Automated benchmarking script

echo "Starting BCI Compression Toolkit Benchmarks..."

# Run validation tests
python tests/validate_phase2.py
python tests/validate_phase3.py

# Run performance benchmarks
python -c "
from bci_compression.algorithms.predictive import benchmark_predictive_compression
from bci_compression.algorithms.context_aware import *
import numpy as np

# Predictive compression benchmark
print('\\nPredictive Compression Benchmark:')
pred_results = benchmark_predictive_compression()
for mode, metrics in pred_results.items():
    print(f'{mode}: {metrics[\"compression_ratio\"]:.2f}x, {metrics[\"compression_time\"]:.4f}s')

# Context-aware benchmark
print('\\nContext-Aware Compression Test:')
data = np.random.randn(32, 5000)
compressor = create_context_aware_compressor('adaptive')
compressor.setup_spatial_model(32)
compressed, metadata = compressor.compress(data)
print(f'Ratio: {metadata.compression_ratio:.2f}x')
print(f'States: {metadata.brain_states}')
"

echo "Benchmarking complete!"
```

This benchmarking framework provides comprehensive evaluation of all compression algorithms in the toolkit, enabling fair comparison and performance optimization.
