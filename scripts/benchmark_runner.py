#!/usr/bin/env python3
"""
Benchmark runner for compression algorithms.
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from bci_compression import NeuralCompressor, load_neural_data
from bci_compression.data_processing import generate_synthetic_neural_data


def run_benchmark(algorithm: str, data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmark for a single algorithm."""
    compressor = NeuralCompressor(algorithm=algorithm, **config)
    
    # Measure compression time
    start_time = time.time()
    compressed_data = compressor.compress(data)
    compression_time = time.time() - start_time
    
    # Measure decompression time
    start_time = time.time()
    decompressed_data = compressor.decompress(compressed_data)
    decompression_time = time.time() - start_time
    
    # Calculate metrics
    original_size = data.nbytes
    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size
    
    # Calculate quality metrics (simplified)
    mse = np.mean((data - decompressed_data) ** 2)
    snr = 10 * np.log10(np.var(data) / mse) if mse > 0 else float('inf')
    
    return {
        'algorithm': algorithm,
        'compression_ratio': compression_ratio,
        'compression_time': compression_time,
        'decompression_time': decompression_time,
        'total_time': compression_time + decompression_time,
        'throughput_mbps': (original_size / 1e6) / (compression_time + decompression_time),
        'snr_db': snr,
        'mse': mse,
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size
    }


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description='Run compression benchmarks')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['adaptive_lz', 'neural_quantization', 'wavelet_transform'],
                       help='Algorithms to benchmark')
    parser.add_argument('--data-file', type=str, help='Path to neural data file')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic data instead of file')
    parser.add_argument('--channels', type=int, default=64, 
                       help='Number of channels for synthetic data')
    parser.add_argument('--samples', type=int, default=30000,
                       help='Number of samples for synthetic data')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per algorithm')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.synthetic or args.data_file is None:
        print(f"Generating synthetic data: {args.channels} channels, {args.samples} samples")
        data, metadata = generate_synthetic_neural_data(
            n_channels=args.channels,
            n_samples=args.samples,
            seed=42
        )
        data_info = metadata
    else:
        print(f"Loading data from: {args.data_file}")
        data = load_neural_data(args.data_file)
        data_info = {
            'source': args.data_file,
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    
    print(f"Data shape: {data.shape}")
    print(f"Data size: {data.nbytes / 1e6:.2f} MB")
    
    # Run benchmarks
    all_results = []
    
    for algorithm in args.algorithms:
        print(f"\nBenchmarking {algorithm}...")
        
        algorithm_results = []
        for run in range(args.runs):
            try:
                result = run_benchmark(algorithm, data, {})
                algorithm_results.append(result)
                print(f"  Run {run + 1}: {result['compression_ratio']:.2f}:1, "
                      f"{result['throughput_mbps']:.2f} MB/s")
            except Exception as e:
                print(f"  Run {run + 1} failed: {e}")
        
        # Calculate average results
        if algorithm_results:
            avg_result = {
                'algorithm': algorithm,
                'runs': len(algorithm_results),
                'avg_compression_ratio': np.mean([r['compression_ratio'] for r in algorithm_results]),
                'avg_compression_time': np.mean([r['compression_time'] for r in algorithm_results]),
                'avg_decompression_time': np.mean([r['decompression_time'] for r in algorithm_results]),
                'avg_throughput_mbps': np.mean([r['throughput_mbps'] for r in algorithm_results]),
                'avg_snr_db': np.mean([r['snr_db'] for r in algorithm_results]),
                'std_compression_ratio': np.std([r['compression_ratio'] for r in algorithm_results]),
                'individual_runs': algorithm_results
            }
            all_results.append(avg_result)
    
    # Save results
    output_data = {
        'data_info': data_info,
        'benchmark_config': {
            'algorithms': args.algorithms,
            'runs_per_algorithm': args.runs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nBenchmark results saved to: {args.output}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Ratio':<10} {'Time (ms)':<12} {'Throughput':<12} {'SNR (dB)':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['algorithm']:<20} "
              f"{result['avg_compression_ratio']:<10.2f} "
              f"{result['avg_compression_time'] * 1000:<12.2f} "
              f"{result['avg_throughput_mbps']:<12.2f} "
              f"{result['avg_snr_db']:<10.2f}")


if __name__ == '__main__':
    main()
