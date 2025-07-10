#!/usr/bin/env python3
"""
Performance profiler for neural data compression algorithms.

This script provides detailed performance analysis including CPU usage,
memory consumption, and GPU utilization for compression algorithms.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_compression import NeuralCompressor


class PerformanceProfiler:
    """Detailed performance profiling for compression algorithms."""
    
    def __init__(self, output_dir: str = "results/profiling/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_data = []
    
    def profile_compression(
        self,
        algorithm_name: str,
        compressor,
        data: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Profile compression performance in detail."""
        print(f"Profiling {algorithm_name}...")
        
        profile_result = {
            'algorithm': algorithm_name,
            'data_shape': data.shape,
            'data_size_mb': data.nbytes / 1024 / 1024,
            'timestamp': time.time()
        }
        
        # System information
        profile_result['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        if detailed:
            # Detailed profiling with monitoring
            profile_result.update(self._detailed_profiling(compressor, data))
        else:
            # Basic timing only
            profile_result.update(self._basic_profiling(compressor, data))
        
        return profile_result
    
    def _basic_profiling(self, compressor, data: np.ndarray) -> Dict[str, Any]:
        """Basic timing profiling."""
        # Compression
        start_time = time.perf_counter()
        compressed_data = compressor.compress(data)
        compression_time = time.perf_counter() - start_time
        
        # Decompression
        start_time = time.perf_counter()
        reconstructed_data = compressor.decompress(compressed_data)
        decompression_time = time.perf_counter() - start_time
        
        # Metrics
        compression_ratio = compressor.compression_ratio
        reconstruction_error = np.mean((data - reconstructed_data)**2)
        
        return {
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': compression_ratio,
            'reconstruction_error': reconstruction_error,
            'compressed_size_mb': len(compressed_data) / 1024 / 1024
        }
    
    def _detailed_profiling(self, compressor, data: np.ndarray) -> Dict[str, Any]:
        """Detailed profiling with system monitoring."""
        # Monitor system resources during compression
        compression_profile = self._monitor_operation(
            lambda: compressor.compress(data),
            "compression"
        )
        
        compressed_data = compression_profile['result']
        
        # Monitor decompression
        decompression_profile = self._monitor_operation(
            lambda: compressor.decompress(compressed_data),
            "decompression"
        )
        
        reconstructed_data = decompression_profile['result']
        
        # Calculate quality metrics
        compression_ratio = compressor.compression_ratio
        reconstruction_error = np.mean((data - reconstructed_data)**2)
        snr = 10 * np.log10(np.var(data) / (reconstruction_error + 1e-10))
        
        return {
            'compression_time': compression_profile['execution_time'],
            'decompression_time': decompression_profile['execution_time'],
            'compression_ratio': compression_ratio,
            'reconstruction_error': reconstruction_error,
            'snr_db': snr,
            'compressed_size_mb': len(compressed_data) / 1024 / 1024,
            'compression_monitoring': compression_profile['monitoring'],
            'decompression_monitoring': decompression_profile['monitoring']
        }
    
    def _monitor_operation(self, operation, operation_name: str) -> Dict[str, Any]:
        """Monitor system resources during an operation."""
        # Pre-operation measurements
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu_percent = process.cpu_percent()
        
        # Execute operation with monitoring
        monitoring_data = {
            'cpu_percent': [],
            'memory_mb': [],
            'timestamps': []
        }
        
        start_time = time.perf_counter()
        
        # Start monitoring in background (simplified for demo)
        result = operation()
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Post-operation measurements
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu_percent = process.cpu_percent()
        
        # Simplified monitoring data
        monitoring_data.update({
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': final_memory,  # Simplified
            'memory_delta_mb': final_memory - initial_memory,
            'avg_cpu_percent': (initial_cpu_percent + final_cpu_percent) / 2
        })
        
        return {
            'result': result,
            'execution_time': execution_time,
            'monitoring': monitoring_data
        }
    
    def profile_scaling(
        self,
        algorithm_name: str,
        base_data: np.ndarray,
        scale_factors: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> Dict[str, Any]:
        """Profile how algorithm scales with data size."""
        print(f"Profiling scaling behavior for {algorithm_name}...")
        
        scaling_results = {
            'algorithm': algorithm_name,
            'scale_factors': scale_factors,
            'results': []
        }
        
        for scale_factor in scale_factors:
            # Scale data
            if scale_factor <= 1.0:
                # Subsample
                n_samples = int(base_data.shape[1] * scale_factor)
                scaled_data = base_data[:, :n_samples]
            else:
                # Repeat data
                repeats = int(scale_factor)
                scaled_data = np.tile(base_data, (1, repeats))
            
            print(f"  Testing scale factor {scale_factor} (shape: {scaled_data.shape})")
            
            # Profile at this scale
            compressor = NeuralCompressor(algorithm=algorithm_name)
            result = self.profile_compression(
                f"{algorithm_name}_scale_{scale_factor}",
                compressor,
                scaled_data,
                detailed=False
            )
            
            scaling_results['results'].append({
                'scale_factor': scale_factor,
                'data_shape': scaled_data.shape,
                'data_size_mb': result['data_size_mb'],
                'compression_time': result['compression_time'],
                'compression_ratio': result['compression_ratio']
            })
        
        return scaling_results
    
    def compare_algorithms(
        self,
        algorithms: List[str],
        data: np.ndarray
    ) -> Dict[str, Any]:
        """Compare multiple algorithms side by side."""
        print("Running algorithm comparison...")
        
        comparison_results = {
            'algorithms': algorithms,
            'data_shape': data.shape,
            'results': []
        }
        
        for algorithm in algorithms:
            try:
                compressor = NeuralCompressor(algorithm=algorithm)
                result = self.profile_compression(algorithm, compressor, data, detailed=True)
                comparison_results['results'].append(result)
            except Exception as e:
                print(f"Error profiling {algorithm}: {e}")
                continue
        
        return comparison_results
    
    def generate_report(
        self,
        profile_data: Dict[str, Any],
        output_file: str = None
    ) -> str:
        """Generate a comprehensive performance report."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_report_{timestamp}"
        
        # Save raw data
        json_file = self.output_dir / f"{output_file}.json"
        with open(json_file, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        # Generate plots if comparison data available
        if 'results' in profile_data and len(profile_data['results']) > 1:
            self._generate_comparison_plots(profile_data, output_file)
        
        # Generate text report
        report_file = self.output_dir / f"{output_file}.txt"
        with open(report_file, 'w') as f:
            f.write(self._format_text_report(profile_data))
        
        print(f"Performance report saved to: {report_file}")
        return str(report_file)
    
    def _generate_comparison_plots(self, data: Dict[str, Any], filename: str):
        """Generate comparison plots."""
        try:
            results = data['results']
            algorithms = [r['algorithm'] for r in results]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Compression ratio
            ratios = [r['compression_ratio'] for r in results]
            ax1.bar(algorithms, ratios)
            ax1.set_title('Compression Ratio')
            ax1.set_ylabel('Ratio')
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # Compression time
            times = [r['compression_time'] * 1000 for r in results]  # Convert to ms
            ax2.bar(algorithms, times)
            ax2.set_title('Compression Time')
            ax2.set_ylabel('Time (ms)')
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Memory usage (if available)
            if all('compression_monitoring' in r for r in results):
                memory_usage = [r['compression_monitoring']['memory_delta_mb'] for r in results]
                ax3.bar(algorithms, memory_usage)
                ax3.set_title('Memory Usage')
                ax3.set_ylabel('Memory (MB)')
                plt.setp(ax3.get_xticklabels(), rotation=45)
            
            # Reconstruction error
            errors = [r['reconstruction_error'] for r in results]
            ax4.bar(algorithms, errors)
            ax4.set_title('Reconstruction Error')
            ax4.set_ylabel('MSE')
            plt.setp(ax4.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plot_file = self.output_dir / f"{filename}_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison plots saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def _format_text_report(self, data: Dict[str, Any]) -> str:
        """Format a text report."""
        report = ["PERFORMANCE PROFILING REPORT", "=" * 50, ""]
        
        if 'results' in data:
            # Multiple algorithm comparison
            for result in data['results']:
                report.extend(self._format_single_result(result))
                report.append("")
        else:
            # Single algorithm result
            report.extend(self._format_single_result(data))
        
        return "\n".join(report)
    
    def _format_single_result(self, result: Dict[str, Any]) -> List[str]:
        """Format a single algorithm result."""
        lines = [
            f"Algorithm: {result['algorithm']}",
            f"Data Shape: {result['data_shape']}",
            f"Data Size: {result['data_size_mb']:.2f} MB",
            f"Compression Time: {result['compression_time']*1000:.2f} ms",
            f"Compression Ratio: {result['compression_ratio']:.2f}:1",
            f"Reconstruction Error: {result['reconstruction_error']:.6f}"
        ]
        
        if 'snr_db' in result:
            lines.append(f"SNR: {result['snr_db']:.2f} dB")
        
        if 'compression_monitoring' in result:
            monitoring = result['compression_monitoring']
            lines.extend([
                f"Memory Usage: {monitoring['memory_delta_mb']:.2f} MB",
                f"CPU Usage: {monitoring['avg_cpu_percent']:.1f}%"
            ])
        
        lines.append("-" * 40)
        return lines


def main():
    parser = argparse.ArgumentParser(description='Profile compression algorithm performance')
    parser.add_argument('--algorithm', type=str, default='adaptive_lz',
                       help='Algorithm to profile (default: adaptive_lz)')
    parser.add_argument('--algorithms', type=str,
                       help='Comma-separated list of algorithms to compare')
    parser.add_argument('--data', type=str,
                       help='Path to test data file')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scaling analysis')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed profiling with system monitoring')
    parser.add_argument('--output', type=str, default='results/profiling/',
                       help='Output directory (default: results/profiling/)')
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = PerformanceProfiler(args.output)
    
    # Load or generate test data
    if args.data:
        if args.data.endswith('.h5'):
            import h5py
            with h5py.File(args.data, 'r') as f:
                data = f['neural_data'][:]
        else:
            data = np.load(args.data)
    else:
        # Generate synthetic data
        print("Generating synthetic test data...")
        np.random.seed(42)
        data = np.random.randn(64, 300000)  # 64 channels, 10s at 30kHz
    
    print(f"Data shape: {data.shape}")
    print(f"Data size: {data.nbytes / 1024 / 1024:.2f} MB")
    
    # Run profiling based on arguments
    if args.algorithms:
        # Compare multiple algorithms
        algorithms = [alg.strip() for alg in args.algorithms.split(',')]
        result = profiler.compare_algorithms(algorithms, data)
    elif args.scaling:
        # Scaling analysis
        result = profiler.profile_scaling(args.algorithm, data)
    else:
        # Single algorithm profiling
        compressor = NeuralCompressor(algorithm=args.algorithm)
        result = profiler.profile_compression(
            args.algorithm, compressor, data, detailed=args.detailed
        )
    
    # Generate report
    profiler.generate_report(result)


if __name__ == "__main__":
    main()
