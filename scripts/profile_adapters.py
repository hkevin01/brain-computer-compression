"""
Adapter Performance Profiling

Profiles the overhead and performance of BCI device adapters:
- Mapping overhead
- Resampling performance
- Channel grouping efficiency
- Memory usage
- Compression pipeline throughput
"""

import numpy as np
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bci_compression.adapters.openbci import OpenBCIAdapter
from bci_compression.adapters.blackrock import BlackrockAdapter
from bci_compression.adapters.intan import IntanAdapter
from bci_compression.adapters.hdf5 import HDF5Adapter
from bci_compression.adapters import map_channels, resample, apply_channel_groups
from bci_compression.algorithms.lossless import NeuralLZ77Compressor


@dataclass
class ProfileResult:
    """Performance profiling result."""
    operation: str
    device: str
    time_ms: float
    throughput_samples_per_sec: float
    memory_mb: float


class AdapterProfiler:
    """Profile adapter performance."""
    
    def __init__(self):
        """Initialize profiler."""
        self.results: List[ProfileResult] = []
    
    def profile_mapping(
        self,
        adapter,
        data: np.ndarray,
        n_runs: int = 100
    ) -> ProfileResult:
        """
        Profile channel mapping overhead.
        
        Args:
            adapter: Device adapter
            data: Input data
            n_runs: Number of runs for averaging
        
        Returns:
            ProfileResult with timing information
        """
        device_name = adapter.__class__.__name__.replace('Adapter', '')
        
        # Warmup
        for _ in range(10):
            _ = map_channels(data, adapter.mapping)
        
        # Profile
        start = time.time()
        for _ in range(n_runs):
            _ = map_channels(data, adapter.mapping)
        elapsed = (time.time() - start) / n_runs * 1000  # ms
        
        throughput = data.shape[1] / (elapsed / 1000)  # samples/sec
        memory = data.nbytes / 1024 / 1024  # MB
        
        result = ProfileResult(
            operation='mapping',
            device=device_name,
            time_ms=elapsed,
            throughput_samples_per_sec=throughput,
            memory_mb=memory
        )
        self.results.append(result)
        return result
    
    def profile_resampling(
        self,
        adapter,
        data: np.ndarray,
        target_rate: int,
        n_runs: int = 10
    ) -> ProfileResult:
        """
        Profile resampling performance.
        
        Args:
            adapter: Device adapter
            data: Input data
            target_rate: Target sampling rate
            n_runs: Number of runs for averaging
        
        Returns:
            ProfileResult with timing information
        """
        device_name = adapter.__class__.__name__.replace('Adapter', '')
        
        # Warmup
        for _ in range(3):
            _ = resample(data, adapter.mapping['sampling_rate'], target_rate)
        
        # Profile
        start = time.time()
        for _ in range(n_runs):
            _ = resample(data, adapter.mapping['sampling_rate'], target_rate)
        elapsed = (time.time() - start) / n_runs * 1000  # ms
        
        throughput = data.shape[1] / (elapsed / 1000)
        memory = data.nbytes / 1024 / 1024
        
        result = ProfileResult(
            operation=f'resample_{target_rate}Hz',
            device=device_name,
            time_ms=elapsed,
            throughput_samples_per_sec=throughput,
            memory_mb=memory
        )
        self.results.append(result)
        return result
    
    def profile_channel_groups(
        self,
        adapter,
        data: np.ndarray,
        n_runs: int = 100
    ) -> ProfileResult:
        """
        Profile channel grouping.
        
        Args:
            adapter: Device adapter
            data: Input data
            n_runs: Number of runs
        
        Returns:
            ProfileResult
        """
        device_name = adapter.__class__.__name__.replace('Adapter', '')
        
        # Warmup
        for _ in range(10):
            _ = adapter.get_channel_groups()
        
        # Profile
        start = time.time()
        for _ in range(n_runs):
            groups = adapter.get_channel_groups()
            for group_name, indices in groups.items():
                _ = data[indices, :]
        elapsed = (time.time() - start) / n_runs * 1000
        
        throughput = data.shape[1] / (elapsed / 1000)
        memory = data.nbytes / 1024 / 1024
        
        result = ProfileResult(
            operation='channel_groups',
            device=device_name,
            time_ms=elapsed,
            throughput_samples_per_sec=throughput,
            memory_mb=memory
        )
        self.results.append(result)
        return result
    
    def profile_full_pipeline(
        self,
        adapter,
        data: np.ndarray,
        n_runs: int = 10
    ) -> ProfileResult:
        """
        Profile full adapter + compression pipeline.
        
        Args:
            adapter: Device adapter
            data: Input data
            n_runs: Number of runs
        
        Returns:
            ProfileResult
        """
        device_name = adapter.__class__.__name__.replace('Adapter', '')
        compressor = NeuralLZ77Compressor()
        
        # Warmup
        for _ in range(3):
            converted = adapter.convert(data, apply_mapping=False)
            _ = compressor.compress(converted)
        
        # Profile
        start = time.time()
        for _ in range(n_runs):
            converted = adapter.convert(data, apply_mapping=False)
            _ = compressor.compress(converted)
        elapsed = (time.time() - start) / n_runs * 1000
        
        throughput = data.shape[1] / (elapsed / 1000)
        memory = data.nbytes / 1024 / 1024
        
        result = ProfileResult(
            operation='full_pipeline',
            device=device_name,
            time_ms=elapsed,
            throughput_samples_per_sec=throughput,
            memory_mb=memory
        )
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print profiling summary."""
        if not self.results:
            print("No profiling results available.")
            return
        
        print("\n" + "="*80)
        print("Adapter Performance Profiling Summary")
        print("="*80)
        print()
        
        # Group by device
        devices = set(r.device for r in self.results)
        
        for device in sorted(devices):
            device_results = [r for r in self.results if r.device == device]
            
            print(f"{device} Adapter:")
            print(f"  {'Operation':<20} {'Time (ms)':<12} {'Throughput (k samples/s)':<25} {'Memory (MB)':<12}")
            print(f"  {'-'*20} {'-'*12} {'-'*25} {'-'*12}")
            
            for result in device_results:
                throughput_k = result.throughput_samples_per_sec / 1000
                print(f"  {result.operation:<20} {result.time_ms:>10.3f}  {throughput_k:>23.1f}  {result.memory_mb:>10.2f}")
            print()
    
    def save_report(self, filepath: str):
        """Save profiling report to file."""
        with open(filepath, 'w') as f:
            f.write("Adapter Performance Profiling Report\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                f.write(f"Device: {result.device}\n")
                f.write(f"Operation: {result.operation}\n")
                f.write(f"Time: {result.time_ms:.3f} ms\n")
                f.write(f"Throughput: {result.throughput_samples_per_sec:,.0f} samples/sec\n")
                f.write(f"Memory: {result.memory_mb:.2f} MB\n")
                f.write("-"*80 + "\n")


def benchmark_all_adapters():
    """Run comprehensive benchmark of all adapters."""
    print("="*80)
    print("Comprehensive Adapter Benchmarking")
    print("="*80)
    print()
    
    profiler = AdapterProfiler()
    
    # Test configurations
    configs = [
        {
            'name': 'OpenBCI Cyton',
            'adapter': OpenBCIAdapter(device='cyton_8ch'),
            'data': np.random.randn(8, 10000),  # 40 seconds @ 250Hz
            'resample_rate': 1000
        },
        {
            'name': 'Blackrock Neuroport',
            'adapter': BlackrockAdapter(device='neuroport_96ch'),
            'data': np.random.randn(96, 30000),  # 1 second @ 30kHz
            'resample_rate': 1000
        },
        {
            'name': 'Intan RHD2164',
            'adapter': IntanAdapter(device='rhd2164_64ch'),
            'data': np.random.randn(64, 20000),  # 1 second @ 20kHz
            'resample_rate': 1000
        }
    ]
    
    for config in configs:
        print(f"Profiling {config['name']}...")
        
        # Mapping
        result = profiler.profile_mapping(config['adapter'], config['data'])
        print(f"  Mapping: {result.time_ms:.3f}ms ({result.throughput_samples_per_sec/1000:.1f}k samples/s)")
        
        # Resampling
        result = profiler.profile_resampling(config['adapter'], config['data'], config['resample_rate'])
        print(f"  Resampling: {result.time_ms:.3f}ms ({result.throughput_samples_per_sec/1000:.1f}k samples/s)")
        
        # Channel groups
        result = profiler.profile_channel_groups(config['adapter'], config['data'])
        print(f"  Channel groups: {result.time_ms:.3f}ms ({result.throughput_samples_per_sec/1000:.1f}k samples/s)")
        
        # Full pipeline
        result = profiler.profile_full_pipeline(config['adapter'], config['data'])
        print(f"  Full pipeline: {result.time_ms:.3f}ms ({result.throughput_samples_per_sec/1000:.1f}k samples/s)")
        print()
    
    profiler.print_summary()
    
    # Save report
    report_path = Path(__file__).parent.parent / 'results' / 'adapter_profiling_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    profiler.save_report(str(report_path))
    print(f"Report saved to: {report_path}")


def profile_hot_paths():
    """Profile hot paths using cProfile."""
    print("\n" + "="*80)
    print("Hot Path Profiling (cProfile)")
    print("="*80)
    print()
    
    # Setup
    adapter = OpenBCIAdapter(device='cyton_8ch')
    data = np.random.randn(8, 10000)
    compressor = NeuralLZ77Compressor()
    
    # Profile with cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    # Run workload
    for _ in range(100):
        converted = adapter.convert(data, apply_mapping=False)
        _ = compressor.compress(converted)
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Top 20 functions by cumulative time:")
    print(s.getvalue())


def benchmark_memory_usage():
    """Benchmark memory usage for different data sizes."""
    print("\n" + "="*80)
    print("Memory Usage Benchmarking")
    print("="*80)
    print()
    
    adapter = BlackrockAdapter(device='neuroport_96ch')
    
    # Test different data sizes
    sizes = [1000, 5000, 10000, 30000, 60000]  # samples
    
    print(f"{'Samples':<10} {'Channels':<10} {'Input (MB)':<12} {'Output (MB)':<12} {'Ratio':<10}")
    print("-"*60)
    
    for n_samples in sizes:
        data = np.random.randn(96, n_samples)
        input_mb = data.nbytes / 1024 / 1024
        
        # Measure output size
        converted = adapter.convert(data, apply_mapping=False)
        output_mb = converted.nbytes / 1024 / 1024
        
        ratio = output_mb / input_mb
        
        print(f"{n_samples:<10} {96:<10} {input_mb:>10.2f}  {output_mb:>10.2f}  {ratio:>8.2f}x")
    print()


def main():
    """Run all profiling benchmarks."""
    try:
        benchmark_all_adapters()
        profile_hot_paths()
        benchmark_memory_usage()
        
        print("\n" + "="*80)
        print("✓ Profiling complete!")
        print("="*80)
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
