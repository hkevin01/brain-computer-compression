#!/usr/bin/env python3
"""
Streaming benchmark for BCI compression toolkit.

Simulates real-time neural data compression with configurable parameters:
- 100-1024 channels at 30-50 kHz sampling rates
- Bounded buffer management
- Tail latency measurement under load
- Cross-backend comparison (CPU vs CUDA vs ROCm)
"""

import argparse
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import statistics
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import psutil

# Import BCC modules
try:
    from bcc.api import compress, capabilities, configure
    from bcc.accel import get_backend, set_backend
except ImportError:
    print("BCC package not found. Please install with: pip install -e .")
    exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for streaming benchmark."""
    channels: int = 256
    sampling_rate: int = 30000  # Hz
    duration_seconds: float = 10.0
    buffer_size_ms: float = 10.0  # Buffer size in milliseconds
    backend: str = "cpu"
    algorithms: List[str] = None
    output_file: str = "benchmark_results.json"
    warmup_seconds: float = 2.0
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["lz4", "zstd", "blosc", "neural_lz77"]


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""
    algorithm: str
    backend: str
    channels: int
    sampling_rate: int
    compression_ratio: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_mbps: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float] = None
    samples_processed: int = 0
    errors: int = 0


class StreamingBenchmark:
    """Real-time streaming benchmark for neural data compression."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: List[PerformanceMetrics] = []
        
        # Calculate derived parameters
        self.buffer_samples = int(config.sampling_rate * config.buffer_size_ms / 1000)
        self.total_samples = int(config.sampling_rate * config.duration_seconds)
        self.bytes_per_sample = 2  # 16-bit samples
        
        self.logger.info(f"Benchmark config: {config}")
        self.logger.info(f"Buffer size: {self.buffer_samples} samples")
        self.logger.info(f"Total samples: {self.total_samples}")
    
    async def generate_neural_data(self) -> np.ndarray:
        """Generate synthetic neural data with realistic characteristics."""
        # Generate multi-channel neural data with:
        # - Spike trains (sparse events)
        # - LFP oscillations (alpha, beta, gamma bands)
        # - Noise and artifacts
        
        data = np.random.randn(self.config.channels, self.buffer_samples).astype(np.float32)
        
        # Add spike trains (sparse events)
        spike_prob = 0.01  # 1% chance of spike per sample
        spike_mask = np.random.random((self.config.channels, self.buffer_samples)) < spike_prob
        data[spike_mask] += np.random.normal(0, 5, spike_mask.sum())
        
        # Add oscillatory components
        t = np.linspace(0, self.config.buffer_size_ms/1000, self.buffer_samples)
        for ch in range(self.config.channels):
            # Alpha band (8-13 Hz)
            alpha_freq = 8 + 5 * np.random.random()
            data[ch] += 0.5 * np.sin(2 * np.pi * alpha_freq * t)
            
            # Beta band (13-30 Hz)  
            beta_freq = 13 + 17 * np.random.random()
            data[ch] += 0.3 * np.sin(2 * np.pi * beta_freq * t)
        
        # Convert to int16 (typical ADC output)
        data = (data * 1000).astype(np.int16)
        
        return data
    
    async def benchmark_algorithm(self, algorithm: str) -> PerformanceMetrics:
        """Benchmark a single compression algorithm."""
        self.logger.info(f"Benchmarking {algorithm} on {self.config.backend}")
        
        # Initialize metrics tracking
        latencies = []
        compression_ratios = []
        samples_processed = 0
        errors = 0
        
        # System monitoring
        process = psutil.Process()
        backend = get_backend()
        
        # Warmup phase
        self.logger.info("Warmup phase...")
        warmup_end = time.time() + self.config.warmup_seconds
        while time.time() < warmup_end:
            data = await self.generate_neural_data()
            try:
                config = configure({"algorithm": algorithm})
                result = compress(data, config)
                compression_ratios.append(result.compression_ratio)
            except Exception as e:
                self.logger.warning(f"Warmup error: {e}")
        
        # Reset metrics after warmup
        latencies.clear()
        compression_ratios.clear()
        
        # Main benchmark loop
        self.logger.info("Starting benchmark...")
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds
        
        cpu_usage_samples = []
        memory_usage_samples = []
        
        while time.time() < end_time:
            # Generate data
            data = await self.generate_neural_data()
            
            # Measure compression latency
            compress_start = time.perf_counter()
            try:
                config = configure({"algorithm": algorithm})
                result = compress(data, config)
                compress_end = time.perf_counter()
                
                latency_ms = (compress_end - compress_start) * 1000
                latencies.append(latency_ms)
                compression_ratios.append(result.compression_ratio)
                samples_processed += data.size
                
            except Exception as e:
                self.logger.error(f"Compression error: {e}")
                errors += 1
                continue
            
            # Sample system metrics
            cpu_usage_samples.append(process.cpu_percent())
            memory_usage_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            # Simulate real-time constraints
            await asyncio.sleep(self.config.buffer_size_ms / 1000)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        total_bytes = samples_processed * self.bytes_per_sample
        throughput_mbps = (total_bytes / (1024 * 1024)) / total_time
        
        # Get GPU memory usage if available
        gpu_memory_mb = None
        if backend.name in ["CUDA", "ROCm"]:
            try:
                free_mem, total_mem = backend.get_memory_info()
                gpu_memory_mb = (total_mem - free_mem) / (1024 * 1024)
            except Exception:
                pass
        
        metrics = PerformanceMetrics(
            algorithm=algorithm,
            backend=self.config.backend,
            channels=self.config.channels,
            sampling_rate=self.config.sampling_rate,
            compression_ratio=statistics.mean(compression_ratios) if compression_ratios else 0.0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p50_latency_ms=statistics.quantiles(latencies, n=2)[0] if latencies else 0.0,
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0.0,
            p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            throughput_mbps=throughput_mbps,
            cpu_usage_percent=statistics.mean(cpu_usage_samples) if cpu_usage_samples else 0.0,
            memory_usage_mb=statistics.mean(memory_usage_samples) if memory_usage_samples else 0.0,
            gpu_memory_usage_mb=gpu_memory_mb,
            samples_processed=samples_processed,
            errors=errors
        )
        
        self.logger.info(f"Algorithm {algorithm} completed:")
        self.logger.info(f"  Compression ratio: {metrics.compression_ratio:.2f}x")
        self.logger.info(f"  Mean latency: {metrics.mean_latency_ms:.2f}ms")
        self.logger.info(f"  P99 latency: {metrics.p99_latency_ms:.2f}ms")
        self.logger.info(f"  Throughput: {metrics.throughput_mbps:.2f} MB/s")
        self.logger.info(f"  Errors: {metrics.errors}")
        
        return metrics
    
    async def run_benchmark(self) -> List[PerformanceMetrics]:
        """Run complete benchmark suite."""
        self.logger.info("Starting streaming benchmark suite")
        
        # Set backend
        set_backend(self.config.backend)
        
        # Verify capabilities
        caps = capabilities()
        self.logger.info(f"Backend capabilities: {caps}")
        
        # Run benchmarks for each algorithm
        for algorithm in self.config.algorithms:
            try:
                metrics = await self.benchmark_algorithm(algorithm)
                self.results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to benchmark {algorithm}: {e}")
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "config": asdict(self.config),
            "timestamp": time.time(),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")


async def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="BCI streaming compression benchmark")
    parser.add_argument("--channels", type=int, default=256, help="Number of channels")
    parser.add_argument("--sampling-rate", type=int, default=30000, help="Sampling rate (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration (seconds)")
    parser.add_argument("--buffer-size", type=float, default=10.0, help="Buffer size (ms)")
    parser.add_argument("--backend", choices=["cpu", "cuda", "rocm"], default="cpu", help="Acceleration backend")
    parser.add_argument("--algorithms", nargs="+", default=["lz4", "zstd", "blosc"], help="Algorithms to test")
    parser.add_argument("--output", default="logs/benchmark_results.json", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create benchmark config
    config = BenchmarkConfig(
        channels=args.channels,
        sampling_rate=args.sampling_rate,
        duration_seconds=args.duration,
        buffer_size_ms=args.buffer_size,
        backend=args.backend,
        algorithms=args.algorithms,
        output_file=args.output
    )
    
    # Run benchmark
    benchmark = StreamingBenchmark(config)
    results = await benchmark.run_benchmark()
    benchmark.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result.algorithm:15} | "
              f"{result.compression_ratio:6.2f}x | "
              f"{result.mean_latency_ms:7.2f}ms | "
              f"{result.p99_latency_ms:7.2f}ms | "
              f"{result.throughput_mbps:8.1f} MB/s")


if __name__ == "__main__":
    asyncio.run(main())
