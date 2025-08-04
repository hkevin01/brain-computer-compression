"""Example script demonstrating the benchmarking framework."""
import numpy as np

from src.benchmarking import (
    CompressionBenchmark,
    BenchmarkDataset,
    HardwareBenchmark,
    generate_synthetic_data,
    run_benchmark,
    STANDARD_BENCHMARKS
)


class MockCompressor:
    """Mock compressor for demonstration."""

    def __init__(self, compression_ratio=10.0, snr_db=30.0, latency_ms=1.0):
        self.compression_ratio = compression_ratio
        self.snr_db = snr_db
        self.latency_ms = latency_ms

    def compress(self, data):
        """Simulate compression by returning smaller array."""
        compressed_size = int(data.size / self.compression_ratio)
        return np.random.normal(0, 1, compressed_size)

    def decompress(self, compressed_data):
        """Simulate decompression with controlled SNR."""
        original_shape = (
            int(compressed_data.size * self.compression_ratio / 1000),
            1000
        )
        reconstructed = np.random.normal(0, 1, original_shape)
        # Add noise to achieve target SNR
        noise_std = 10 ** (-self.snr_db / 20)
        reconstructed += np.random.normal(0, noise_std, original_shape)
        return reconstructed


def main():
    """Run example benchmarks."""
    print("Running benchmarking examples...")

    # Create mock compressor
    compressor = MockCompressor(
        compression_ratio=15.0,
        snr_db=30.0,
        latency_ms=1.0
    )

    # Run standard benchmarks
    for name, config in STANDARD_BENCHMARKS.items():
        print(f"\nRunning {name} benchmark...")

        # Set up hardware config
        if config.channels >= 1024:
            hardware = HardwareBenchmark(
                name="gpu_high_density",
                device="cuda",
                batch_size=16,
                max_memory_mb=8000
            )
        else:
            hardware = HardwareBenchmark(
                name="cpu_standard",
                device="cpu",
                batch_size=8,
                max_memory_mb=2000
            )

        # Run benchmark
        results = run_benchmark(compressor, config, hardware)

        # Print results
        print(f"Compression ratio: {results['compression_ratio']:.2f}x")
        print(f"SNR: {results['snr_db']:.1f} dB")
        print(f"Latency (P95): {results['latency_ms']['p95']:.2f} ms")
        print(f"Passed: {results['passed']}")

        if 'hardware_metrics' in results:
            print("\nHardware metrics:")
            for key, value in results['hardware_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")


if __name__ == "__main__":
    main()
