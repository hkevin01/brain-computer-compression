"""
Streaming Data Compression Example

Demonstrates real-time compression of streaming neural data with:
- Sliding window processing
- Buffer management
- Online compression with minimal latency
- Memory-efficient operation
"""

import numpy as np
import sys
import time
from pathlib import Path
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bci_compression.adapters.openbci import OpenBCIAdapter
from bci_compression.algorithms.lossless import NeuralLZ77Compressor
from bci_compression.adapters import apply_calibration


class StreamingCompressor:
    """
    Real-time streaming compressor for neural data.
    
    Features:
    - Fixed-size sliding window
    - Overlap for continuity
    - Online compression with minimal latency
    - Memory-efficient circular buffer
    """
    
    def __init__(
        self,
        n_channels: int,
        window_size: int = 1000,
        overlap: int = 100,
        adapter=None,
        compressor=None
    ):
        """
        Initialize streaming compressor.
        
        Args:
            n_channels: Number of channels
            window_size: Size of processing window in samples
            overlap: Overlap between windows in samples
            adapter: Device adapter (e.g., OpenBCIAdapter)
            compressor: Compression algorithm
        """
        self.n_channels = n_channels
        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = window_size - overlap
        
        self.adapter = adapter or OpenBCIAdapter()
        self.compressor = compressor or NeuralLZ77Compressor()
        
        # Circular buffer for incoming data
        self.buffer = deque(maxlen=window_size)
        
        # Stats
        self.windows_processed = 0
        self.total_samples_in = 0
        self.total_bytes_compressed = 0
        self.total_compression_time = 0.0
    
    def process_chunk(self, chunk: np.ndarray) -> bytes:
        """
        Process a chunk of incoming data.
        
        Args:
            chunk: New data chunk (channels x samples)
        
        Returns:
            Compressed bytes for this window
        """
        n_samples = chunk.shape[1]
        self.total_samples_in += n_samples
        
        # Add to buffer
        for i in range(n_samples):
            self.buffer.append(chunk[:, i])
        
        # Check if we have enough data to process a window
        if len(self.buffer) < self.window_size:
            return b''  # Not enough data yet
        
        # Extract window
        window_data = np.array(list(self.buffer)[-self.window_size:]).T
        
        # Apply adapter preprocessing
        if self.adapter:
            window_data = self.adapter.convert(window_data, apply_mapping=False)
        
        # Compress
        start_time = time.time()
        compressed = self.compressor.compress(window_data)
        compress_time = time.time() - start_time
        
        # Update stats
        self.windows_processed += 1
        self.total_bytes_compressed += len(compressed)
        self.total_compression_time += compress_time
        
        return compressed
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        if self.windows_processed == 0:
            return {'windows_processed': 0}
        
        return {
            'windows_processed': self.windows_processed,
            'total_samples_in': self.total_samples_in,
            'total_bytes_compressed': self.total_bytes_compressed,
            'avg_compression_time_ms': (self.total_compression_time / self.windows_processed) * 1000,
            'avg_bytes_per_window': self.total_bytes_compressed / self.windows_processed,
            'compression_ratio': (self.total_samples_in * self.n_channels * 8) / self.total_bytes_compressed,
        }


def simulate_data_stream(n_channels=8, n_samples=10000, chunk_size=250):
    """
    Simulate a streaming data source.
    
    Args:
        n_channels: Number of channels
        n_samples: Total samples to generate
        chunk_size: Samples per chunk
    
    Yields:
        Data chunks (channels x chunk_size)
    """
    for i in range(0, n_samples, chunk_size):
        # Generate realistic neural-like data
        t = np.arange(chunk_size) / 250.0 + i / 250.0
        chunk = np.zeros((n_channels, chunk_size))
        
        for ch in range(n_channels):
            # Mix of frequencies
            alpha = np.sin(2 * np.pi * (8 + ch * 0.5) * t)
            beta = np.sin(2 * np.pi * (15 + ch * 0.5) * t) * 0.5
            gamma = np.sin(2 * np.pi * (30 + ch) * t) * 0.3
            noise = np.random.randn(chunk_size) * 0.1
            
            chunk[ch] = alpha + beta + gamma + noise
        
        yield chunk
        
        # Simulate real-time delay
        time.sleep(0.01)  # 10ms delay


def demo_streaming_compression():
    """Demonstrate streaming compression."""
    print("="*70)
    print("Streaming Data Compression Demo")
    print("="*70)
    print()
    
    # Setup
    n_channels = 8
    window_size = 1000  # 4 seconds @ 250Hz
    overlap = 250  # 1 second overlap
    
    print(f"Configuration:")
    print(f"  Channels: {n_channels}")
    print(f"  Window size: {window_size} samples (4.0s @ 250Hz)")
    print(f"  Overlap: {overlap} samples (1.0s)")
    print(f"  Hop size: {window_size - overlap} samples")
    print()
    
    # Create streaming compressor
    adapter = OpenBCIAdapter(device='cyton_8ch')
    compressor = StreamingCompressor(
        n_channels=n_channels,
        window_size=window_size,
        overlap=overlap,
        adapter=adapter
    )
    
    print("Starting stream...")
    print()
    
    # Process streaming data
    chunk_count = 0
    for chunk in simulate_data_stream(n_channels=n_channels, n_samples=5000, chunk_size=250):
        compressed = compressor.process_chunk(chunk)
        
        chunk_count += 1
        if chunk_count % 4 == 0:  # Update every 4 chunks (1 second)
            stats = compressor.get_stats()
            if stats['windows_processed'] > 0:
                print(f"Time: {chunk_count * 250 / 250:.1f}s | "
                      f"Windows: {stats['windows_processed']} | "
                      f"Ratio: {stats['compression_ratio']:.2f}x | "
                      f"Latency: {stats['avg_compression_time_ms']:.2f}ms")
    
    print()
    print("Stream complete!")
    print()
    
    # Final stats
    stats = compressor.get_stats()
    print("Final Statistics:")
    print(f"  Windows processed: {stats['windows_processed']}")
    print(f"  Total samples: {stats['total_samples_in']:,}")
    print(f"  Compressed size: {stats['total_bytes_compressed']:,} bytes ({stats['total_bytes_compressed']/1024:.2f} KB)")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Avg latency: {stats['avg_compression_time_ms']:.2f}ms per window")
    print()
    
    # Calculate throughput
    total_time = stats['total_samples_in'] / 250.0  # seconds at 250Hz
    throughput = stats['total_samples_in'] * n_channels / total_time
    print(f"  Throughput: {throughput:,.0f} samples/sec")
    print()


def demo_low_latency_mode():
    """Demonstrate ultra-low latency streaming."""
    print("="*70)
    print("Low-Latency Streaming Mode")
    print("="*70)
    print()
    
    print("Configuration:")
    print("  Window size: 100 samples (400ms @ 250Hz)")
    print("  Overlap: 10 samples (40ms)")
    print("  Target latency: <1ms")
    print()
    
    compressor = StreamingCompressor(
        n_channels=8,
        window_size=100,
        overlap=10
    )
    
    latencies = []
    
    print("Processing...")
    for i, chunk in enumerate(simulate_data_stream(n_channels=8, n_samples=1000, chunk_size=50)):
        start = time.time()
        compressed = compressor.process_chunk(chunk)
        latency = (time.time() - start) * 1000  # ms
        
        if compressed:
            latencies.append(latency)
        
        if i >= 20:  # Process 20 chunks
            break
    
    print()
    print("Latency Statistics:")
    if latencies:
        print(f"  Mean: {np.mean(latencies):.3f}ms")
        print(f"  Median: {np.median(latencies):.3f}ms")
        print(f"  Min: {np.min(latencies):.3f}ms")
        print(f"  Max: {np.max(latencies):.3f}ms")
        print(f"  Std: {np.std(latencies):.3f}ms")
    print()


def main():
    """Run all demonstrations."""
    try:
        demo_streaming_compression()
        demo_low_latency_mode()
        
        print("="*70)
        print("✓ All streaming demos completed successfully!")
        print("="*70)
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
