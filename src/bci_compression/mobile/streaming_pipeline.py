"""
Mobile Streaming Pipeline for Real-Time BCI Compression

Provides a lightweight, low-latency streaming pipeline for mobile and embedded BCI devices.
Handles chunked data, buffering, and real-time compression using MobileBCICompressor.
"""

import time
from typing import Callable, Optional

import numpy as np

from .mobile_compressor import MobileBCICompressor


class MobileStreamingPipeline:
    """
    Real-time streaming pipeline for mobile BCI compression.
    - Processes incoming data in chunks
    - Maintains low-latency and bounded memory usage
    - Integrates with MobileBCICompressor
    """
    def __init__(self,
                 compressor: Optional[MobileBCICompressor] = None,
                 buffer_size: int = 256,
                 overlap: int = 32):
        self.compressor = compressor or MobileBCICompressor()
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.buffer = []
        self.stats = {
            'chunks_processed': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0
        }

    def process_stream(self, data_stream: Callable[[], Optional[np.ndarray]]):
        """
        Process a data stream in real-time, compressing each chunk.
        data_stream: Callable that yields np.ndarray or None when done.
        """
        buffer = []
        while True:
            chunk = data_stream()
            if chunk is None:
                break
            buffer.extend(chunk)
            while len(buffer) >= self.buffer_size:
                window = np.array(buffer[:self.buffer_size])
                start = time.time()
                compressed = self.compressor.compress(window)
                latency = (time.time() - start) * 1000
                self.stats['chunks_processed'] += 1
                self.stats['total_latency_ms'] += latency
                print(f"Chunk {self.stats['chunks_processed']}: Latency={latency:.2f}ms, Size={len(compressed)} bytes")
                buffer = buffer[self.buffer_size - self.overlap:]
        if self.stats['chunks_processed']:
            self.stats['avg_latency_ms'] = self.stats['total_latency_ms'] / self.stats['chunks_processed']

    def reset_stats(self):
        self.stats = {
            'chunks_processed': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0
        }
