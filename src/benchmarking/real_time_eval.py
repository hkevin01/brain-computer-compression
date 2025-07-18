"""
Real-Time Performance Evaluation for BCI Compression Toolkit

Evaluates compression algorithms under streaming/real-time conditions.

References:
- Real-time BCI systems
- IEEE real-time signal processing
"""

import time
from typing import Callable, Any


class RealTimeEvaluator:
    """
    Evaluates compression in streaming scenarios.
    """
    def __init__(self, compressor: Any, buffer_size: int = 1024, overlap: int = 256):
        self.compressor = compressor
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.state = None

    def stream_process(self, data_stream: Callable[[], Any]):
        """
        Process data in a sliding window fashion for real-time evaluation.
        """
        buffer = []
        while True:
            chunk = data_stream()
            if chunk is None:
                break
            buffer.extend(chunk)
            if len(buffer) >= self.buffer_size:
                window = buffer[:self.buffer_size]
                start = time.time()
                compressed = self.compressor.compress(window)
                end = time.time()
                # Optionally decompress and check quality
                latency = (end - start) * 1000
                print(f"Window processed: Latency={latency:.2f}ms, Size={len(compressed)} bytes")
                buffer = buffer[self.buffer_size - self.overlap:]
