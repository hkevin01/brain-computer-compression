"""
Real-time streaming processor for neural data.
Implements sliding window buffering and stateful chunk processing.

References:
- Real-time BCI applications
- Buffer management, low-latency processing
"""
from typing import Iterator, Tuple
import numpy as np


def stream_processor(data: np.ndarray, buffer_size: int = 256, overlap: int = 64) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Processes data in sliding windows for real-time streaming.
    Args:
        data: Input neural data (channels x samples)
        buffer_size: Window size
        overlap: Number of overlapping samples
    Yields:
        Tuple of (windowed data, window start index)
    """
    num_samples = data.shape[1]
    start = 0
    while start < num_samples:
        end = min(start + buffer_size, num_samples)
        window = data[:, start:end]
        yield window, start
        start += buffer_size - overlap
