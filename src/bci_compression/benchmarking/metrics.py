"""
BenchmarkMetrics stub for testing.
"""
class BenchmarkMetrics:
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        """
        Calculate the compression ratio.
        Formula: original_size / compressed_size
        Returns ratio >= 1 for compression, < 1 for expansion.
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size

    @staticmethod
    def processing_latency(start_time: float, end_time: float) -> float:
        """
        Calculate processing latency in seconds.
        Formula: end_time - start_time
        """
        return end_time - start_time
