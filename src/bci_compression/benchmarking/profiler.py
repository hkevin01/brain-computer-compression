"""
PerformanceProfiler stub for benchmarking tests.
"""

class PerformanceProfiler:
    def __init__(self):
        pass

    def profile(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def cpu_usage(self):
        # Stub: return a float value
        return 0.0

    def memory_usage(self):
        # Stub: return an int value
        return 0
