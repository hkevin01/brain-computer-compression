"""
Power Optimizer for Mobile BCI Compression

Dynamically adjusts compression parameters to optimize for battery life, balanced operation, or performance.
"""


class PowerOptimizer:
    """
    Adjusts MobileBCICompressor parameters for power, memory, and performance trade-offs.
    """
    def __init__(self, compressor):
        self.compressor = compressor

    def set_mode(self, mode: str):
        """
        Set power optimization mode: 'battery_save', 'balanced', 'performance'.
        """
        if mode == 'battery_save':
            self.compressor.quality_level = 0.6
            self.compressor.buffer_size = 128
            self.compressor.max_memory_mb = 20
        elif mode == 'balanced':
            self.compressor.quality_level = 0.8
            self.compressor.buffer_size = 256
            self.compressor.max_memory_mb = 50
        elif mode == 'performance':
            self.compressor.quality_level = 1.0
            self.compressor.buffer_size = 512
            self.compressor.max_memory_mb = 100
        else:
            raise ValueError(f"Unknown power mode: {mode}")
        self.compressor.power_mode = mode

    def get_current_mode(self):
        return self.compressor.power_mode
