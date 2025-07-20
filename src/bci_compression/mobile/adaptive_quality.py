"""
Adaptive Quality Controller for Mobile BCI Compression

Dynamically adjusts compression quality in real-time based on signal or device state.
"""

from typing import Optional


class AdaptiveQualityController:
    """
    Adjusts MobileBCICompressor quality level in real-time.
    Can use signal statistics, device battery, or user input.
    """
    def __init__(self, compressor):
        self.compressor = compressor

    def adjust_quality(self, signal_snr: Optional[float] = None, battery_level: Optional[float] = None):
        """
        Adjust quality based on SNR or battery level.
        - If SNR drops, increase quality.
        - If battery is low, decrease quality.
        """
        if battery_level is not None:
            if battery_level < 0.2:
                self.compressor.quality_level = max(0.5, self.compressor.quality_level - 0.1)
            elif battery_level > 0.8:
                self.compressor.quality_level = min(1.0, self.compressor.quality_level + 0.1)
        if signal_snr is not None:
            if signal_snr < 10:
                self.compressor.quality_level = min(1.0, self.compressor.quality_level + 0.1)
            elif signal_snr > 25:
                self.compressor.quality_level = max(0.5, self.compressor.quality_level - 0.1)

    def get_quality_level(self):
        return self.compressor.quality_level
