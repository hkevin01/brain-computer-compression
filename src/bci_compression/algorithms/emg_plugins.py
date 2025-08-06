"""
EMG Plugin Registration

This module registers EMG compression algorithms with the plugin system
to make them discoverable and accessible through the standard interface.
"""

from bci_compression.plugins import register_plugin, CompressorPlugin
from typing import Any
import numpy as np


# Register EMG LZ Compressor
@register_plugin("emg_lz")
class EMGLZCompressorPlugin(CompressorPlugin):
    """EMG LZ Compressor plugin wrapper."""
    name = "emg_lz"
    
    def __init__(self, **kwargs):
        from bci_compression.algorithms.emg_compression import EMGLZCompressor
        self.compressor = EMGLZCompressor(**kwargs)
    
    def compress(self, data: Any, **kwargs) -> Any:
        return self.compressor.compress(data, **kwargs)
    
    def decompress(self, data: Any, **kwargs) -> Any:
        return self.compressor.decompress(data, **kwargs)
    
    def fit(self, data: Any, **kwargs) -> None:
        if hasattr(self.compressor, 'fit'):
            self.compressor.fit(data, **kwargs)


# Register EMG Perceptual Quantizer
@register_plugin("emg_perceptual")
class EMGPerceptualQuantizerPlugin(CompressorPlugin):
    """EMG Perceptual Quantizer plugin wrapper."""
    name = "emg_perceptual"
    
    def __init__(self, **kwargs):
        from bci_compression.algorithms.emg_compression import EMGPerceptualQuantizer
        self.compressor = EMGPerceptualQuantizer(**kwargs)
    
    def compress(self, data: Any, **kwargs) -> Any:
        return self.compressor.compress(data, **kwargs)
    
    def decompress(self, data: Any, **kwargs) -> Any:
        return self.compressor.decompress(data, **kwargs)
    
    def fit(self, data: Any, **kwargs) -> None:
        if hasattr(self.compressor, 'fit'):
            self.compressor.fit(data, **kwargs)


# Register EMG Predictive Compressor
@register_plugin("emg_predictive")
class EMGPredictiveCompressorPlugin(CompressorPlugin):
    """EMG Predictive Compressor plugin wrapper."""
    name = "emg_predictive"
    
    def __init__(self, **kwargs):
        from bci_compression.algorithms.emg_compression import EMGPredictiveCompressor
        self.compressor = EMGPredictiveCompressor(**kwargs)
    
    def compress(self, data: Any, **kwargs) -> Any:
        return self.compressor.compress(data, **kwargs)
    
    def decompress(self, data: Any, **kwargs) -> Any:
        return self.compressor.decompress(data, **kwargs)
    
    def fit(self, data: Any, **kwargs) -> None:
        if hasattr(self.compressor, 'fit'):
            self.compressor.fit(data, **kwargs)


# Register Mobile EMG Compressor
@register_plugin("mobile_emg")
class MobileEMGCompressorPlugin(CompressorPlugin):
    """Mobile EMG Compressor plugin wrapper."""
    name = "mobile_emg"
    
    def __init__(self, **kwargs):
        from bci_compression.mobile.emg_mobile import MobileEMGCompressor
        self.compressor = MobileEMGCompressor(**kwargs)
    
    def compress(self, data: Any, **kwargs) -> Any:
        return self.compressor.compress(data, **kwargs)
    
    def decompress(self, data: Any, **kwargs) -> Any:
        return self.compressor.decompress(data, **kwargs)
    
    def fit(self, data: Any, **kwargs) -> None:
        if hasattr(self.compressor, 'fit'):
            self.compressor.fit(data, **kwargs)


def get_emg_compressors():
    """Get all registered EMG compressors."""
    from bci_compression.plugins import PLUGIN_REGISTRY
    
    emg_compressors = {}
    for name, plugin_class in PLUGIN_REGISTRY.items():
        if name.startswith('emg_') or name.startswith('mobile_emg'):
            emg_compressors[name] = plugin_class
    
    return emg_compressors


def create_emg_compressor(name: str, **kwargs) -> CompressorPlugin:
    """
    Create an EMG compressor by name.
    
    Parameters
    ----------
    name : str
        Name of the EMG compressor ('emg_lz', 'emg_perceptual', 'emg_predictive', 'mobile_emg')
    **kwargs
        Parameters to pass to the compressor constructor
        
    Returns
    -------
    CompressorPlugin
        Initialized EMG compressor plugin
    """
    from bci_compression.plugins import get_plugin
    
    plugin_class = get_plugin(name)
    return plugin_class(**kwargs)


# Demo function
def demo_emg_plugins():
    """Demonstrate EMG plugin usage."""
    print("Available EMG compressors:")
    emg_compressors = get_emg_compressors()
    for name in emg_compressors:
        print(f"  - {name}")
    
    # Create test data
    test_data = np.random.randn(4, 1000)  # 4 channels, 1000 samples
    
    # Test each compressor
    for name in emg_compressors:
        try:
            print(f"\nTesting {name}...")
            compressor = create_emg_compressor(name, sampling_rate=1000.0)
            
            # Compress
            compressed = compressor.compress(test_data)
            print(f"  Compressed: {len(compressed)} bytes")
            
            # Decompress
            decompressed = compressor.decompress(compressed)
            print(f"  Decompressed shape: {decompressed.shape}")
            
        except Exception as e:
            print(f"  Error testing {name}: {e}")


if __name__ == "__main__":
    demo_emg_plugins()
