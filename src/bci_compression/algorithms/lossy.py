"""
Lossy compression algorithms for neural data.
"""

import numpy as np
import pywt
from typing import Optional
from ..core import BaseCompressor


class QuantizationCompressor(BaseCompressor):
    """
    Quantization-based lossy compression for neural data.
    """
    
    def __init__(self, bits: int = 8, adaptive: bool = True):
        super().__init__()
        self.bits = bits
        self.adaptive = adaptive
        self.scale_factor: Optional[float] = None
        self.offset: Optional[float] = None
    
    def compress(self, data: np.ndarray) -> bytes:
        """Compress using uniform quantization."""
        # Calculate quantization parameters
        if self.adaptive:
            self.scale_factor = (data.max() - data.min()) / (2**self.bits - 1)
            self.offset = data.min()
        else:
            self.scale_factor = 1.0
            self.offset = 0.0
        
        # Quantize data
        quantized = ((data - self.offset) / self.scale_factor).astype(np.uint8)
        
        original_size = data.nbytes
        compressed_size = quantized.nbytes + 16  # +16 for metadata
        self.compression_ratio = original_size / compressed_size
        
        # Pack metadata and quantized data
        metadata = np.array([self.scale_factor, self.offset], dtype=np.float64)
        return metadata.tobytes() + quantized.tobytes()
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress quantized data."""
        # Unpack metadata
        metadata_size = 16  # 2 float64 values
        metadata = np.frombuffer(
            compressed_data[:metadata_size], dtype=np.float64
        )
        scale_factor, offset = metadata
        
        # Unpack quantized data
        quantized = np.frombuffer(compressed_data[metadata_size:], dtype=np.uint8)
        
        # Dequantize
        return (quantized.astype(np.float32) * scale_factor) + offset


class WaveletCompressor(BaseCompressor):
    """
    Wavelet-based lossy compression for neural data.
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 5, threshold: float = 0.1):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.threshold = threshold
    
    def compress(self, data: np.ndarray) -> bytes:
        """Compress using wavelet transform and thresholding."""
        # Wavelet decomposition
        coeffs = pywt.wavedec(data.flatten(), self.wavelet, level=self.levels)
        
        # Threshold small coefficients
        coeffs_thresh = []
        for coeff in coeffs:
            thresh_coeff = pywt.threshold(coeff, self.threshold * np.max(np.abs(coeff)))
            coeffs_thresh.append(thresh_coeff)
        
        # Convert to bytes (simplified)
        flattened = np.concatenate([c.flatten() for c in coeffs_thresh])
        compressed = flattened.astype(np.float32).tobytes()
        
        original_size = data.nbytes
        self.compression_ratio = original_size / len(compressed)
        
        return compressed
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress wavelet-compressed data."""
        # This is a simplified implementation
        # In practice, would need to store coefficient structure
        coeffs_flat = np.frombuffer(compressed_data, dtype=np.float32)
        
        # Simplified reconstruction (would need proper coefficient structure)
        return coeffs_flat
