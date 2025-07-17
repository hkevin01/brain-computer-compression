"""
Lossy compression algorithms for neural data.
"""

from typing import Optional

import numpy as np
import pywt

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
        self._last_shape = data.shape
        self._last_dtype = data.dtype
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
        metadata_size = 16  # 2 float64 values
        metadata = np.frombuffer(
            compressed_data[:metadata_size], dtype=np.float64
        )
        scale_factor, offset = metadata
        
        # Unpack quantized data
        quantized = np.frombuffer(compressed_data[metadata_size:], dtype=np.uint8)
        
        # Dequantize
        data = (quantized.astype(np.float32) * scale_factor) + offset
        if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
            try:
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            self._check_integrity(np.zeros(self._last_shape, dtype=self._last_dtype), data, check_shape=True, check_dtype=True, check_hash=False)
        return data


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
        self._last_shape = data.shape
        self._last_dtype = data.dtype
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
        coeffs_flat = np.frombuffer(compressed_data, dtype=np.float32)
        data = coeffs_flat
        if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
            try:
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            self._check_integrity(np.zeros(self._last_shape, dtype=self._last_dtype), data, check_shape=True, check_dtype=True, check_hash=False)
        return data
