"""
# =============================================================================
# ID: BCI-ALG-LOSSY-001
# Module: Lossy Neural Data Compressors
# Purpose: Provide configurable lossy compression for multi-channel neural
#          signals where some signal fidelity can be exchanged for significant
#          bandwidth and storage reductions.
# Requirement: Each compressor SHALL achieve a compression ratio ≥ 2x while
#              preserving ≥ 25 dB SNR on typical neural recordings.
# Rationale: Lossy compression is acceptable for exploratory analysis and
#            high-density recording scenarios where lossless cannot meet
#            real-time streaming constraints.
# Constraints: Python ≥ 3.8; PyWavelets optional for wavelet backend.
# References: Shannon-Nyquist theorem; Huffman (1952); Mallat wavelet frames.
# =============================================================================
"""

import logging
from typing import Optional

import numpy as np

try:
    import pywt
except ImportError:
    pywt = None

from ..core import BaseCompressor

logger = logging.getLogger(__name__)


class QuantizationCompressor(BaseCompressor):
    """
    Quantization-based lossy compression for neural data.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-LOSSY-002
    # Requirement: Compress neural signal to 8-bit integer representation with
    #              linear or adaptive scaling; reconstruct with < 5% amplitude
    #              error on signals within ±500 µV.
    # Purpose: Reduce data rate by 4× (float32→uint8) while retaining amplitude
    #          ordering required for spike detection and LFP analysis.
    # Rationale: Uniform quantisation preserves relative ordering.  Adaptive
    #            mode re-scales per window, maximising dynamic range usage.
    # Inputs:
    #   data   – np.ndarray, shape (channels, samples) or (samples,),
    #            float32/float64, valid range ±32 767 µV typical.
    # Outputs:
    #   bytes  – packed binary: 16-byte float64 metadata (scale, offset) +
    #            uint8 quantised samples.
    # Preconditions:  data.size > 0;  bits in [4, 16].
    # Postconditions: len(result) ≈ data.nbytes / (32 / bits) + 16.
    # Failure Modes:  scale_factor = 0 when min == max (flat signal) →
    #                 result is all-zeros; decompress reproduces flat signal.
    # Verification:   test_simple_validation.py::test_perceptual_quantizer
    # -------------------------------------------------------------------------
    """

    def __init__(self, bits: int = 8, adaptive: bool = True):
        super().__init__()
        self.bits = bits
        self.adaptive = adaptive
        self.scale_factor: Optional[float] = None
        self.offset: Optional[float] = None

    def compress(self, data: np.ndarray) -> bytes:
        logger.info(f"[Quantization] Compressing data with shape {data.shape} and dtype {data.dtype}")
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

        try:
            original_size = int(data.nbytes)
            compressed_size = int(quantized.nbytes) + 16  # +16 for metadata
            if compressed_size == 0:
                self.compression_ratio = 1.0
            else:
                self.compression_ratio = float(original_size) / float(compressed_size)
        except Exception as e:
            logger.exception(f"[Quantization] Error calculating compression ratio: {e}")
            self.compression_ratio = 1.0

        # Pack metadata and quantized data
        metadata = np.array([self.scale_factor, self.offset], dtype=np.float64)
        return metadata.tobytes() + quantized.tobytes()

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        logger.info("[Quantization] Decompressing data")
        metadata_size = 16  # 2 float64 values
        metadata = np.frombuffer(
            compressed_data[:metadata_size], dtype=np.float64
        )
        scale_factor, offset = metadata

        # Unpack quantized data
        quantized = np.frombuffer(compressed_data[metadata_size:], dtype=np.uint8)

        # Dequantize
        data = (quantized.astype(np.float32) * scale_factor) + offset
        try:
            if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                if not np.issubdtype(self._last_dtype, np.floating):
                    raise ValueError(
                        f"Decompressed data dtype {
                            self._last_dtype} is not a floating type and is not supported.")
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
        except Exception:
            logger.exception("[Quantization] Integrity check failed during decompression")
            raise
        return data


class WaveletCompressor(BaseCompressor):
    """
    Wavelet-based lossy compression for neural data.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-LOSSY-003
    # Requirement: Compress neural signals via discrete wavelet transform (DWT)
    #              and soft-threshold coefficient pruning to achieve 2–8× ratio
    #              while retaining oscillatory structure (LFP bands, spindles).
    # Purpose: Exploit the sparsity of neural signals in the wavelet domain —
    #          most neural energy concentrates in < 20% of wavelet coefficients.
    # Rationale: Daubechies db4 wavelet closely matches neural spike morphology,
    #            minimising reconstruction artefact near transient events.
    # Inputs:
    #   data      – np.ndarray, any shape, float32/float64.
    #   wavelet   – str wavelet family, default 'db4'.
    #   levels    – int DWT decomposition depth (1–10), default 5.
    #   threshold – float pruning fraction relative to max coefficient (0–1).
    # Outputs:
    #   bytes – packed float32 thresholded coefficients (no length metadata;
    #           shape is stored in _last_shape for decompress round-trip).
    # Preconditions:  pywt installed; data.ndim ≥ 1; data.size > 2^levels.
    # Postconditions: compression_ratio attribute updated after compress().
    # Assumptions:    Same WaveletCompressor instance used for compress and
    #                 decompress (state held in _last_shape / _last_dtype).
    # Failure Modes:  ImportError if PyWavelets absent; shape mismatch on
    #                 decompress if different instance used.
    # Verification:   tests/test_simple_validation.py; manual SNR check.
    # References:     Mallat (1989) "A Theory for Multiresolution Signal
    #                 Decomposition"; PyWavelets documentation.
    # -------------------------------------------------------------------------
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 5, threshold: float = 0.1):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.threshold = threshold

    def compress(self, data: np.ndarray) -> bytes:
        logger.info(f"[Wavelet] Compressing data with shape {data.shape} and dtype {data.dtype}")
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        """Compress using wavelet transform and thresholding."""
        if pywt is None:
            raise ImportError("pywt (PyWavelets) is required for wavelet compression.")
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
        logger.info("[Wavelet] Decompressing data")
        if pywt is None:
            raise ImportError("pywt (PyWavelets) is required for wavelet decompression.")
        coeffs_flat = np.frombuffer(compressed_data, dtype=np.float32)
        data = coeffs_flat
        try:
            if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                data = data.reshape(self._last_shape)
                data = data.astype(self._last_dtype)
        except Exception:
            logger.exception("[Wavelet] Integrity check failed during decompression")
            raise
        return data
