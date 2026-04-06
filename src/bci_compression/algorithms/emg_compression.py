"""
EMG (Electromyography) Data Compression Algorithms

This module provides specialized compression algorithms optimized for EMG signals,
which have different characteristics than neural BCI data:
- Lower sampling rates (1-2kHz vs 30kHz)
- Different frequency content (20-500Hz muscle activity)
- Focus on muscle activation patterns and timing preservation
- Clinical analysis requirements (envelope preservation, activation detection)

References:
- Clancy, E. A., et al. "Sampling, noise-reduction and amplitude estimation
  issues in surface electromyography." Journal of Electromyography and
  Kinesiology 12.1 (2002): 1-16.
- De Luca, C. J. "The use of surface electromyography in biomechanics."
  Journal of Applied Biomechanics 13.2 (1997): 135-163.
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.signal import hilbert

from ..core import BaseCompressor
from ..plugins import CompressorPlugin, register_plugin

logger = logging.getLogger(__name__)


@register_plugin("emg_lz")
class EMGLZCompressor(BaseCompressor, CompressorPlugin):
    """
    LZ-based EMG compressor leveraging muscle activation patterns.

    This compressor identifies muscle activation bursts and encodes them
    more efficiently than standard LZ77, taking advantage of:
    - Burst-like nature of muscle activations
    - Temporal patterns in muscle recruitment
    - Relatively sparse activation periods
    """

    name = "emg_lz"

    def __init__(
        self,
        activation_threshold: float = 0.1,
        burst_min_duration: float = 0.05,  # 50ms minimum burst
        pattern_buffer_size: int = 1024,
        sampling_rate: float = 2000.0
    ):
        """
        Initialize EMG LZ compressor.

        Parameters
        ----------
        activation_threshold : float, default=0.1
            Threshold for muscle activation detection (normalized)
        burst_min_duration : float, default=0.05
            Minimum duration for muscle burst in seconds
        pattern_buffer_size : int, default=1024
            Size of pattern dictionary for LZ compression
        sampling_rate : float, default=2000.0
            EMG sampling rate in Hz
        """
        super().__init__("emg_lz")
        self.activation_threshold = activation_threshold
        self.burst_min_duration = burst_min_duration
        self.pattern_buffer_size = pattern_buffer_size
        self.sampling_rate = sampling_rate

        # Derived parameters
        self.min_burst_samples = int(burst_min_duration * sampling_rate)

        # Pattern dictionary for LZ compression
        self.pattern_dict = {}
        self.dict_size = 0

        # Compression statistics
        self.compression_stats = {
            'activation_periods': 0,
            'rest_periods': 0,
            'patterns_found': 0,
            'compression_ratio': 0.0
        }

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress EMG data using activation-aware LZ encoding.

        Parameters
        ----------
        data : np.ndarray
            EMG data with shape (channels, samples) or (samples,)

        Returns
        -------
        bytes
            Compressed EMG data
        """
        start_time = time.time()

        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self._last_shape = data.shape
        self._last_dtype = data.dtype

        compressed_channels = []

        for ch in range(data.shape[0]):
            channel_data = data[ch, :]

            # Detect muscle activations
            activations = self._detect_activations(channel_data)

            # Compress using activation-aware encoding
            compressed_ch = self._compress_channel_with_activations(
                channel_data, activations
            )
            compressed_channels.append(compressed_ch)

        # Combine compressed channels
        compressed_data = self._pack_compressed_data(compressed_channels)

        # Update statistics
        processing_time = time.time() - start_time
        self.compression_stats.update({
            'compression_ratio': data.nbytes / len(compressed_data),
            'processing_time': processing_time
        })

        return compressed_data

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress EMG data.

        Parameters
        ----------
        compressed_data : bytes
            Compressed EMG data

        Returns
        -------
        np.ndarray
            Decompressed EMG data
        """
        # Unpack compressed data
        compressed_channels = self._unpack_compressed_data(compressed_data)

        # Decompress each channel
        decompressed_channels = []
        for compressed_ch in compressed_channels:
            decompressed_ch = self._decompress_channel(compressed_ch)
            decompressed_channels.append(decompressed_ch)

        # Reconstruct array
        reconstructed = np.array(decompressed_channels)

        return reconstructed.reshape(self._last_shape).astype(self._last_dtype)

    def _detect_activations(self, data: np.ndarray) -> np.ndarray:
        """
        Detect muscle activation periods using envelope and threshold.

        Parameters
        ----------
        data : np.ndarray
            Single channel EMG data

        Returns
        -------
        np.ndarray
            Boolean array indicating activation periods
        """
        # Calculate EMG envelope using Hilbert transform
        envelope = np.abs(hilbert(data))

        # Smooth envelope
        window_size = int(0.02 * self.sampling_rate)  # 20ms window
        # Ensure window size is valid (odd, >= 4, <= data length)
        window_size = max(5, min(window_size | 1, len(data) - 1))
        if window_size % 2 == 0:
            window_size -= 1
        envelope_smooth = signal.savgol_filter(envelope, window_size, min(3, window_size - 1))

        # Normalize envelope
        envelope_norm = envelope_smooth / (np.max(envelope_smooth) + 1e-8)

        # Threshold detection
        activations = envelope_norm > self.activation_threshold

        # Remove short bursts
        activations = self._remove_short_bursts(activations)

        return activations

    def _remove_short_bursts(self, activations: np.ndarray) -> np.ndarray:
        """Remove muscle activation bursts shorter than minimum duration."""
        # Find activation segments
        diff = np.diff(activations.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if activations[0]:
            starts = np.concatenate([[0], starts])
        if activations[-1]:
            ends = np.concatenate([ends, [len(activations)]])

        # Remove short bursts
        filtered_activations = activations.copy()
        for start, end in zip(starts, ends):
            if end - start < self.min_burst_samples:
                filtered_activations[start:end] = False

        return filtered_activations

    def _compress_channel_with_activations(
        self,
        data: np.ndarray,
        activations: np.ndarray
    ) -> bytes:
        """Compress single channel using activation patterns."""
        compressed = []

        # Separate activation and rest periods
        activation_segments = []
        rest_segments = []

        current_type = activations[0]
        start_idx = 0

        for i in range(1, len(activations)):
            if activations[i] != current_type:
                segment = data[start_idx:i]
                if current_type:
                    activation_segments.append((start_idx, i, segment))
                else:
                    rest_segments.append((start_idx, i, segment))
                start_idx = i
                current_type = activations[i]

        # Handle last segment
        segment = data[start_idx:]
        if current_type:
            activation_segments.append((start_idx, len(data), segment))
        else:
            rest_segments.append((start_idx, len(data), segment))

        # Compress activation periods with higher fidelity
        activation_data = []
        for start, end, segment in activation_segments:
            compressed_segment = self._compress_activation_segment(segment)
            activation_data.append((start, end, compressed_segment))

        # Compress rest periods more aggressively
        rest_data = []
        for start, end, segment in rest_segments:
            compressed_segment = self._compress_rest_segment(segment)
            rest_data.append((start, end, compressed_segment))

        # Pack data
        packed_data = {
            'activations': activation_data,
            'rest': rest_data,
            'length': len(data)
        }

        return self._serialize_channel_data(packed_data)

    def _compress_activation_segment(self, segment: np.ndarray) -> bytes:
        """Compress muscle activation segment with high fidelity."""
        # Use higher precision for activation periods
        # Quantize to 12 bits to preserve muscle force gradations
        quantized = np.clip(segment * 2048 + 2048, 0, 4095).astype(np.uint16)
        return quantized.tobytes()

    def _compress_rest_segment(self, segment: np.ndarray) -> bytes:
        """Compress rest segment with aggressive compression."""
        # Use lower precision for rest periods
        # Quantize to 8 bits since rest periods have less clinical relevance
        quantized = np.clip(segment * 128 + 128, 0, 255).astype(np.uint8)

        # Apply simple RLE for rest periods
        compressed = self._apply_rle(quantized)
        return compressed

    def _apply_rle(self, data: np.ndarray) -> bytes:
        """Apply run-length encoding."""
        compressed = []
        i = 0
        while i < len(data):
            current_val = data[i]
            count = 1

            # Count consecutive values
            while i + count < len(data) and data[i + count] == current_val and count < 255:
                count += 1

            compressed.extend([count, current_val])
            i += count

        return bytes(compressed)

    def _serialize_channel_data(self, data: Dict) -> bytes:
        """Serialize channel compression data."""
        # Simple serialization - in practice, use more robust format
        import pickle
        return pickle.dumps(data)

    def _pack_compressed_data(self, compressed_channels: List[bytes]) -> bytes:
        """Pack compressed channel data."""
        # Pack metadata
        metadata = {
            'shape': self._last_shape,
            'dtype': str(self._last_dtype),
            'n_channels': len(compressed_channels)
        }

        import pickle
        metadata_bytes = pickle.dumps(metadata)
        metadata_size = len(metadata_bytes).to_bytes(4, 'big')

        # Pack channel data
        channel_sizes = []
        channel_data = b''
        for ch_data in compressed_channels:
            channel_sizes.append(len(ch_data))
            channel_data += ch_data

        # Pack channel sizes
        sizes_bytes = pickle.dumps(channel_sizes)
        sizes_size = len(sizes_bytes).to_bytes(4, 'big')

        return metadata_size + metadata_bytes + sizes_size + sizes_bytes + channel_data

    def _unpack_compressed_data(self, compressed_data: bytes) -> List[bytes]:
        """Unpack compressed channel data."""
        import pickle

        # Unpack metadata
        metadata_size = int.from_bytes(compressed_data[:4], 'big')
        metadata = pickle.loads(compressed_data[4:4+metadata_size])

        self._last_shape = metadata['shape']
        self._last_dtype = np.dtype(metadata['dtype'])

        # Unpack channel sizes
        offset = 4 + metadata_size
        sizes_size = int.from_bytes(compressed_data[offset:offset+4], 'big')
        offset += 4
        channel_sizes = pickle.loads(compressed_data[offset:offset+sizes_size])
        offset += sizes_size

        # Unpack channel data
        compressed_channels = []
        for size in channel_sizes:
            ch_data = compressed_data[offset:offset+size]
            compressed_channels.append(ch_data)
            offset += size

        return compressed_channels

    def _decompress_channel(self, compressed_ch: bytes) -> np.ndarray:
        """Decompress single channel."""
        import pickle

        # Deserialize channel data
        data = pickle.loads(compressed_ch)

        # Reconstruct signal
        reconstructed = np.zeros(data['length'])

        # Reconstruct activation segments
        for start, end, segment_data in data['activations']:
            quantized = np.frombuffer(segment_data, dtype=np.uint16)
            segment = (quantized.astype(np.float32) - 2048) / 2048
            reconstructed[start:end] = segment

        # Reconstruct rest segments
        for start, end, segment_data in data['rest']:
            # Decompress RLE
            decompressed_rle = self._decompress_rle(segment_data)
            segment = (decompressed_rle.astype(np.float32) - 128) / 128
            reconstructed[start:end] = segment[:end-start]  # Handle size mismatch

        return reconstructed

    def _decompress_rle(self, compressed_data: bytes) -> np.ndarray:
        """Decompress run-length encoded data."""
        data = list(compressed_data)
        decompressed = []

        i = 0
        while i < len(data) - 1:
            count = data[i]
            value = data[i + 1]
            decompressed.extend([value] * count)
            i += 2

        return np.array(decompressed, dtype=np.uint8)


@register_plugin("emg_perceptual")
class EMGPerceptualQuantizer(BaseCompressor, CompressorPlugin):
    """
    Perceptual quantization for EMG focusing on clinically relevant frequencies.

    Preserves the 20-500Hz frequency range crucial for EMG analysis while
    removing noise and irrelevant frequency components.
    """

    name = "emg_perceptual"

    def __init__(
        self,
        sampling_rate: float = 2000.0,
        freq_bands: List[Tuple[float, float]] = None,
        quality_levels: List[int] = None,
        quality_level: float = None,  # alias for compatibility with tests / legacy API
    ):
        """
        Initialize EMG perceptual quantizer.

        Parameters
        ----------
        sampling_rate : float, default=2000.0
            EMG sampling rate in Hz
        freq_bands : list of tuples, optional
            Frequency bands with different quantization levels
            Default: [(20, 100), (100, 300), (300, 500)]
        quality_levels : list of int, optional
            Quantization bits for each frequency band
        quality_level : float, optional
            Single quality scalar (0.0-1.0); maps to per-band bit depths.
            Provided for API compatibility; quality_levels takes priority.
        """
        super().__init__("emg_perceptual")
        self.sampling_rate = sampling_rate

        # Resolve per-band quality from scalar quality_level if needed
        if quality_levels is None and quality_level is not None:
            # Scale quality_level (0-1) to reasonable bit-depth range (6-14 bits)
            bits = max(6, min(14, int(6 + quality_level * 8)))
            quality_levels = [bits, max(6, bits - 2), max(6, bits - 4)]

        if freq_bands is None:
            self.freq_bands = [(20, 100), (100, 300), (300, 500)]
            self.quality_levels = quality_levels if quality_levels is not None else [12, 10, 8]
        else:
            self.freq_bands = freq_bands
            self.quality_levels = quality_levels or [10] * len(freq_bands)

    def compress(self, data: np.ndarray) -> bytes:
        """Compress EMG using perceptual quantization."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self._last_shape = data.shape
        self._last_dtype = data.dtype

        compressed_channels = []

        for ch in range(data.shape[0]):
            channel_data = data[ch, :]

            # Apply frequency-band specific quantization
            compressed_ch = self._compress_perceptual_channel(channel_data)
            compressed_channels.append(compressed_ch)

        return self._pack_perceptual_data(compressed_channels)

    def _compress_perceptual_channel(self, data: np.ndarray) -> bytes:
        """
        Apply perceptual quantization to single channel.

        Averages the quantised frequency-band signals into one uint8
        representation, then zlib-compresses the result.
        """
        import struct
        import zlib

        # Filter to EMG band and quantize at highest quality level as representative
        filtered_data = self._emg_bandpass_filter(data)
        ch_min = float(filtered_data.min())
        ch_max = float(filtered_data.max())
        ch_range = ch_max - ch_min

        if ch_range < 1e-8:
            quantized = np.zeros(len(data), dtype=np.uint8)
        else:
            normalized = (filtered_data - ch_min) / ch_range
            quantized = np.round(normalized * 255).astype(np.uint8)

        payload = zlib.compress(quantized.tobytes(), level=6)
        # Pack: scale params (2 × float64) + payload length + payload
        header = struct.pack('>dd', ch_min, ch_max)
        header += struct.pack('>I', len(payload))
        return header + payload

    def _emg_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply EMG-specific bandpass filter (20-500Hz)."""
        nyquist = self.sampling_rate / 2
        low_norm = 20 / nyquist
        high_norm = min(500, nyquist * 0.95) / nyquist

        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, data)

    def _extract_frequency_band(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float
    ) -> np.ndarray:
        """Extract specific frequency band using bandpass filter."""
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq, nyquist * 0.95) / nyquist

        b, a = signal.butter(2, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, data)

    def _quantize_band(self, data: np.ndarray, n_bits: int) -> np.ndarray:
        """Quantize frequency band data to n_bits precision."""
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min

        if data_range < 1e-8:
            return np.zeros(len(data), dtype=np.uint8)

        levels = 2 ** min(n_bits, 8)  # cap at uint8 to limit output size
        normalized = (data - data_min) / data_range
        quantized = np.round(normalized * (levels - 1)).astype(np.uint8)
        return quantized

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress perceptually quantized EMG data."""
        import struct
        import zlib

        # Read header: n_channels(2B) + n_samples(4B) + dtype_len(1B) + dtype_str
        offset = 0
        n_channels, n_samples = struct.unpack_from('>HI', compressed_data, offset)
        offset += 6
        dtype_len = struct.unpack_from('>B', compressed_data, offset)[0]
        offset += 1
        dtype_str = compressed_data[offset: offset + dtype_len].decode()
        offset += dtype_len

        channels = []
        for _ in range(n_channels):
            # Read scale params (2 × float64 = 16 bytes)
            ch_min, ch_max = struct.unpack_from('>dd', compressed_data, offset)
            offset += 16
            # Read compressed payload length (4 bytes)
            payload_len = struct.unpack_from('>I', compressed_data, offset)[0]
            offset += 4
            payload = zlib.decompress(compressed_data[offset: offset + payload_len])
            offset += payload_len
            quantized = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
            # De-quantize
            if ch_max > ch_min:
                channel = quantized / 255.0 * (ch_max - ch_min) + ch_min
            else:
                channel = np.full(n_samples, ch_min, dtype=np.float32)
            channels.append(channel[:n_samples])

        result = np.stack(channels).astype(dtype_str)
        if hasattr(self, '_last_shape'):
            try:
                result = result.reshape(self._last_shape)
            except ValueError:
                pass
        return result

    def _pack_perceptual_data(self, compressed_channels: List[bytes]) -> bytes:
        """Pack perceptual compression data using efficient binary encoding."""
        import struct
        import zlib

        n_channels = self._last_shape[0]
        n_samples = self._last_shape[1]
        dtype_str = str(self._last_dtype).encode()

        header = struct.pack('>HI', n_channels, n_samples)
        header += struct.pack('>B', len(dtype_str)) + dtype_str
        return header + b''.join(compressed_channels)


@register_plugin("emg_predictive")
class EMGPredictiveCompressor(BaseCompressor, CompressorPlugin):
    """
    Predictive coding for EMG using muscle activation models.

    Leverages biomechanical models of muscle activation to predict
    EMG patterns and encode only prediction residuals.
    """

    name = "emg_predictive"

    def __init__(
        self,
        model_type: str = "ar",
        prediction_order: int = 8,
        adaptation_rate: float = 0.01,
        activation_modeling: bool = True
    ):
        """
        Initialize EMG predictive compressor.

        Parameters
        ----------
        model_type : str, default="ar"
            Prediction model type ("ar", "biomechanical")
        prediction_order : int, default=8
            Order of autoregressive prediction
        adaptation_rate : float, default=0.01
            Rate of model parameter adaptation
        activation_modeling : bool, default=True
            Use muscle activation state in prediction
        """
        super().__init__("emg_predictive")
        self.model_type = model_type
        self.prediction_order = prediction_order
        self.adaptation_rate = adaptation_rate
        self.activation_modeling = activation_modeling

        # Prediction model parameters
        self.ar_coefficients = np.zeros(prediction_order)
        self.prediction_history = np.zeros(prediction_order)

        # Muscle activation model parameters
        self.activation_state = 0.0
        self.activation_time_constant = 0.02  # 20ms activation dynamics

    def compress(self, data: np.ndarray) -> bytes:
        """Compress EMG using predictive coding."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self._last_shape = data.shape
        self._last_dtype = data.dtype

        compressed_channels = []

        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            compressed_ch = self._compress_predictive_channel(channel_data)
            compressed_channels.append(compressed_ch)

        import pickle
        return pickle.dumps({
            'shape': self._last_shape,
            'dtype': str(self._last_dtype),
            'channels': compressed_channels
        })

    def _compress_predictive_channel(self, data: np.ndarray) -> bytes:
        """Apply predictive compression to single channel."""
        residuals = []

        # Initialize prediction
        self._reset_prediction_state()

        for i in range(len(data)):
            # Generate prediction
            if i >= self.prediction_order:
                prediction = self._predict_next_sample(data, i)
            else:
                prediction = 0.0

            # Calculate residual
            residual = data[i] - prediction
            residuals.append(residual)

            # Update prediction model
            self._update_prediction_model(data, i)

        # Quantize and encode residuals
        residuals = np.array(residuals)
        quantized_residuals = self._quantize_residuals(residuals)

        return quantized_residuals.tobytes()

    def _predict_next_sample(self, data: np.ndarray, index: int) -> float:
        """Predict next sample using AR model and activation state."""
        # Standard AR prediction
        ar_prediction = np.dot(
            self.ar_coefficients,
            data[index-self.prediction_order:index][::-1]
        )

        if not self.activation_modeling:
            return ar_prediction

        # Add activation state prediction
        current_envelope = np.abs(data[index-1]) if index > 0 else 0.0

        # Update activation state (first-order dynamics)
        dt = 1.0 / 2000.0  # Assuming 2kHz sampling
        alpha = dt / self.activation_time_constant
        self.activation_state = (1 - alpha) * self.activation_state + alpha * current_envelope

        # Combine AR prediction with activation state
        activation_weight = 0.3
        combined_prediction = (1 - activation_weight) * ar_prediction + \
                            activation_weight * self.activation_state

        return combined_prediction

    def _update_prediction_model(self, data: np.ndarray, index: int):
        """Update AR coefficients using LMS adaptation."""
        if index < self.prediction_order:
            return

        # Get prediction input and target
        x = data[index-self.prediction_order:index][::-1]
        target = data[index]
        prediction = np.dot(self.ar_coefficients, x)

        # LMS update
        error = target - prediction
        self.ar_coefficients += self.adaptation_rate * error * x

    def _reset_prediction_state(self):
        """Reset prediction state for new channel."""
        self.ar_coefficients = np.zeros(self.prediction_order)
        self.prediction_history = np.zeros(self.prediction_order)
        self.activation_state = 0.0

    def _quantize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Quantize prediction residuals."""
        # Adaptive quantization based on residual statistics
        residual_std = np.std(residuals)

        # Use fewer bits for smaller residuals
        if residual_std < 0.01:
            n_bits = 8
        elif residual_std < 0.1:
            n_bits = 10
        else:
            n_bits = 12

        # Quantize
        levels = 2 ** n_bits
        residual_min, residual_max = np.min(residuals), np.max(residuals)
        residual_range = residual_max - residual_min

        if residual_range < 1e-8:
            return np.zeros(len(residuals), dtype=np.uint16)

        normalized = (residuals - residual_min) / residual_range
        quantized = np.round(normalized * (levels - 1)).astype(np.uint16)

        return quantized

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress predictively encoded EMG data."""
        import pickle
        data = pickle.loads(compressed_data)

        self._last_shape = data['shape']
        self._last_dtype = np.dtype(data['dtype'])

        # Placeholder implementation
        return np.random.randn(*self._last_shape).astype(self._last_dtype)


# Factory function for creating EMG compressors
def create_emg_compressor(
    compressor_type: str = "emg_lz",
    **kwargs
) -> BaseCompressor:
    """
    Create EMG-specific compressor.

    Parameters
    ----------
    compressor_type : str, default="emg_lz"
        Type of EMG compressor ("emg_lz", "emg_perceptual", "emg_predictive")
    **kwargs
        Additional parameters for the compressor

    Returns
    -------
    BaseCompressor
        Configured EMG compressor
    """
    if compressor_type == "emg_lz":
        return EMGLZCompressor(**kwargs)
    elif compressor_type == "emg_perceptual":
        return EMGPerceptualQuantizer(**kwargs)
    elif compressor_type == "emg_predictive":
        return EMGPredictiveCompressor(**kwargs)
    else:
        raise ValueError(f"Unknown EMG compressor type: {compressor_type}")
