"""
Advanced lossy compression methods for neural signals.

This module implements sophisticated lossy compression techniques
specifically designed for neural data, including perceptually-guided
quantization, adaptive wavelet compression, and neural network approaches.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


class PerceptualQuantizer:
    """
    Perceptually-guided quantization for neural signals.

    This quantizer adapts bit allocation based on the perceptual
    importance of different frequency bands and temporal regions
    in neural signals.
    """

    def __init__(
        self,
        base_bits: int = 12,
        frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        temporal_window: float = 0.1,  # 100ms windows
        sampling_rate: float = 30000.0
    ):
        """
        Initialize perceptual quantizer.

        Parameters
        ----------
        base_bits : int, default=12
            Base quantization bits
        frequency_bands : dict, optional
            Frequency bands with importance weights
        temporal_window : float, default=0.1
            Temporal window size in seconds
        sampling_rate : float, default=30000.0
            Sampling rate in Hz
        """
        self.base_bits = base_bits
        self.sampling_rate = sampling_rate
        self.temporal_window = temporal_window
        self.window_samples = int(temporal_window * sampling_rate)

        # Default frequency bands with perceptual importance
        if frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4.0, 0.7),      # (low, high, importance)
                'theta': (4.0, 8.0, 0.8),
                'alpha': (8.0, 13.0, 1.0),     # Most important
                'beta': (13.0, 30.0, 0.9),
                'gamma': (30.0, 100.0, 0.8),
                'high_gamma': (100.0, 300.0, 0.6)
            }
        else:
            self.frequency_bands = frequency_bands

    def _analyze_spectral_content(self, data: np.ndarray) -> np.ndarray:
        """
        Analyze spectral content to determine bit allocation.

        Parameters
        ----------
        data : np.ndarray
            Input neural data

        Returns
        -------
        np.ndarray
            Bit allocation map
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        n_windows = (n_samples + self.window_samples - 1) // self.window_samples

        bit_allocation = np.full((n_channels, n_windows), self.base_bits)

        for ch in range(n_channels):
            for w in range(n_windows):
                start_idx = w * self.window_samples
                end_idx = min((w + 1) * self.window_samples, n_samples)
                window_data = data[ch, start_idx:end_idx]

                if len(window_data) < 32:  # Too short for reliable analysis
                    continue

                # Compute power spectral density
                try:
                    freqs, psd = signal.welch(
                        window_data,
                        fs=self.sampling_rate,
                        nperseg=min(256, len(window_data))
                    )

                    # Calculate band powers and importance
                    total_importance = 0
                    weighted_power = 0

                    for band_name, (low_freq, high_freq, importance) in self.frequency_bands.items():
                        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        if np.any(band_mask):
                            band_power = np.mean(psd[band_mask])
                            weighted_power += band_power * importance
                            total_importance += importance

                    # Normalize and scale bit allocation
                    if total_importance > 0:
                        avg_weighted_power = weighted_power / total_importance
                        # Scale bits based on signal importance (6-16 bits range)
                        allocated_bits = self.base_bits + int(
                            4 * np.log10(max(avg_weighted_power, 1e-10) + 1)
                        )
                        bit_allocation[ch, w] = np.clip(allocated_bits, 6, 16)

                except Exception:
                    # Fallback to base bits if analysis fails
                    bit_allocation[ch, w] = self.base_bits

        return bit_allocation

    def _temporal_masking(self, data: np.ndarray) -> np.ndarray:
        """
        Apply temporal masking to reduce quantization in masked regions.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Temporal masking weights
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        masking_weights = np.ones_like(data)

        for ch in range(n_channels):
            # Detect high-energy transients
            envelope = np.abs(signal.hilbert(data[ch]))

            # Smooth envelope
            window_size = max(1, int(0.01 * self.sampling_rate))  # 10ms
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                smooth_envelope = np.convolve(envelope, kernel, mode='same')
            else:
                smooth_envelope = envelope

            # Find transients (high-energy events)
            threshold = np.percentile(smooth_envelope, 90)
            transient_mask = smooth_envelope > threshold

            # Reduce quantization precision in regions following transients
            # (temporal masking effect)
            for i in range(n_samples):
                if transient_mask[i]:
                    # Masking window after transient
                    mask_window = int(0.05 * self.sampling_rate)  # 50ms
                    end_idx = min(i + mask_window, n_samples)
                    masking_weights[ch, i:end_idx] *= 0.7  # Reduce precision

        return masking_weights

    def quantize(self, data: np.ndarray, quality_level: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Apply perceptual quantization to neural data.

        Parameters
        ----------
        data : np.ndarray
            Input neural data
        quality_level : float, default=1.0
            Quality level (0.1 to 1.0)

        Returns
        -------
        tuple
            (quantized_data, quantization_info)
        """
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape

        # Analyze spectral content for bit allocation
        bit_allocation = self._analyze_spectral_content(data)

        # Apply temporal masking
        masking_weights = self._temporal_masking(data)

        # Scale bit allocation by quality level
        scaled_bits = bit_allocation * quality_level
        scaled_bits = np.clip(scaled_bits, 4, 16)

        # Quantize data
        quantized_data = np.zeros_like(data)
        quantization_info = {
            'bit_allocation': bit_allocation,
            'masking_weights': masking_weights,
            'data_range': {},
            'quality_level': quality_level,
            'original_shape': original_shape
        }

        for ch in range(n_channels):
            channel_data = data[ch]
            data_min, data_max = np.min(channel_data), np.max(channel_data)
            quantization_info['data_range'][ch] = (data_min, data_max)

            # Apply window-based quantization
            n_windows = bit_allocation.shape[1]

            for w in range(n_windows):
                start_idx = w * self.window_samples
                end_idx = min((w + 1) * self.window_samples, n_samples)
                window_data = channel_data[start_idx:end_idx]
                window_mask = masking_weights[ch, start_idx:end_idx]

                if len(window_data) == 0:
                    continue

                # Effective bits considering masking
                effective_bits = int(scaled_bits[ch, w] * np.mean(window_mask))
                effective_bits = max(4, effective_bits)
                n_levels = 2 ** effective_bits

                # Quantize window
                if data_max > data_min:
                    normalized = (window_data - data_min) / (data_max - data_min)
                    quantized_window = np.round(normalized * (n_levels - 1))
                    quantized_window = quantized_window / (n_levels - 1)
                    quantized_data[ch, start_idx:end_idx] = (
                        quantized_window * (data_max - data_min) + data_min
                    )
                else:
                    quantized_data[ch, start_idx:end_idx] = window_data

        return quantized_data.reshape(original_shape), quantization_info


class AdaptiveWaveletCompressor:
    """
    Adaptive wavelet compression for neural signals.

    This compressor uses multi-resolution wavelet analysis with
    adaptive coefficient thresholding based on neural signal
    characteristics.
    """

    def __init__(
        self,
        wavelet: str = 'db8',
        decomposition_levels: int = 6,
        threshold_mode: str = 'adaptive',
        preserve_bands: Optional[List[str]] = None
    ):
        """
        Initialize adaptive wavelet compressor.

        Parameters
        ----------
        wavelet : str, default='db8'
            Wavelet type to use
        decomposition_levels : int, default=6
            Number of decomposition levels
        threshold_mode : str, default='adaptive'
            Thresholding mode ('adaptive', 'global', 'band_specific')
        preserve_bands : list, optional
            Frequency bands to preserve with higher quality
        """
        self.wavelet = wavelet
        self.decomposition_levels = decomposition_levels
        self.threshold_mode = threshold_mode

        if preserve_bands is None:
            self.preserve_bands = ['alpha', 'beta', 'gamma']
        else:
            self.preserve_bands = preserve_bands

        # Band-specific preservation factors
        self.band_preservation = {
            'delta': 0.3,
            'theta': 0.5,
            'alpha': 0.9,    # High preservation
            'beta': 0.8,
            'gamma': 0.7,
            'high_gamma': 0.4
        }

    def _estimate_noise_variance(self, coeffs: List[np.ndarray]) -> float:
        """
        Estimate noise variance from wavelet coefficients.

        Parameters
        ----------
        coeffs : list
            Wavelet coefficients

        Returns
        -------
        float
            Estimated noise variance
        """
        # Use finest detail coefficients for noise estimation
        detail_coeffs = coeffs[-1]  # Highest frequency details

        # Robust estimation using median absolute deviation
        mad = np.median(np.abs(detail_coeffs))
        sigma = mad / 0.6745  # Convert MAD to standard deviation

        return sigma ** 2

    def _adaptive_threshold(
        self,
        coeffs: List[np.ndarray],
        compression_ratio: float = 0.1
    ) -> List[float]:
        """
        Calculate adaptive thresholds for each decomposition level.

        Parameters
        ----------
        coeffs : list
            Wavelet coefficients
        compression_ratio : float, default=0.1
            Target compression ratio (fraction of coefficients to keep)

        Returns
        -------
        list
            Thresholds for each level
        """
        thresholds = []

        # Estimate noise level
        noise_var = self._estimate_noise_variance(coeffs)
        base_threshold = np.sqrt(2 * noise_var * np.log(len(coeffs[0])))

        for level, coeff in enumerate(coeffs[1:], 1):  # Skip approximation
            # Calculate level-specific threshold
            if self.threshold_mode == 'adaptive':
                # Energy-based adaptive threshold
                coeff_energy = np.var(coeff)
                snr_estimate = coeff_energy / (noise_var + 1e-10)

                # Scale threshold based on SNR
                if snr_estimate > 10:  # High SNR
                    level_threshold = base_threshold * 0.5
                elif snr_estimate > 2:  # Medium SNR
                    level_threshold = base_threshold
                else:  # Low SNR
                    level_threshold = base_threshold * 2.0

            elif self.threshold_mode == 'global':
                # Global threshold based on compression ratio
                level_threshold = np.percentile(
                    np.abs(coeff),
                    (1 - compression_ratio) * 100
                )

            else:  # band_specific
                # Band-specific thresholding based on frequency content
                band_factor = 1.0
                if level <= 2:  # High frequency bands
                    band_factor = 0.7
                elif level <= 4:  # Medium frequency bands
                    band_factor = 0.5
                else:  # Low frequency bands
                    band_factor = 0.3

                level_threshold = base_threshold * band_factor

            thresholds.append(level_threshold)

        return thresholds

    def compress(self, data: np.ndarray, compression_ratio: float = 0.1,
                 quality_factor: float = 1.0) -> Tuple[List, Dict]:
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets (pywt) required for wavelet compression")

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        compressed_channels = []
        channel_metadata = []

        for ch in range(n_channels):
            channel_data = data[ch]

            # Wavelet decomposition
            coeffs = pywt.wavedec(
                channel_data,
                self.wavelet,
                level=self.decomposition_levels,
                mode='symmetric'
            )

            # Calculate adaptive thresholds
            thresholds = self._adaptive_threshold(coeffs, compression_ratio)

            # Apply thresholding with quality factor
            compressed_coeffs = [coeffs[0]]  # Keep approximation coefficients

            for i, (coeff, threshold) in enumerate(zip(coeffs[1:], thresholds)):
                # Adjust threshold by quality factor
                adjusted_threshold = threshold / quality_factor

                # Soft thresholding
                compressed_coeff = pywt.threshold(
                    coeff,
                    adjusted_threshold,
                    mode='soft'
                )
                compressed_coeffs.append(compressed_coeff)

            # Store compressed coefficients and metadata
            compressed_channels.append(compressed_coeffs)

            # Calculate compression statistics
            total_coeffs = sum(len(c) for c in coeffs)
            nonzero_coeffs = sum(np.count_nonzero(c) for c in compressed_coeffs)
            actual_compression = nonzero_coeffs / total_coeffs if total_coeffs > 0 else 0

            channel_metadata.append({
                'wavelet': self.wavelet,
                'levels': self.decomposition_levels,
                'thresholds': thresholds,
                'original_length': len(channel_data),
                'total_coeffs': total_coeffs,
                'nonzero_coeffs': nonzero_coeffs,
                'compression_ratio': actual_compression
            })

        metadata = {
            'original_shape': original_shape,
            'n_channels': n_channels,
            'channel_metadata': channel_metadata,
            'compression_ratio': compression_ratio,
            'quality_factor': quality_factor
        }

        return compressed_channels, metadata

    def decompress(self, compressed_coeffs: List, metadata: Dict) -> np.ndarray:
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets (pywt) required for wavelet decompression")

        n_channels = metadata['n_channels']
        metadata['original_shape']

        decompressed_channels = []

        for ch in range(n_channels):
            ch_coeffs = compressed_coeffs[ch]
            ch_metadata = metadata['channel_metadata'][ch]

            # Wavelet reconstruction
            reconstructed = pywt.waverec(
                ch_coeffs,
                ch_metadata['wavelet'],
                mode='symmetric'
            )

            # Ensure correct length
            original_length = ch_metadata['original_length']
            if len(reconstructed) > original_length:
                reconstructed = reconstructed[:original_length]
            elif len(reconstructed) < original_length:
                # Pad with zeros if necessary
                padding = np.zeros(original_length - len(reconstructed))
                reconstructed = np.concatenate([reconstructed, padding])

            decompressed_channels.append(reconstructed)

        # Reshape to original format
        decompressed_data = np.array(decompressed_channels)
        try:
            if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                decompressed_data = decompressed_data.reshape(self._last_shape)
                decompressed_data = decompressed_data.astype(self._last_dtype)
                # self._check_integrity(np.zeros(self._last_shape,
                # dtype=self._last_dtype), decompressed_data, check_shape=True,
                # check_dtype=True, check_hash=False) # This line was not in the new_code,
                # so it's removed.
        except Exception as e:
            raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
        return decompressed_data.reshape(metadata['original_shape'])


class NeuralAutoencoder:
    """
    Neural network-based autoencoder for neural signal compression.

    This class implements a deep autoencoder specifically designed
    for neural signal compression with configurable architectures.
    """

    def __init__(
        self,
        input_size: int = 1024,
        encoding_size: int = 128,
        hidden_layers: List[int] = None,
        activation: str = 'relu',
        learning_rate: float = 0.001
    ):
        """
        Initialize neural autoencoder.

        Parameters
        ----------
        input_size : int, default=1024
            Size of input segments
        encoding_size : int, default=128
            Size of compressed encoding
        hidden_layers : list, optional
            Hidden layer sizes
        activation : str, default='relu'
            Activation function
        learning_rate : float, default=0.001
            Learning rate for training
        """
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.learning_rate = learning_rate

        if hidden_layers is None:
            # Default architecture
            self.hidden_layers = [512, 256]
        else:
            self.hidden_layers = hidden_layers

        self.model = None
        self.is_trained = False

        # Compression statistics
        self.training_loss = []
        self.compression_ratio = input_size / encoding_size

    def _build_model(self):
        """Build the autoencoder model."""
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for neural autoencoder")

        # Encoder layers
        encoder_layers = []
        current_size = self.input_size

        for hidden_size in self.hidden_layers:
            encoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size

        # Encoding layer
        encoder_layers.append(nn.Linear(current_size, self.encoding_size))

        # Decoder layers (reverse of encoder)
        decoder_layers = []
        current_size = self.encoding_size

        for hidden_size in reversed(self.hidden_layers):
            decoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size

        # Output layer
        decoder_layers.append(nn.Linear(current_size, self.input_size))

        # Complete model
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        class Autoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

            def encode(self, x):
                return self.encoder(x)

            def decode(self, encoded):
                return self.decoder(encoded)

        self.model = Autoencoder(self.encoder, self.decoder)

    def train(
        self,
        training_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the autoencoder on neural data.

        Parameters
        ----------
        training_data : np.ndarray
            Training neural data
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Fraction of data for validation

        Returns
        -------
        dict
            Training statistics
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required for neural autoencoder training")

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Prepare data segments
        segments = self._create_segments(training_data)

        # Split into train/validation
        n_samples = len(segments)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        train_segments = segments[:n_train]
        val_segments = segments[n_train:]

        # Create data loaders
        train_tensor = torch.FloatTensor(train_segments)
        val_tensor = torch.FloatTensor(val_segments)

        train_dataset = TensorDataset(train_tensor, train_tensor)  # Autoencoder targets = inputs
        val_dataset = TensorDataset(val_tensor, val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_data, batch_targets in train_loader:
                optimizer.zero_grad()
                encoded, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    encoded, decoded = self.model(batch_data)
                    loss = criterion(decoded, batch_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        self.is_trained = True
        self.training_loss = train_losses

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'epochs': epochs
        }

    def _create_segments(self, data: np.ndarray) -> np.ndarray:
        """Create fixed-size segments from neural data."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        n_segments_per_channel = n_samples // self.input_size

        segments = []
        for ch in range(n_channels):
            for i in range(n_segments_per_channel):
                start_idx = i * self.input_size
                end_idx = start_idx + self.input_size
                segment = data[ch, start_idx:end_idx]
                segments.append(segment)

        return np.array(segments)

    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for neural autoencoder")

        original_shape = data.shape
        segments = self._create_segments(data)

        # Encode segments
        self.model.eval()
        encodings = []

        with torch.no_grad():
            for segment in segments:
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0)
                encoding = self.model.encode(segment_tensor)
                encodings.append(encoding.numpy())

        compressed_data = np.array(encodings).squeeze()

        metadata = {
            'original_shape': original_shape,
            'input_size': self.input_size,
            'encoding_size': self.encoding_size,
            'n_segments': len(segments),
            'compression_ratio': self.compression_ratio
        }

        return compressed_data, metadata

    def decompress(self, compressed_data: np.ndarray, metadata: Dict) -> np.ndarray:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for neural autoencoder")

        # Decode segments
        self.model.eval()
        decoded_segments = []

        with torch.no_grad():
            for encoding in compressed_data:
                if encoding.ndim == 1:
                    encoding = encoding.reshape(1, -1)
                encoding_tensor = torch.FloatTensor(encoding)
                decoded = self.model.decode(encoding_tensor)
                decoded_segments.append(decoded.numpy().squeeze())

        # Reconstruct original data structure
        original_shape = metadata['original_shape']

        if len(original_shape) == 1:
            # Single channel data
            reconstructed = np.concatenate(decoded_segments)
            return reconstructed[:original_shape[0]]
        else:
            # Multi-channel data
            n_channels, n_samples = original_shape
            segments_per_channel = len(decoded_segments) // n_channels

            reconstructed = np.zeros(original_shape)
            segment_idx = 0

            for ch in range(n_channels):
                channel_data = []
                for _ in range(segments_per_channel):
                    channel_data.append(decoded_segments[segment_idx])
                    segment_idx += 1

                channel_signal = np.concatenate(channel_data)
                reconstructed[ch] = channel_signal[:n_samples]

            decoded_data = np.array(decoded_segments)
            try:
                if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                    decoded_data = decoded_data.reshape(self._last_shape)
                    decoded_data = decoded_data.astype(self._last_dtype)
                    # self._check_integrity(np.zeros(self._last_shape,
                    # dtype=self._last_dtype), decoded_data, check_shape=True,
                    # check_dtype=True, check_hash=False) # This line was not in the new_code,
                    # so it's removed.
            except Exception as e:
                raise ValueError(f"Failed to reshape or cast decompressed data: {e}")
            return decoded_data.reshape(original_shape)


def create_lossy_compressor_suite(
    compression_mode: str = 'balanced'
) -> Dict:
    """
    Create a complete suite of lossy compression methods.

    Parameters
    ----------
    compression_mode : str, default='balanced'
        Compression mode ('fast', 'balanced', 'quality')

    Returns
    -------
    dict
        Dictionary of configured compressors
    """
    if compression_mode == 'fast':
        quantizer = PerceptualQuantizer(base_bits=8, temporal_window=0.2)
        wavelet = AdaptiveWaveletCompressor(decomposition_levels=4)
        autoencoder = NeuralAutoencoder(input_size=512, encoding_size=64)
    elif compression_mode == 'quality':
        quantizer = PerceptualQuantizer(base_bits=14, temporal_window=0.05)
        wavelet = AdaptiveWaveletCompressor(decomposition_levels=8)
        autoencoder = NeuralAutoencoder(input_size=2048, encoding_size=256)
    else:  # balanced
        quantizer = PerceptualQuantizer()
        wavelet = AdaptiveWaveletCompressor()
        autoencoder = NeuralAutoencoder()

    return {
        'perceptual_quantizer': quantizer,
        'adaptive_wavelet': wavelet,
        'neural_autoencoder': autoencoder
    }
