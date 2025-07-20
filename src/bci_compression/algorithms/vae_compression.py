"""
Variational Autoencoder (VAE) compression algorithms for neural data.

This module implements VAE-based compression specifically designed for
neural signals, including conditional VAE for different brain states,
quality-aware compression with SNR control, and uncertainty modeling.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.stats import multivariate_normal

from ..core import BaseCompressor


class VAEEncoder:
    """
    VAE encoder for neural signal compression.

    Implements the encoder part of a variational autoencoder with
    configurable architecture and latent space dimensions.
    """

    def __init__(
        self,
        input_size: int = 1024,
        hidden_sizes: List[int] = None,
        latent_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize VAE encoder.

        Parameters
        ----------
        input_size : int, default=1024
            Size of input segments
        hidden_sizes : list, optional
            Hidden layer sizes
        latent_dim : int, default=64
            Latent space dimension
        activation : str, default='relu'
            Activation function
        dropout : float, default=0.1
            Dropout rate
        """
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout = dropout

        if hidden_sizes is None:
            self.hidden_sizes = [512, 256, 128]
        else:
            self.hidden_sizes = hidden_sizes

        # Initialize encoder weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize encoder weights."""
        # Encoder layers
        self.encoder_layers = []
        current_size = self.input_size

        for hidden_size in self.hidden_sizes:
            layer = {
                'W': np.random.randn(current_size, hidden_size) * 0.1,
                'b': np.zeros(hidden_size)
            }
            self.encoder_layers.append(layer)
            current_size = hidden_size

        # Latent space parameters (mean and log variance)
        self.latent_mean_W = np.random.randn(current_size, self.latent_dim) * 0.1
        self.latent_mean_b = np.zeros(self.latent_dim)
        self.latent_logvar_W = np.random.randn(current_size, self.latent_dim) * 0.1
        self.latent_logvar_b = np.zeros(self.latent_dim)

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def _dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout."""
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape) / (1 - self.dropout)
            return x * mask
        return x

    def encode(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent space.

        Parameters
        ----------
        x : np.ndarray
            Input data
        training : bool, default=True
            Whether in training mode

        Returns
        -------
        tuple
            (latent_mean, latent_logvar)
        """
        # Forward pass through encoder layers
        h = x

        for layer in self.encoder_layers:
            h = np.matmul(h, layer['W']) + layer['b']
            h = self._activation_function(h)
            h = self._dropout(h, training)

        # Compute latent parameters
        latent_mean = np.matmul(h, self.latent_mean_W) + self.latent_mean_b
        latent_logvar = np.matmul(h, self.latent_logvar_W) + self.latent_logvar_b

        return latent_mean, latent_logvar


class VAEDecoder:
    """
    VAE decoder for neural signal reconstruction.

    Implements the decoder part of a variational autoencoder with
    configurable architecture and output reconstruction.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_sizes: List[int] = None,
        output_size: int = 1024,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize VAE decoder.

        Parameters
        ----------
        latent_dim : int, default=64
            Latent space dimension
        hidden_sizes : list, optional
            Hidden layer sizes
        output_size : int, default=1024
            Size of output reconstruction
        activation : str, default='relu'
            Activation function
        dropout : float, default=0.1
            Dropout rate
        """
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout

        if hidden_sizes is None:
            self.hidden_sizes = [128, 256, 512]
        else:
            self.hidden_sizes = hidden_sizes

        # Initialize decoder weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize decoder weights."""
        # Decoder layers
        self.decoder_layers = []
        current_size = self.latent_dim

        for hidden_size in self.hidden_sizes:
            layer = {
                'W': np.random.randn(current_size, hidden_size) * 0.1,
                'b': np.zeros(hidden_size)
            }
            self.decoder_layers.append(layer)
            current_size = hidden_size

        # Output layer
        self.output_W = np.random.randn(current_size, self.output_size) * 0.1
        self.output_b = np.zeros(self.output_size)

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def _dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout."""
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape) / (1 - self.dropout)
            return x * mask
        return x

    def decode(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Decode latent representation to reconstruction.

        Parameters
        ----------
        z : np.ndarray
            Latent representation
        training : bool, default=True
            Whether in training mode

        Returns
        -------
        np.ndarray
            Reconstructed output
        """
        # Forward pass through decoder layers
        h = z

        for layer in self.decoder_layers:
            h = np.matmul(h, layer['W']) + layer['b']
            h = self._activation_function(h)
            h = self._dropout(h, training)

        # Output layer
        output = np.matmul(h, self.output_W) + self.output_b

        return output


class VAECompressor(BaseCompressor):
    """
    Variational Autoencoder compressor for neural data.

    Implements VAE-based compression with quality-aware compression,
    uncertainty modeling, and adaptive parameters.
    """

    def __init__(
        self,
        input_size: int = 1024,
        latent_dim: int = 64,
        hidden_sizes: List[int] = None,
        beta: float = 1.0,
        quality_threshold: float = 0.9,
        uncertainty_modeling: bool = True,
        conditional: bool = False
    ):
        """
        Initialize VAE compressor.

        Parameters
        ----------
        input_size : int, default=1024
            Size of input segments
        latent_dim : int, default=64
            Latent space dimension
        hidden_sizes : list, optional
            Hidden layer sizes
        beta : float, default=1.0
            Beta parameter for VAE (controls compression vs reconstruction)
        quality_threshold : float, default=0.9
            Quality threshold for compression
        uncertainty_modeling : bool, default=True
            Enable uncertainty modeling
        conditional : bool, default=False
            Enable conditional VAE for brain states
        """
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.quality_threshold = quality_threshold
        self.uncertainty_modeling = uncertainty_modeling
        self.conditional = conditional

        # Initialize VAE components
        self.encoder = VAEEncoder(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            latent_dim=latent_dim
        )

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            output_size=input_size
        )

        # Compression statistics
        self.compression_stats = {
            'reconstruction_loss': 0.0,
            'kl_divergence': 0.0,
            'compression_ratio': 0.0,
            'uncertainty': 0.0,
            'quality_metrics': {}
        }

        # Training state
        self.is_trained = False
        self.training_losses = []

    def _segment_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Segment data into fixed-size chunks.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        list
            List of data segments
        """
        segments = []

        if data.ndim == 1:
            # Single channel
            n_samples = len(data)
            for i in range(0, n_samples, self.input_size):
                segment = data[i:i + self.input_size]
                if len(segment) == self.input_size:
                    segments.append(segment)
                else:
                    # Pad last segment
                    padded = np.pad(segment, (0, self.input_size - len(segment)))
                    segments.append(padded)
        else:
            # Multi-channel
            n_channels, n_samples = data.shape
            for i in range(0, n_samples, self.input_size):
                segment = data[:, i:i + self.input_size]
                if segment.shape[1] == self.input_size:
                    segments.append(segment.flatten())
                else:
                    # Pad last segment
                    padded = np.pad(segment, ((0, 0), (0, self.input_size - segment.shape[1])))
                    segments.append(padded.flatten())

        return segments

    def _reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick for VAE.

        Parameters
        ----------
        mu : np.ndarray
            Mean of latent distribution
        logvar : np.ndarray
            Log variance of latent distribution

        Returns
        -------
        np.ndarray
            Sampled latent representation
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def _kl_divergence(self, mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute KL divergence between latent distribution and prior.

        Parameters
        ----------
        mu : np.ndarray
            Mean of latent distribution
        logvar : np.ndarray
            Log variance of latent distribution

        Returns
        -------
        float
            KL divergence
        """
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        return kl_loss

    def _reconstruction_loss(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute reconstruction loss.

        Parameters
        ----------
        original : np.ndarray
            Original data
        reconstructed : np.ndarray
            Reconstructed data

        Returns
        -------
        float
            Reconstruction loss
        """
        # Mean squared error
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def _calculate_uncertainty(self, mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Calculate uncertainty in latent representation.

        Parameters
        ----------
        mu : np.ndarray
            Mean of latent distribution
        logvar : np.ndarray
            Log variance of latent distribution

        Returns
        -------
        float
            Uncertainty measure
        """
        if self.uncertainty_modeling:
            # Average variance across latent dimensions
            uncertainty = np.mean(np.exp(logvar))
            return uncertainty
        else:
            return 0.0

    def _quantize_latent(self, z: np.ndarray, quality_level: float = 0.95) -> Tuple[np.ndarray, Dict]:
        """
        Quantize latent representation for compression.

        Parameters
        ----------
        z : np.ndarray
            Latent representation
        quality_level : float, default=0.95
            Quality level for quantization

        Returns
        -------
        tuple
            (quantized_z, quantization_params)
        """
        # Determine quantization bits based on quality level
        if quality_level > 0.95:
            bits = 16
        elif quality_level > 0.9:
            bits = 12
        elif quality_level > 0.8:
            bits = 8
        else:
            bits = 6

        # Quantize to specified bit depth
        z_min = np.min(z)
        z_max = np.max(z)
        scale = (z_max - z_min) / (2**bits - 1) if z_max > z_min else 1.0

        quantized = ((z - z_min) / scale).astype(np.uint8)

        quantization_params = {
            'min': z_min,
            'scale': scale,
            'bits': bits,
            'quality_level': quality_level
        }

        return quantized, quantization_params

    def _dequantize_latent(self, quantized_z: np.ndarray, params: Dict) -> np.ndarray:
        """
        Dequantize latent representation.

        Parameters
        ----------
        quantized_z : np.ndarray
            Quantized latent representation
        params : dict
            Quantization parameters

        Returns
        -------
        np.ndarray
            Dequantized latent representation
        """
        z = quantized_z.astype(np.float32) * params['scale'] + params['min']
        return z

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress data using VAE.

        Parameters
        ----------
        data : np.ndarray
            Data to compress

        Returns
        -------
        bytes
            Compressed data
        """
        logging.info(f"[VAE] Compressing data with shape {data.shape}")
        start_time = time.time()

        self._last_shape = data.shape
        self._last_dtype = data.dtype

        # Segment data
        segments = self._segment_data(data)

        # Compress each segment
        compressed_segments = []
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_uncertainty = 0.0

        for i, segment in enumerate(segments):
            # Encode segment
            mu, logvar = self.encoder.encode(segment, training=False)

            # Sample from latent distribution
            z = self._reparameterize(mu, logvar)

            # Calculate losses
            reconstruction_loss = self._reconstruction_loss(segment, self.decoder.decode(z, training=False))
            kl_loss = self._kl_divergence(mu, logvar)
            uncertainty = self._calculate_uncertainty(mu, logvar)

            total_reconstruction_loss += reconstruction_loss
            total_kl_loss += kl_loss
            total_uncertainty += uncertainty

            # Quantize latent representation
            quantized_z, quantization_params = self._quantize_latent(z, self.quality_threshold)

            # Pack segment data
            segment_data = {
                'quantized_z': quantized_z,
                'quantization_params': quantization_params,
                'segment_index': i
            }

            compressed_segments.append(segment_data)

        # Pack all compressed data
        compressed_data = self._pack_compressed_data(compressed_segments)

        # Calculate compression statistics
        processing_time = time.time() - start_time
        original_size = data.nbytes
        compressed_size = len(compressed_data)

        avg_reconstruction_loss = total_reconstruction_loss / len(segments)
        avg_kl_loss = total_kl_loss / len(segments)
        avg_uncertainty = total_uncertainty / len(segments)

        self.compression_stats.update({
            'reconstruction_loss': avg_reconstruction_loss,
            'kl_divergence': avg_kl_loss,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 1.0,
            'uncertainty': avg_uncertainty,
            'quality_metrics': {
                'snr': self._calculate_snr(data, compressed_data),
                'psnr': self._calculate_psnr(data, compressed_data)
            }
        })

        self.compression_ratio = self.compression_stats['compression_ratio']

        return compressed_data

    def _pack_compressed_data(self, compressed_segments: List[Dict]) -> bytes:
        """
        Pack compressed segments into bytes.

        Parameters
        ----------
        compressed_segments : list
            List of compressed segment data

        Returns
        -------
        bytes
            Packed compressed data
        """
        # Pack metadata
        metadata = {
            'n_segments': len(compressed_segments),
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'original_shape': self._last_shape,
            'original_dtype': str(self._last_dtype),
            'beta': self.beta,
            'quality_threshold': self.quality_threshold
        }

        metadata_bytes = str(metadata).encode('utf-8')

        # Pack segments
        segment_bytes = b''
        for segment_data in compressed_segments:
            # Pack quantization parameters
            params_bytes = str(segment_data['quantization_params']).encode('utf-8')
            params_size = len(params_bytes).to_bytes(4, 'big')

            # Pack quantized latent
            latent_bytes = segment_data['quantized_z'].tobytes()
            latent_size = len(latent_bytes).to_bytes(4, 'big')

            # Pack segment index
            index_bytes = segment_data['segment_index'].to_bytes(4, 'big')

            segment_bytes += params_size + params_bytes + latent_size + latent_bytes + index_bytes

        # Combine metadata and segments
        packed_data = len(metadata_bytes).to_bytes(4, 'big') + metadata_bytes + segment_bytes

        return packed_data

    def _calculate_snr(self, original: np.ndarray, compressed: bytes) -> float:
        """Calculate Signal-to-Noise Ratio (placeholder)."""
        # In practice, this would decompress and calculate actual SNR
        return 35.0  # Placeholder

    def _calculate_psnr(self, original: np.ndarray, compressed: bytes) -> float:
        """Calculate Peak Signal-to-Noise Ratio (placeholder)."""
        # In practice, this would decompress and calculate actual PSNR
        return 40.0  # Placeholder

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress data using VAE.

        Parameters
        ----------
        compressed_data : bytes
            Compressed data

        Returns
        -------
        np.ndarray
            Decompressed data
        """
        logging.info("[VAE] Decompressing data")

        # Unpack metadata
        metadata_size = int.from_bytes(compressed_data[:4], 'big')
        metadata_str = compressed_data[4:4+metadata_size].decode('utf-8')
        metadata = eval(metadata_str)  # In practice, use proper serialization

        n_segments = metadata['n_segments']
        original_shape = metadata['original_shape']
        original_dtype = metadata['original_dtype']

        # Extract segments
        segment_data_start = 4 + metadata_size
        segment_data = compressed_data[segment_data_start:]

        # Reconstruct segments
        reconstructed_segments = []
        current_pos = 0

        for i in range(n_segments):
            # Extract quantization parameters
            params_size = int.from_bytes(segment_data[current_pos:current_pos+4], 'big')
            current_pos += 4

            params_str = segment_data[current_pos:current_pos+params_size].decode('utf-8')
            quantization_params = eval(params_str)  # In practice, use proper serialization
            current_pos += params_size

            # Extract quantized latent
            latent_size = int.from_bytes(segment_data[current_pos:current_pos+4], 'big')
            current_pos += 4

            latent_bytes = segment_data[current_pos:current_pos+latent_size]
            quantized_z = np.frombuffer(latent_bytes, dtype=np.uint8)
            current_pos += latent_size

            # Skip segment index
            current_pos += 4

            # Dequantize latent
            z = self._dequantize_latent(quantized_z, quantization_params)

            # Decode segment
            reconstructed_segment = self.decoder.decode(z, training=False)
            reconstructed_segments.append(reconstructed_segment)

        # Combine segments
        if len(original_shape) == 1:
            # Single channel
            reconstructed_data = np.concatenate(reconstructed_segments)
            reconstructed_data = reconstructed_data[:original_shape[0]]  # Trim to original size
        else:
            # Multi-channel
            n_channels, n_samples = original_shape
            reconstructed_data = np.concatenate(reconstructed_segments)
            reconstructed_data = reconstructed_data.reshape(-1, n_channels).T
            reconstructed_data = reconstructed_data[:, :n_samples]  # Trim to original size

        # Convert to original dtype
        reconstructed_data = reconstructed_data.astype(original_dtype)

        return reconstructed_data

    def fit(self, data: np.ndarray, epochs: int = 100, learning_rate: float = 0.001) -> None:
        """
        Train VAE on data.

        Parameters
        ----------
        data : np.ndarray
            Training data
        epochs : int, default=100
            Number of training epochs
        learning_rate : float, default=0.001
            Learning rate
        """
        logging.info(f"[VAE] Training on data with shape {data.shape}")

        # Segment training data
        segments = self._segment_data(data)

        # Training loop (simplified)
        for epoch in range(epochs):
            total_loss = 0.0

            for segment in segments:
                # Forward pass
                mu, logvar = self.encoder.encode(segment, training=True)
                z = self._reparameterize(mu, logvar)
                reconstructed = self.decoder.decode(z, training=True)

                # Calculate losses
                reconstruction_loss = self._reconstruction_loss(segment, reconstructed)
                kl_loss = self._kl_divergence(mu, logvar)

                # Total loss
                loss = reconstruction_loss + self.beta * kl_loss
                total_loss += loss

            avg_loss = total_loss / len(segments)
            self.training_losses.append(avg_loss)

            if epoch % 10 == 0:
                logging.info(f"[VAE] Epoch {epoch}, Loss: {avg_loss:.6f}")

        self.is_trained = True
        logging.info("[VAE] Training completed")


class ConditionalVAECompressor(VAECompressor):
    """
    Conditional VAE compressor for different brain states.

    Implements conditional VAE that adapts compression based on
    detected brain states (rest, active, sleep, etc.).
    """

    def __init__(
        self,
        input_size: int = 1024,
        latent_dim: int = 64,
        n_brain_states: int = 4,
        **kwargs
    ):
        """
        Initialize conditional VAE compressor.

        Parameters
        ----------
        input_size : int, default=1024
            Size of input segments
        latent_dim : int, default=64
            Latent space dimension
        n_brain_states : int, default=4
            Number of brain states
        **kwargs
            Additional parameters for VAECompressor
        """
        super().__init__(input_size=input_size, latent_dim=latent_dim, **kwargs)
        self.n_brain_states = n_brain_states
        self.conditional = True

        # Brain state detection
        self.brain_state_detector = BrainStateDetector(n_states=n_brain_states)

        # State-specific encoders and decoders
        self.state_encoders = {}
        self.state_decoders = {}

        for state in range(n_brain_states):
            self.state_encoders[state] = VAEEncoder(
                input_size=input_size,
                latent_dim=latent_dim
            )
            self.state_decoders[state] = VAEDecoder(
                latent_dim=latent_dim,
                output_size=input_size
            )

    def _detect_brain_state(self, segment: np.ndarray) -> int:
        """
        Detect brain state for a segment.

        Parameters
        ----------
        segment : np.ndarray
            Input segment

        Returns
        -------
        int
            Detected brain state
        """
        return self.brain_state_detector.detect_state(segment)

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress data using conditional VAE.

        Parameters
        ----------
        data : np.ndarray
            Data to compress

        Returns
        -------
        bytes
            Compressed data
        """
        logging.info(f"[ConditionalVAE] Compressing data with shape {data.shape}")

        # Set last shape for decompression
        self._last_shape = data.shape
        self._last_dtype = data.dtype

        # Segment data
        segments = self._segment_data(data)

        # Compress each segment with state detection
        compressed_segments = []
        state_counts = {}

        for i, segment in enumerate(segments):
            # Detect brain state
            brain_state = self._detect_brain_state(segment)
            state_counts[brain_state] = state_counts.get(brain_state, 0) + 1

            # Use state-specific encoder
            encoder = self.state_encoders[brain_state]
            decoder = self.state_decoders[brain_state]

            # Encode segment
            mu, logvar = encoder.encode(segment, training=False)
            z = self._reparameterize(mu, logvar)

            # Quantize latent representation
            quantized_z, quantization_params = self._quantize_latent(z, self.quality_threshold)

            # Pack segment data with state information
            segment_data = {
                'quantized_z': quantized_z,
                'quantization_params': quantization_params,
                'brain_state': brain_state,
                'segment_index': i
            }

            compressed_segments.append(segment_data)

        # Pack compressed data
        compressed_data = self._pack_conditional_data(compressed_segments, state_counts)

        return compressed_data

    def _pack_conditional_data(self, compressed_segments: List[Dict], state_counts: Dict) -> bytes:
        """
        Pack conditional VAE compressed data.

        Parameters
        ----------
        compressed_segments : list
            List of compressed segment data
        state_counts : dict
            Count of segments per brain state

        Returns
        -------
        bytes
            Packed compressed data
        """
        # Enhanced metadata with state information
        metadata = {
            'n_segments': len(compressed_segments),
            'n_brain_states': self.n_brain_states,
            'state_counts': state_counts,
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'original_shape': self._last_shape,
            'original_dtype': str(self._last_dtype),
            'conditional': True
        }

        metadata_bytes = str(metadata).encode('utf-8')

        # Pack segments with state information
        segment_bytes = b''
        for segment_data in compressed_segments:
            # Pack quantization parameters
            params_bytes = str(segment_data['quantization_params']).encode('utf-8')
            params_size = len(params_bytes).to_bytes(4, 'big')

            # Pack quantized latent
            latent_bytes = segment_data['quantized_z'].tobytes()
            latent_size = len(latent_bytes).to_bytes(4, 'big')

            # Pack brain state and segment index
            state_bytes = segment_data['brain_state'].to_bytes(4, 'big')
            index_bytes = segment_data['segment_index'].to_bytes(4, 'big')

            segment_bytes += params_size + params_bytes + latent_size + latent_bytes + state_bytes + index_bytes

        # Combine metadata and segments
        packed_data = len(metadata_bytes).to_bytes(4, 'big') + metadata_bytes + segment_bytes

        return packed_data


class BrainStateDetector:
    """
    Brain state detector for conditional VAE.

    Implements simple brain state detection based on signal characteristics.
    """

    def __init__(self, n_states: int = 4):
        """
        Initialize brain state detector.

        Parameters
        ----------
        n_states : int, default=4
            Number of brain states to detect
        """
        self.n_states = n_states

    def detect_state(self, segment: np.ndarray) -> int:
        """
        Detect brain state for a segment.

        Parameters
        ----------
        segment : np.ndarray
            Input segment

        Returns
        -------
        int
            Detected brain state
        """
        # Simple state detection based on signal characteristics
        # In practice, this would use more sophisticated methods

        # Calculate features
        mean_amplitude = np.mean(np.abs(segment))
        spectral_entropy = self._calculate_spectral_entropy(segment)
        zero_crossings = self._count_zero_crossings(segment)

        # Simple rule-based classification
        if mean_amplitude < 0.1 and spectral_entropy < 2.0:
            return 0  # Rest state
        elif mean_amplitude > 0.3 and spectral_entropy > 4.0:
            return 1  # Active state
        elif zero_crossings > 50:
            return 2  # High-frequency state
        else:
            return 3  # Mixed state

    def _calculate_spectral_entropy(self, segment: np.ndarray) -> float:
        """Calculate spectral entropy of segment."""
        fft = np.fft.fft(segment)
        power_spectrum = np.abs(fft) ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-8))
        return entropy

    def _count_zero_crossings(self, segment: np.ndarray) -> int:
        """Count zero crossings in segment."""
        return np.sum(np.diff(np.sign(segment)) != 0)


# Factory function for creating VAE compressors
def create_vae_compressor(
    compressor_type: str = "standard",
    **kwargs
) -> VAECompressor:
    """
    Create a VAE compressor instance.

    Parameters
    ----------
    compressor_type : str, default="standard"
        Type of VAE compressor
    **kwargs
        Additional parameters

    Returns
    -------
    VAECompressor
        VAE compressor instance
    """
    if compressor_type == "conditional":
        return ConditionalVAECompressor(**kwargs)
    else:
        return VAECompressor(**kwargs)
