"""
Predictive compression algorithms for neural signals.

This module implements temporal prediction models that exploit the predictable
components of neural signals to achieve superior compression performance.
"""

import logging
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PredictionMetadata:
    """Metadata for predictive compression."""
    predictor_type: str
    prediction_order: int
    channels: int
    samples: int
    coefficients: Dict
    prediction_accuracy: float
    compression_time: float
    original_bits: int
    compressed_bits: int


class NeuralLinearPredictor:
    """
    Linear predictive coding optimized for neural data.

    Uses the Levinson-Durbin algorithm with modifications for neural signal
    characteristics including spike handling and frequency-dependent modeling.
    """

    def __init__(self, order: int = 10, channels: Optional[List[int]] = None):
        """
        Initialize neural linear predictor.

        Args:
            order: Prediction filter order (default: 10)
            channels: Channel indices to process (None for all)
        """
        self.order = order
        self.channels = channels
        self.coefficients = {}
        self.prediction_errors = {}
        self.adaptation_rate = 0.01

    def _compute_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """Compute autocorrelation function for neural signal."""
        n = len(signal)
        autocorr = np.zeros(self.order + 1)

        # Handle neural signal characteristics
        signal_normalized = signal - np.mean(signal)

        for lag in range(self.order + 1):
            if lag < n:
                autocorr[lag] = np.sum(signal_normalized[:-lag or None] *
                                       signal_normalized[lag:]) / (n - lag)

        return autocorr

    def _levinson_durbin(self, autocorr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Levinson-Durbin algorithm for optimal prediction coefficients.

        Modified for neural signal stability and numerical precision.
        """
        order = len(autocorr) - 1
        coeffs = np.zeros(order)

        if autocorr[0] == 0:
            return coeffs, 0.0

        # Initialize
        error = autocorr[0]

        for i in range(order):
            # Reflection coefficient
            reflection = autocorr[i + 1]
            for j in range(i):
                reflection -= coeffs[j] * autocorr[i - j]
            reflection /= error

            # Update coefficients
            new_coeffs = np.zeros(i + 1)
            new_coeffs[i] = reflection
            for j in range(i):
                new_coeffs[j] = coeffs[j] - reflection * coeffs[i - 1 - j]

            coeffs[:i + 1] = new_coeffs

            # Update prediction error
            error *= (1 - reflection**2)

            # Stability check for neural signals
            if error <= 0 or abs(reflection) >= 1:
                break

        return coeffs, error

    def fit_predictor(self, data: np.ndarray, channel_id: int) -> Dict:
        """
        Fit LPC coefficients for a specific channel.

        Args:
            data: Neural signal data for the channel
            channel_id: Channel identifier

        Returns:
            Dictionary with prediction coefficients and metrics
        """
        # Preprocess neural signal
        signal = data.astype(np.float64)

        # Remove DC component and normalize
        signal = signal - np.mean(signal)
        signal_std = np.std(signal)
        if signal_std > 0:
            signal = signal / signal_std

        # Compute autocorrelation
        autocorr = self._compute_autocorrelation(signal)

        # Apply Levinson-Durbin algorithm
        coeffs, pred_error = self._levinson_durbin(autocorr)

        # Store coefficients
        self.coefficients[channel_id] = {
            'coefficients': coeffs,
            'prediction_error': pred_error,
            'mean': np.mean(data),
            'std': signal_std,
            'order': self.order
        }

        # Calculate prediction accuracy
        if len(signal) > self.order:
            # Get the correct prediction length
            predictions = self.predict_samples(signal, channel_id)
            actual = signal[self.order:self.order + len(predictions)]

            if len(actual) > 0 and len(predictions) > 0:
                mse = np.mean((actual - predictions)**2)
                accuracy = 1.0 - mse / np.var(actual) if np.var(actual) > 0 else 0.0
            else:
                accuracy = 0.0
        else:
            accuracy = 0.0

        self.coefficients[channel_id]['accuracy'] = max(0.0, accuracy)

        return self.coefficients[channel_id]

    def predict_samples(self, history: np.ndarray, channel_id: int) -> np.ndarray:
        """
        Predict next samples based on history.

        Args:
            history: Historical signal samples
            channel_id: Channel identifier

        Returns:
            Predicted samples
        """
        if channel_id not in self.coefficients:
            return np.zeros(len(history) - self.order)

        coeffs_data = self.coefficients[channel_id]
        coeffs = coeffs_data['coefficients']
        mean_val = coeffs_data['mean']
        std_val = coeffs_data['std']

        # Normalize input
        normalized_history = (history - mean_val)
        if std_val > 0:
            normalized_history = normalized_history / std_val

        # Predict samples
        predictions = []
        padded_history = np.concatenate([np.zeros(self.order), normalized_history])

        # Only predict for samples where we have enough history
        max_predictions = len(normalized_history) - self.order
        if max_predictions <= 0:
            return np.array([])

        for i in range(self.order, self.order + max_predictions):
            prediction = np.dot(coeffs, padded_history[i - self.order:i][::-1])
            predictions.append(prediction)

        # Denormalize predictions
        predictions = np.array(predictions)
        if std_val > 0:
            predictions = predictions * std_val
        predictions = predictions + mean_val

        return predictions

    def encode_residuals(self, signal: np.ndarray, predictions: np.ndarray,
                         quantization_bits: int = 12) -> Tuple[bytes, Dict]:
        """
        Encode prediction residuals efficiently.

        Args:
            signal: Original signal
            predictions: Predicted signal
            quantization_bits: Bits for residual quantization

        Returns:
            Encoded residuals and metadata
        """
        # Calculate residuals
        residuals = signal[self.order:] - predictions

        # Quantize residuals
        residual_max = np.max(np.abs(residuals)) if len(residuals) > 0 else 1.0
        if residual_max == 0:
            residual_max = 1.0

        scale_factor = (2**(quantization_bits - 1) - 1) / residual_max
        quantized_residuals = np.round(residuals * scale_factor).astype(np.int16)

        # Encode residuals using variable length encoding
        encoded_data = bytearray()

        # Header: quantization info
        encoded_data.extend(struct.pack('<f', residual_max))
        encoded_data.extend(struct.pack('<H', quantization_bits))
        encoded_data.extend(struct.pack('<I', len(quantized_residuals)))

        # Encode residuals with simple run-length encoding for zeros
        i = 0
        while i < len(quantized_residuals):
            value = quantized_residuals[i]
            if value == 0:
                # Count consecutive zeros
                zero_count = 1
                while (i + zero_count < len(quantized_residuals) and
                       quantized_residuals[i + zero_count] == 0 and
                       zero_count < 255):
                    zero_count += 1

                # Encode zero run
                encoded_data.extend(struct.pack('<B', 0))  # Zero marker
                encoded_data.extend(struct.pack('<B', zero_count))
                i += zero_count
            else:
                # Encode non-zero value
                encoded_data.extend(struct.pack('<h', value))
                i += 1

        metadata = {
            'residual_max': residual_max,
            'quantization_bits': quantization_bits,
            'original_samples': len(signal),
            'compressed_size': len(encoded_data),
            'residual_variance': np.var(residuals) if len(residuals) > 0 else 0.0
        }

        return bytes(encoded_data), metadata

    def decompress(self, compressed_data, metadata):
        import logging
        logging.warning("[NeuralLinearPredictor] Decompression is not implemented. Returning zeros as placeholder.")
        n_samples = metadata.samples if hasattr(metadata, 'samples') else 0
        dtype = getattr(self, '_last_dtype', np.float32)
        return np.zeros(n_samples, dtype=dtype)


class AdaptiveNeuralPredictor:
    """
    Adaptive predictor that updates coefficients in real-time.

    Uses normalized least mean squares (NLMS) algorithm for online adaptation
    to changing neural signal characteristics.
    """

    def __init__(self, order: int = 8, mu: float = 0.01, channels: int = 1):
        """
        Initialize adaptive predictor.

        Args:
            order: Filter order
            mu: Adaptation step size
            channels: Number of channels
        """
        self.order = order
        self.mu = mu
        self.channels = channels

        # Initialize coefficients for each channel
        self.coefficients = np.zeros((channels, order))
        self.input_buffers = [deque(maxlen=order) for _ in range(channels)]

        # Performance tracking
        self.prediction_errors = np.zeros(channels)
        self.adaptation_history = []

    def update_predictor(self, channel_id: int, sample: float,
                         target: Optional[float] = None) -> float:
        """
        Update predictor coefficients and predict next sample.

        Args:
            channel_id: Channel index
            sample: Current input sample
            target: Target value for adaptation (if available)

        Returns:
            Predicted next sample
        """
        if channel_id >= self.channels:
            return 0.0

        # Add sample to buffer
        self.input_buffers[channel_id].append(sample)

        # Only predict when buffer is full
        if len(self.input_buffers[channel_id]) < self.order:
            return 0.0

        # Get input vector
        x = np.array(list(self.input_buffers[channel_id]))[::-1]  # Reverse for convolution

        # Predict
        prediction = np.dot(self.coefficients[channel_id], x)

        # Adapt coefficients if target is provided
        if target is not None:
            error = target - prediction
            self.prediction_errors[channel_id] = error

            # NLMS update
            x_power = np.dot(x, x)
            if x_power > 1e-10:  # Avoid division by zero
                step_size = self.mu / (x_power + 1e-10)
                self.coefficients[channel_id] += step_size * error * x

            # Track adaptation
            if len(self.adaptation_history) < 1000:  # Limit history size
                self.adaptation_history.append({
                    'channel': channel_id,
                    'error': error,
                    'prediction': prediction,
                    'target': target
                })

        return prediction

    def get_prediction_statistics(self) -> Dict:
        """Get prediction performance statistics."""
        if not self.adaptation_history:
            return {'mean_error': 0.0, 'mse': 0.0, 'adaptation_rate': 0.0}

        errors = [h['error'] for h in self.adaptation_history]
        return {
            'mean_error': np.mean(errors),
            'mse': np.mean(np.square(errors)),
            'std_error': np.std(errors),
            'adaptation_rate': self.mu,
            'coefficient_norm': np.linalg.norm(self.coefficients),
            'samples_processed': len(self.adaptation_history)
        }


class MultiChannelPredictiveCompressor:
    """
    Multi-channel predictive compressor with cross-channel correlation.

    Combines linear prediction with cross-channel prediction to exploit
    spatial correlations in multi-electrode neural recordings.
    """

    def __init__(self, prediction_order: int = 10,
                 cross_channel_order: int = 3,
                 quantization_bits: int = 12):
        """
        Initialize multi-channel predictive compressor.

        Args:
            prediction_order: Temporal prediction order
            cross_channel_order: Cross-channel prediction order
            quantization_bits: Bits for quantization
        """
        self.prediction_order = prediction_order
        self.cross_channel_order = cross_channel_order
        self.quantization_bits = quantization_bits

        self.temporal_predictors = {}
        self.spatial_predictors = {}
        self.channel_relationships = {}

    def _analyze_channel_relationships(self, data: np.ndarray) -> Dict:
        """Analyze spatial relationships between channels."""
        n_channels, n_samples = data.shape
        correlations = np.corrcoef(data)

        # Find strongly correlated channel pairs
        relationships = {}
        for i in range(n_channels):
            # Find channels with high correlation
            corr_indices = np.argsort(np.abs(correlations[i]))[::-1][1:self.cross_channel_order + 1]
            corr_values = correlations[i][corr_indices]

            relationships[i] = {
                'correlated_channels': corr_indices.tolist(),
                'correlation_values': corr_values.tolist(),
                'spatial_weights': corr_values / np.sum(np.abs(corr_values))
            }

        return relationships

    def compress(self, data: np.ndarray) -> Tuple[List[bytes], PredictionMetadata]:
        logging.info(f"Compressing data with shape {data.shape} and dtype {data.dtype}")
        self._last_shape = data.shape
        self._last_dtype = data.dtype
        start_time = time.time()
        n_channels, n_samples = data.shape

        # Analyze channel relationships
        self.channel_relationships = self._analyze_channel_relationships(data)

        compressed_channels = []
        total_original_bits = data.size * 16  # Assuming 16-bit input
        total_compressed_bits = 0
        channel_accuracies = []

        for ch_idx in range(n_channels):
            # Temporal prediction
            temporal_predictor = NeuralLinearPredictor(order=self.prediction_order)
            temporal_coeffs = temporal_predictor.fit_predictor(data[ch_idx], ch_idx)

            # Temporal predictions
            temporal_predictions = temporal_predictor.predict_samples(
                data[ch_idx], ch_idx
            )

            # Spatial prediction using correlated channels
            spatial_predictions = np.zeros(n_samples - self.prediction_order)
            if ch_idx in self.channel_relationships:
                rel_info = self.channel_relationships[ch_idx]
                corr_channels = rel_info['correlated_channels']
                weights = rel_info['spatial_weights']

                for j, (corr_ch, weight) in enumerate(zip(corr_channels, weights)):
                    if corr_ch < n_channels and abs(weight) > 0.1:  # Minimum correlation threshold
                        # Simple spatial prediction based on correlated channel
                        spatial_component = data[corr_ch][self.prediction_order:] * weight
                        spatial_predictions += spatial_component

            # Combine temporal and spatial predictions
            combined_predictions = 0.7 * temporal_predictions + 0.3 * spatial_predictions

            # Encode residuals
            compressed_data, residual_meta = temporal_predictor.encode_residuals(
                data[ch_idx], combined_predictions, self.quantization_bits
            )

            compressed_channels.append(compressed_data)
            total_compressed_bits += residual_meta['compressed_size'] * 8
            channel_accuracies.append(temporal_coeffs['accuracy'])

            # Store predictors
            self.temporal_predictors[ch_idx] = temporal_predictor

        compression_time = time.time() - start_time

        metadata = PredictionMetadata(
            predictor_type="multi_channel_predictive",
            prediction_order=self.prediction_order,
            channels=n_channels,
            samples=n_samples,
            coefficients={ch: self.temporal_predictors[ch].coefficients[ch]
                          for ch in range(n_channels)},
            prediction_accuracy=np.mean(channel_accuracies),
            compression_time=compression_time,
            original_bits=total_original_bits,
            compressed_bits=total_compressed_bits
        )

        return compressed_channels, metadata

    def decompress(self, compressed_channels: List[bytes],
                   metadata: PredictionMetadata) -> np.ndarray:
        logging.info(f"Decompressing with metadata: channels={metadata.channels}, samples={metadata.samples}")
        n_channels = metadata.channels
        n_samples = metadata.samples
        reconstructed_data = np.zeros((n_channels, n_samples))
        for ch_idx in range(n_channels):
            if ch_idx in self.temporal_predictors:
                predictor = self.temporal_predictors[ch_idx]
                reconstructed_data[ch_idx] = predictor.decompress(compressed_channels[ch_idx], metadata)
            else:
                reconstructed_data[ch_idx] = np.zeros(n_samples)
        # Integrity check
        try:
            if hasattr(self, '_last_shape') and hasattr(self, '_last_dtype'):
                if reconstructed_data.shape != self._last_shape:
                    logging.error(f"Shape mismatch: {reconstructed_data.shape} vs {self._last_shape}")
                    raise ValueError(
                        f"Decompressed data shape {
                            reconstructed_data.shape} does not match original {
                            self._last_shape}")
                reconstructed_data = reconstructed_data.astype(self._last_dtype)
                if reconstructed_data.dtype != self._last_dtype:
                    logging.error(f"Dtype mismatch: {reconstructed_data.dtype} vs {self._last_dtype}")
                    raise ValueError(
                        f"Decompressed data dtype {
                            reconstructed_data.dtype} does not match original {
                            self._last_dtype}")
        except Exception:
            logging.exception("Integrity check failed during decompression")
            raise
        return reconstructed_data


def create_predictive_compressor(mode: str = "balanced") -> MultiChannelPredictiveCompressor:
    """
    Factory function to create predictive compressor with predefined settings.

    Args:
        mode: Compression mode ("speed", "balanced", "quality")

    Returns:
        Configured predictive compressor
    """
    if mode == "speed":
        return MultiChannelPredictiveCompressor(
            prediction_order=6,
            cross_channel_order=2,
            quantization_bits=10
        )
    elif mode == "quality":
        return MultiChannelPredictiveCompressor(
            prediction_order=16,
            cross_channel_order=5,
            quantization_bits=14
        )
    else:  # balanced
        return MultiChannelPredictiveCompressor(
            prediction_order=10,
            cross_channel_order=3,
            quantization_bits=12
        )


# Performance testing functions
def benchmark_predictive_compression():
    """Benchmark predictive compression performance."""
    # Generate test data
    n_channels, n_samples = 16, 3000
    sampling_rate = 30000

    # Synthetic neural data with correlations
    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    data = np.zeros((n_channels, n_samples))

    # Add correlated components
    base_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

    for ch in range(n_channels):
        # Add channel-specific variations
        phase_shift = np.random.random() * 2 * np.pi
        amplitude = 0.5 + 0.5 * np.random.random()

        data[ch] = amplitude * np.roll(base_signal, int(phase_shift * 10))
        data[ch] += np.random.normal(0, 0.1, n_samples)  # Noise

    # Test different compressor modes
    modes = ["speed", "balanced", "quality"]
    results = {}

    for mode in modes:
        compressor = create_predictive_compressor(mode)

        start_time = time.time()
        compressed, metadata = compressor.compress(data)
        compression_time = time.time() - start_time

        compression_ratio = metadata.original_bits / metadata.compressed_bits

        results[mode] = {
            'compression_ratio': compression_ratio,
            'compression_time': compression_time,
            'prediction_accuracy': metadata.prediction_accuracy,
            'bits_per_sample': metadata.compressed_bits / (n_channels * n_samples)
        }

    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_predictive_compression()
    print("Predictive Compression Benchmark Results:")
    for mode, metrics in results.items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Compression Time: {metrics['compression_time']:.4f}s")
        print(f"  Prediction Accuracy: {metrics['prediction_accuracy']:.3f}")
        print(f"  Bits per Sample: {metrics['bits_per_sample']:.2f}")
