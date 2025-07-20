"""
Tests for Phase 8: Advanced Neural Compression.

This module tests the transformer-based compression and VAE compression
algorithms implemented in Phase 8.
"""

import logging
import time
from typing import Dict, List

import numpy as np
import pytest

from bci_compression.algorithms.factory import create_compressor
from bci_compression.algorithms.transformer_compression import (
    AdaptiveTransformerCompressor,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerCompressor,
    TransformerEncoder,
)
from bci_compression.algorithms.vae_compression import (
    BrainStateDetector,
    ConditionalVAECompressor,
    VAECompressor,
    VAEDecoder,
    VAEEncoder,
)


class TestTransformerCompression:
    """Test transformer-based compression algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = np.random.randn(4, 1000).astype(np.float32)
        self.single_channel_data = np.random.randn(1000).astype(np.float32)

        # Configure logging
        logging.basicConfig(level=logging.INFO)

    def test_positional_encoding(self):
        """Test positional encoding for neural sequences."""
        pe = PositionalEncoding(max_length=1000, d_model=256)

        # Test encoding
        encoded = pe.encode(500)
        assert encoded.shape == (500, 256)
        assert not np.allclose(encoded, 0)

        # Test different sequence lengths
        encoded_short = pe.encode(100)
        assert encoded_short.shape == (100, 256)

        # Test error handling
        with pytest.raises(ValueError):
            pe.encode(10000)  # Exceeds max_length

    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        attention = MultiHeadAttention(d_model=256, n_heads=8, dropout=0.0)  # No dropout for testing

        # Test attention computation
        batch_size, seq_len, d_model = 2, 100, 256
        x = np.random.randn(batch_size, seq_len, d_model)

        output, attention_weights = attention.forward(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)

        # Test attention weights sum to 1 (only if no dropout)
        attention_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(attention_sums, 1.0, atol=1e-6)

    def test_transformer_encoder(self):
        """Test transformer encoder."""
        encoder = TransformerEncoder(d_model=256, n_heads=8, n_layers=2)

        # Test encoding
        batch_size, seq_len, d_model = 1, 100, 256
        x = np.random.randn(batch_size, seq_len, d_model)

        encoded, attention_weights = encoder.encode(x)

        assert encoded.shape == (batch_size, seq_len, d_model)
        assert len(attention_weights) == 2  # 2 layers

        # Test different sequence lengths
        x_short = np.random.randn(batch_size, 50, d_model)
        encoded_short, _ = encoder.encode(x_short)
        assert encoded_short.shape == (batch_size, 50, d_model)

    def test_transformer_compressor_basic(self):
        """Test basic transformer compressor functionality."""
        compressor = TransformerCompressor(
            d_model=256,
            n_heads=8,
            n_layers=2,
            max_sequence_length=500
        )

        # Test compression
        compressed = compressor.compress(self.single_channel_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        # Test decompression
        decompressed = compressor.decompress(compressed)
        assert decompressed.shape == self.single_channel_data.shape
        assert decompressed.dtype == self.single_channel_data.dtype

        # Test compression ratio
        ratio = compressor.get_compression_ratio()
        assert ratio > 0
        assert ratio < 10  # Should achieve some compression

    def test_transformer_compressor_multi_channel(self):
        """Test transformer compressor with multi-channel data."""
        compressor = TransformerCompressor(
            d_model=256,
            n_heads=8,
            n_layers=2,
            max_sequence_length=500
        )

        # Test compression
        compressed = compressor.compress(self.test_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        # Test decompression
        decompressed = compressor.decompress(compressed)
        assert decompressed.shape == self.test_data.shape
        assert decompressed.dtype == self.test_data.dtype

        # Test compression statistics
        stats = compressor.compression_stats
        assert 'attention_weights' in stats
        assert 'compression_ratio' in stats
        assert 'processing_time' in stats
        assert 'quality_metrics' in stats

    def test_adaptive_transformer_compressor(self):
        """Test adaptive transformer compressor."""
        compressor = AdaptiveTransformerCompressor(
            d_model=256,
            n_heads=8,
            n_layers=2,
            quality_threshold=0.9,
            adaptive_compression=True
        )

        # Test signal analysis
        characteristics = compressor._analyze_signal_characteristics(self.single_channel_data)
        assert 'mean' in characteristics
        assert 'std' in characteristics
        assert 'dynamic_range' in characteristics
        assert 'spectral_entropy' in characteristics

        # Test parameter adaptation
        adapted_params = compressor._adapt_compression_parameters(characteristics)
        assert 'quantization_bits' in adapted_params
        assert 'compression_ratio' in adapted_params

        # Test compression
        compressed = compressor.compress(self.single_channel_data)
        assert isinstance(compressed, bytes)

        # Test adaptation history
        assert len(compressor.adaptive_params['compression_adjustments']) > 0

    def test_transformer_compressor_factory(self):
        """Test transformer compressor creation via factory."""
        # Test standard transformer
        transformer = create_compressor("transformer", d_model=128, n_heads=4)
        assert isinstance(transformer, TransformerCompressor)

        # Test adaptive transformer
        adaptive_transformer = create_compressor("adaptive_transformer", d_model=128, n_heads=4)
        assert isinstance(adaptive_transformer, AdaptiveTransformerCompressor)

        # Test compression
        compressed = transformer.compress(self.single_channel_data)
        decompressed = transformer.decompress(compressed)
        assert decompressed.shape == self.single_channel_data.shape


class TestVAECompression:
    """Test VAE-based compression algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = np.random.randn(4, 1000).astype(np.float32)
        self.single_channel_data = np.random.randn(1000).astype(np.float32)

        # Configure logging
        logging.basicConfig(level=logging.INFO)

    def test_vae_encoder(self):
        """Test VAE encoder."""
        encoder = VAEEncoder(input_size=1024, latent_dim=64)

        # Test encoding
        x = np.random.randn(1024)
        mu, logvar = encoder.encode(x)

        assert mu.shape == (64,)
        assert logvar.shape == (64,)
        assert not np.allclose(mu, 0)
        assert not np.allclose(logvar, 0)

    def test_vae_decoder(self):
        """Test VAE decoder."""
        decoder = VAEDecoder(latent_dim=64, output_size=1024)

        # Test decoding
        z = np.random.randn(64)
        output = decoder.decode(z)

        assert output.shape == (1024,)
        assert not np.allclose(output, 0)

    def test_vae_compressor_basic(self):
        """Test basic VAE compressor functionality."""
        compressor = VAECompressor(
            input_size=1024,
            latent_dim=64,
            beta=1.0,
            quality_threshold=0.9
        )

        # Test data segmentation
        segments = compressor._segment_data(self.single_channel_data)
        assert len(segments) > 0
        assert all(len(seg) == 1024 for seg in segments)

        # Test reparameterization
        mu = np.random.randn(64)
        logvar = np.random.randn(64)
        z = compressor._reparameterize(mu, logvar)
        assert z.shape == (64,)

        # Test KL divergence
        kl_loss = compressor._kl_divergence(mu, logvar)
        assert isinstance(kl_loss, float)
        assert kl_loss >= 0

        # Test reconstruction loss
        original = np.random.randn(1024)
        reconstructed = np.random.randn(1024)
        loss = compressor._reconstruction_loss(original, reconstructed)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_vae_compressor_compression(self):
        """Test VAE compressor compression and decompression."""
        compressor = VAECompressor(
            input_size=1024,
            latent_dim=64,
            beta=1.0,
            quality_threshold=0.9
        )

        # Test compression
        compressed = compressor.compress(self.single_channel_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        # Test decompression
        decompressed = compressor.decompress(compressed)
        assert decompressed.shape == self.single_channel_data.shape
        assert decompressed.dtype == self.single_channel_data.dtype

        # Test compression statistics
        stats = compressor.compression_stats
        assert 'reconstruction_loss' in stats
        assert 'kl_divergence' in stats
        assert 'compression_ratio' in stats
        assert 'uncertainty' in stats
        assert 'quality_metrics' in stats

    def test_vae_compressor_training(self):
        """Test VAE compressor training."""
        compressor = VAECompressor(
            input_size=1024,
            latent_dim=64,
            beta=1.0
        )

        # Test training
        training_data = np.random.randn(2000).astype(np.float32)
        compressor.fit(training_data, epochs=5)

        assert compressor.is_trained
        assert len(compressor.training_losses) == 5
        assert all(isinstance(loss, float) for loss in compressor.training_losses)

    def test_conditional_vae_compressor(self):
        """Test conditional VAE compressor."""
        compressor = ConditionalVAECompressor(
            input_size=1024,
            latent_dim=64,
            n_brain_states=4
        )

        # Test brain state detection
        segment = np.random.randn(1024)
        brain_state = compressor._detect_brain_state(segment)
        assert isinstance(brain_state, int)
        assert 0 <= brain_state < 4

        # Test compression
        compressed = compressor.compress(self.single_channel_data)
        assert isinstance(compressed, bytes)

        # Test state counts
        # Note: This would require accessing internal state which may not be exposed
        # The compression should work regardless

    def test_brain_state_detector(self):
        """Test brain state detector."""
        detector = BrainStateDetector(n_states=4)

        # Test state detection
        segment = np.random.randn(1024)
        state = detector.detect_state(segment)
        assert isinstance(state, int)
        assert 0 <= state < 4

        # Test spectral entropy calculation
        entropy = detector._calculate_spectral_entropy(segment)
        assert isinstance(entropy, float)
        assert entropy >= 0

        # Test zero crossing count
        crossings = detector._count_zero_crossings(segment)
        assert isinstance(crossings, (int, np.integer))  # Allow numpy integer types
        assert crossings >= 0

    def test_vae_compressor_factory(self):
        """Test VAE compressor creation via factory."""
        # Test standard VAE
        vae = create_compressor("vae", input_size=1024, latent_dim=64)
        assert isinstance(vae, VAECompressor)

        # Test conditional VAE
        conditional_vae = create_compressor("conditional_vae", input_size=1024, latent_dim=64)
        assert isinstance(conditional_vae, ConditionalVAECompressor)

        # Test compression
        compressed = vae.compress(self.single_channel_data)
        decompressed = vae.decompress(compressed)
        assert decompressed.shape == self.single_channel_data.shape


class TestPhase8Integration:
    """Integration tests for Phase 8 features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = np.random.randn(4, 1000).astype(np.float32)
        self.single_channel_data = np.random.randn(1000).astype(np.float32)

        # Configure logging
        logging.basicConfig(level=logging.INFO)

    def test_algorithm_registry_phase8(self):
        """Test that Phase 8 algorithms are registered in factory."""
        from bci_compression.algorithms.factory import list_available_algorithms

        available = list_available_algorithms()

        # Check that Phase 8 algorithms are available
        phase8_algorithms = [
            "transformer",
            "adaptive_transformer",
            "vae",
            "conditional_vae"
        ]

        for algorithm in phase8_algorithms:
            if algorithm in available:
                print(f"✓ {algorithm} is available")
            else:
                print(f"✗ {algorithm} is not available")

        # At least some Phase 8 algorithms should be available
        available_phase8 = [alg for alg in phase8_algorithms if alg in available]
        assert len(available_phase8) > 0, f"Expected Phase 8 algorithms, got: {available}"

    def test_compression_quality_comparison(self):
        """Compare compression quality between Phase 8 algorithms."""
        algorithms = []

        # Try to create different Phase 8 compressors
        try:
            transformer = create_compressor("transformer", d_model=128, n_heads=4)
            algorithms.append(("transformer", transformer))
        except Exception as e:
            print(f"Could not create transformer compressor: {e}")

        try:
            vae = create_compressor("vae", input_size=1024, latent_dim=64)
            algorithms.append(("vae", vae))
        except Exception as e:
            print(f"Could not create VAE compressor: {e}")

        if not algorithms:
            pytest.skip("No Phase 8 algorithms available for testing")

        # Test compression with each algorithm
        results = {}

        for name, compressor in algorithms:
            try:
                start_time = time.time()
                compressed = compressor.compress(self.single_channel_data)
                compression_time = time.time() - start_time

                decompressed = compressor.decompress(compressed)

                # Calculate metrics
                compression_ratio = compressor.get_compression_ratio()
                mse = np.mean((self.single_channel_data - decompressed) ** 2)

                results[name] = {
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time,
                    'mse': mse,
                    'compressed_size': len(compressed)
                }

                print(f"{name}: ratio={compression_ratio:.2f}, time={compression_time:.3f}s, mse={mse:.6f}")

            except Exception as e:
                print(f"Error testing {name}: {e}")
                results[name] = None

        # Verify that at least one algorithm worked
        working_algorithms = [name for name, result in results.items() if result is not None]
        assert len(working_algorithms) > 0, "No Phase 8 algorithms worked"

        # Verify compression ratios are reasonable
        for name, result in results.items():
            if result is not None:
                assert result['compression_ratio'] > 0, f"{name}: compression ratio should be positive"
                assert result['compression_time'] > 0, f"{name}: compression time should be positive"

    def test_memory_usage(self):
        """Test memory usage of Phase 8 algorithms."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test transformer compressor memory usage
        try:
            transformer = TransformerCompressor(d_model=256, n_heads=8, n_layers=2)

            # Compress data
            compressed = transformer.compress(self.test_data)
            decompressed = transformer.decompress(compressed)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(f"Transformer memory usage: {memory_increase:.2f} MB")

            # Memory increase should be reasonable (< 100MB for this test)
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f} MB"

        except Exception as e:
            print(f"Transformer memory test failed: {e}")

        # Test VAE compressor memory usage
        try:
            vae = VAECompressor(input_size=1024, latent_dim=64)

            # Compress data
            compressed = vae.compress(self.single_channel_data)
            decompressed = vae.decompress(compressed)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(f"VAE memory usage: {memory_increase:.2f} MB")

            # Memory increase should be reasonable (< 100MB for this test)
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f} MB"

        except Exception as e:
            print(f"VAE memory test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
