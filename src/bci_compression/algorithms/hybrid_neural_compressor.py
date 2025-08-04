"""
Hybrid Neural Compression Architecture for Brain-Computer Interfaces (Phase 8b+)

Combines multiple advanced architectures for optimal neural signal compression:
- Multi-scale temporal transformers with learnable frequency-aware positional encodings
- Hierarchical VAE with β-scheduling for quality control
- Cross-attention between spatial and temporal dimensions
- Neural ODE layers for continuous-time dynamics
- Quantization-aware training for mobile deployment
- Neural Architecture Search (NAS) optimization

Performance Targets:
- Compression Ratio: 5-15x depending on quality settings
- Signal Quality: 25-40 dB SNR with β-scheduling
- Latency: <2ms for real-time processing
- Memory: <100MB model size with quantization
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchdiffeq import odeint_adjoint as odeint


class FrequencyAwarePositionalEncoding(nn.Module):
    """
    Learnable positional encodings that adapt to different neural frequency bands.
    Handles frequencies from 1-300Hz with specialized embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 1000, n_freqs: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs  # Number of frequency bands

        # Learnable frequency band embeddings
        self.freq_embeddings = nn.Parameter(
            torch.randn(n_freqs, d_model) / math.sqrt(d_model)
        )

        # Learnable temporal positional embeddings
        self.pe = nn.Parameter(
            torch.randn(max_len, d_model) / math.sqrt(d_model)
        )

        # Frequency band centers (Hz): delta, theta, alpha, beta, gamma, high-gamma
        self.freq_bands = torch.tensor([2.0, 6.0, 10.0, 20.0, 40.0, 80.0])

    def forward(self, x: torch.Tensor, sampling_rate: float = 1000.0) -> torch.Tensor:
        """
        Apply frequency-aware positional encoding.

        Args:
            x: Input tensor (batch_size, seq_len, channels, d_model)
            sampling_rate: Sampling rate in Hz

        Returns:
            Encoded tensor with frequency information
        """
        B, T, C, D = x.shape

        # Get frequency content using FFT
        xf = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(T) * sampling_rate

        # Calculate frequency band weights
        weights = []
        for band_center in self.freq_bands:
            # Gaussian weighting around each frequency band
            w = torch.exp(-(freqs - band_center)**2 / (2 * (band_center/4)**2))
            weights.append(w)
        weights = torch.stack(weights, dim=0)  # (n_freqs, n_freq_bins)

        # Apply frequency-aware encoding
        freq_encoding = torch.einsum('nf,bfc->bnc', weights, torch.abs(xf))
        freq_encoding = torch.einsum('bnc,nd->bcd', freq_encoding, self.freq_embeddings)

        # Combine with temporal encoding
        temp_encoding = self.pe[:T].unsqueeze(0).unsqueeze(2)

        return x + freq_encoding.unsqueeze(1) + temp_encoding

class NeuralODEBlock(nn.Module):
    """
    Neural ODE block for modeling continuous-time neural dynamics.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE function for neural dynamics."""
        return self.net(x)

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between spatial (electrode) and temporal dimensions.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between spatial and temporal features.

        Args:
            spatial: Spatial features (batch, n_channels, d_model)
            temporal: Temporal features (batch, seq_len, d_model)

        Returns:
            Attended features combining spatial and temporal information
        """
        attended = self.mha(
            query=self.norm1(spatial),
            key=self.norm2(temporal),
            value=self.norm2(temporal)
        )[0]
        return spatial + attended

class HierarchicalVAE(nn.Module):
    """
    Hierarchical VAE with β-scheduling for quality-controlled compression.
    """

    def __init__(
        self,
        d_model: int,
        latent_dims: List[int] = [256, 128, 64],
        beta_range: Tuple[float, float] = (0.1, 2.0)
    ):
        super().__init__()

        self.latent_dims = latent_dims
        self.beta_range = beta_range
        self.current_beta = beta_range[0]

        # Hierarchical encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_out * 2),
                nn.SiLU(),
                nn.Linear(d_out * 2, d_out * 2)  # μ, log σ
            )
            for d_in, d_out in zip([d_model] + latent_dims[:-1], latent_dims)
        ])

        # Hierarchical decoder
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_out * 2),
                nn.SiLU(),
                nn.Linear(d_out * 2, d_out)
            )
            for d_in, d_out in zip(latent_dims, [d_model] + latent_dims[:-1][::-1])
        ])

    def encode(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Hierarchical encoding with multiple latent layers."""
        stats = []
        current = x

        for encoder in self.encoders:
            params = encoder(current)
            mu, log_var = params.chunk(2, dim=-1)
            stats.append((mu, log_var))

            # Sample latent and pass to next level
            z = self.reparameterize(mu, log_var)
            current = z

        return stats

    def decode(self, zs: List[torch.Tensor]) -> torch.Tensor:
        """Hierarchical decoding from multiple latent layers."""
        current = zs[-1]

        for decoder, z in zip(self.decoders, zs[-2::-1]):
            current = decoder(current)
            # Skip connection from encoder
            current = current + z

        return current

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with diagonal Gaussian."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def update_beta(self, target_quality: float):
        """Update β value based on target quality level."""
        # Scale β between min and max based on quality target
        quality_scale = (target_quality - 15) / (40 - 15)  # 15-40 dB range
        self.current_beta = (
            self.beta_range[0] +
            (self.beta_range[1] - self.beta_range[0]) * quality_scale
        )

class QuantizationLayer(nn.Module):
    """
    Quantization-aware training layer with straight-through estimator.
    """

    def __init__(self, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2**n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        if self.training:
            # Quantization-aware training
            x_scaled = (x * (self.n_levels - 1)).round()
            return x_scaled / (self.n_levels - 1)
        else:
            # Actual integer quantization for inference
            return torch.round(x * (self.n_levels - 1)) / (self.n_levels - 1)

class HybridNeuralCompressor(nn.Module):
    """
    Complete hybrid neural compression architecture.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 1000,
        n_freqs: int = 6,
        latent_dims: List[int] = [256, 128, 64],
        quantization_bits: int = 8
    ):
        super().__init__()

        # Components
        self.pos_encoding = FrequencyAwarePositionalEncoding(
            d_model, max_len, n_freqs
        )

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.neural_ode = NeuralODEBlock(d_model)

        self.cross_attention = CrossAttentionBlock(d_model, n_heads)

        self.vae = HierarchicalVAE(d_model, latent_dims)

        self.quantization = QuantizationLayer(quantization_bits)

        # Architecture search space
        self.arch_params = nn.Parameter(torch.randn(4))  # Weights for different components

    def forward(
        self,
        x: torch.Tensor,
        sampling_rate: float = 1000.0,
        target_quality: float = 30.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hybrid compression network.

        Args:
            x: Input neural signals (batch, seq_len, channels)
            sampling_rate: Sampling rate in Hz
            target_quality: Target SNR in dB

        Returns:
            Tuple of (compressed_representation, metadata)
        """
        B, T, C = x.shape

        # Update VAE β-scheduling based on target quality
        self.vae.update_beta(target_quality)

        # 1. Apply frequency-aware positional encoding
        h = self.pos_encoding(x.unsqueeze(-1) * math.sqrt(self.pos_encoding.d_model),
                            sampling_rate)

        # 2. Neural ODE for continuous-time modeling
        ode_h = odeint(self.neural_ode, h.reshape(B, -1, self.pos_encoding.d_model),
                      torch.linspace(0, 1, 10))[-1]
        ode_h = ode_h.reshape(B, T, C, -1)

        # 3. Cross-attention between spatial and temporal features
        spatial_h = ode_h.mean(dim=1)  # (B, C, D)
        temporal_h = ode_h.mean(dim=2)  # (B, T, D)
        attended_h = self.cross_attention(spatial_h, temporal_h)

        # 4. Transformer layers
        transformer_h = ode_h
        for layer in self.transformer_layers:
            transformer_h = layer(transformer_h.reshape(B, -1, self.pos_encoding.d_model))
        transformer_h = transformer_h.reshape(B, T, C, -1)

        # 5. Hierarchical VAE compression
        vae_stats = self.vae.encode(transformer_h)
        latents = [self.reparameterize(mu, log_var) for mu, log_var in vae_stats]

        # 6. Quantization-aware compression
        quantized = self.quantization(latents[-1])

        # Combine features using learned architecture weights
        weights = F.softmax(self.arch_params, dim=0)
        compressed = (
            weights[0] * quantized +
            weights[1] * attended_h.unsqueeze(1) +
            weights[2] * ode_h.mean(dim=1).unsqueeze(1) +
            weights[3] * transformer_h.mean(dim=2)
        )

        # Store metadata for monitoring
        metadata = {
            'vae_stats': vae_stats,
            'arch_weights': weights,
            'beta': self.vae.current_beta,
            'quantization_bits': self.quantization.n_bits
        }

        return compressed, metadata

    def configure_architecture(self, task_requirements: Dict[str, float]):
        """
        Configure architecture based on task requirements using NAS.

        Args:
            task_requirements: Dictionary with requirements like:
                - target_latency_ms
                - min_quality_db
                - max_model_size_mb
        """
        # Implement neural architecture search optimization
        pass  # TODO: Implement NAS

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
