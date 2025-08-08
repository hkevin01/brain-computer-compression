"""
Deep learning-based compression algorithms for neural data.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # Fallback dummies
    torch = None  # type: ignore
    nn = object  # type: ignore

from ..core import BaseCompressor, Config


class AutoencoderCompressor(BaseCompressor):
    """
    Neural network-based autoencoder compressor for neural data.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        latent_dim: Optional[int] = None,
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.encoding_dim = latent_dim if latent_dim is not None else encoding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Placeholder for encoder/decoder weights
        self.encoder = None
        self.decoder = None
        self._is_trained = False

    def fit(self, data: np.ndarray) -> None:
        """Fit autoencoder model to data (dummy implementation)."""
        # In a real implementation, train encoder/decoder here
        self.encoder = lambda x: x[:, :self.encoding_dim]
        self.decoder = lambda x: np.pad(x, ((0, 0), (0, data.shape[1] - self.encoding_dim)))
        self._is_trained = True
        super().fit(data)

    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress data using encoder."""
        if self.encoder is None:
            self.fit(data)
        compressed = self.encoder(data)
        original_size = data.nbytes
        compressed_size = compressed.nbytes
        self.compression_ratio = original_size / compressed_size
        return compressed

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress data using decoder."""
        if self.decoder is None:
            raise RuntimeError("Model not fitted.")
        return self.decoder(compressed)

    def save_model(self, filepath: str) -> None:
        """Save trained model weights."""
        if not self._is_trained:
            raise ValueError("No trained model to save")

        np.savez(
            filepath,
            encoder_weights=self.encoder_weights,
            decoder_weights=self.decoder_weights,
            latent_dim=self.latent_dim
        )

    def load_model(self, filepath: str) -> None:
        """Load trained model weights."""
        data = np.load(filepath)
        self.encoder_weights = data['encoder_weights']
        self.decoder_weights = data['decoder_weights']
        self.latent_dim = int(data['latent_dim'])
        self._is_trained = True


# ----------------------------- Utility Modules -----------------------------

class _CausalConv1d(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):  # pragma: no cover - simple wrapper
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(in_ch, out_ch, k)

    def forward(self, x):  # x: (b, c, t)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _LocalCausalSelfAttention(nn.Module):  # type: ignore[misc]
    def __init__(self, dim: int, n_heads: int = 4, window: int = 128):  # pragma: no cover
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.window = window

    def forward(self, x):  # x: (b, t, d)
        if x.size(1) > self.window:
            x_win = x[:, -self.window:, :]
        else:
            x_win = x
        attn_out, _ = self.attn(x_win, x_win, x_win, need_weights=False)
        if x_win.size(1) != x.size(1):  # pad to original length
            pad_len = x.size(1) - x_win.size(1)
            pad = x[:, :pad_len, :]
            return torch.cat([pad, attn_out], dim=1)
        return attn_out


class _Quantizer(nn.Module):  # type: ignore[misc]
    def __init__(self, bits: int = 8):  # pragma: no cover
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bits = bits

    def forward(self, x):
        q_levels = 2 ** self.bits - 1
        x_scaled = x / (self.scale + 1e-8)
        x_clamped = torch.tanh(x_scaled)
        q = torch.round((x_clamped + 1) / 2 * q_levels)
        return q, {'scale': float(self.scale.detach().cpu()), 'bits': self.bits}

    def dequantize(self, q, meta):
        q_levels = 2 ** meta['bits'] - 1
        x = (q / q_levels) * 2 - 1
        return torch.atanh(torch.clamp(x, -0.999, 0.999)) * meta['scale']


# ----------------------------- Transformer Compressor -----------------------------

class TransformerCompressor(BaseCompressor):
    """Causal transformer-based neural data compressor.

    Simplified architecture:
    Conv1d frontend -> positional add -> local causal self-attn blocks -> quantize -> entropy code (placeholder)
    """

    def __init__(
        self,
        model_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        window: int = 128,
        quant_bits: int = 8,
        config: Config | None = None,
    ):
        super().__init__(name="transformer", config=config)
        self.available = torch is not None
        self.model_dim = model_dim
        self.depth = depth
        self.window = window
        self.quant_bits = quant_bits
        if self.available:
            self.frontend = _CausalConv1d(1, model_dim, 7)
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    _LocalCausalSelfAttention(model_dim, heads, window),
                    nn.LayerNorm(model_dim),
                    nn.Sequential(nn.Linear(model_dim, model_dim * 4), nn.GELU(), nn.Linear(model_dim * 4, model_dim)),
                    nn.LayerNorm(model_dim),
                )
                for _ in range(depth)
            ])
            self.quantizer = _Quantizer(bits=quant_bits)
        else:  # Fallback
            self.frontend = None
        self.latent_meta: Dict[str, Any] = {}

    def _fit_impl(self, data: np.ndarray) -> None:
        # Placeholder: no training (would implement streaming fine-tune or adaptation)
        return None

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        if not self.available:
            # Fallback: raw bytes passthrough
            return data.tobytes(), {'mode': 'fallback'}
        arr = data.astype(np.float32)
        x = torch.from_numpy(arr)[None, None, :]  # (1,1,T)
        x = self.frontend(x)  # (1,d,T')
        x = x.transpose(1, 2)  # (1,T',d)
        for blk in self.blocks:
            x = x + blk(x)
        q, q_meta = self.quantizer(x)
        # Entropy coding placeholder: store as bytes
        q_bytes = q.detach().cpu().numpy().astype(np.uint16).tobytes()
        q_meta['latent_shape'] = list(q.shape)
        return q_bytes, q_meta

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        if not self.available or metadata.get('mode') == 'fallback':
            # Fallback: reinterpret original shape unknown -> return empty
            return np.frombuffer(compressed, dtype=np.float32)
        shape = metadata['latent_shape']
        q = np.frombuffer(compressed, dtype=np.uint16).reshape(shape)
        q_t = torch.from_numpy(q)
        x = self.quantizer.dequantize(q_t, metadata)
        x = x.transpose(1, 2)  # (1,d,T)
        # Simple inverse of conv not implemented; return projected signal
        rec = x.mean(dim=1)  # crude placeholder
        return rec.detach().cpu().numpy()[0]

    def stream_chunk(self, chunk: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        return self.compress(chunk)


def create_transformer_compressor(mode: str = "balanced", **kwargs) -> TransformerCompressor:
    presets = {
        'fast': dict(model_dim=48, depth=1, heads=2, window=96, quant_bits=8),
        'balanced': dict(model_dim=64, depth=2, heads=4, window=128, quant_bits=8),
        'quality': dict(model_dim=96, depth=3, heads=6, window=192, quant_bits=10),
    }
    cfg = presets.get(mode, presets['balanced'])
    cfg.update(kwargs)
    return TransformerCompressor(**cfg)


# ----------------------------- VAE Compressor -----------------------------

class _VAE(nn.Module):  # type: ignore[misc]
    def __init__(self, latent_dim: int = 32):  # pragma: no cover
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2 * latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.dec(z)
        return recon, mu, logvar, z


class VAECompressor(BaseCompressor):
    """Beta-VAE compressor with uncertainty-driven bitrate (simplified)."""

    def __init__(
        self,
        window: int = 512,
        latent_dim: int = 32,
        beta: float = 1.0,
        train_steps: int = 50,
        lr: float = 1e-3,
        config: Config | None = None,
    ):
        super().__init__(name="vae", config=config)
        self.available = torch is not None
        self.window = window
        self.latent_dim = latent_dim
        self.beta = beta
        self.train_steps = train_steps
        self.lr = lr
        if self.available:
            self.model = _VAE(latent_dim)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = None
        self.trained = False

    def _fit_impl(self, data: np.ndarray) -> None:
        if not self.available:
            return
        # Prepare windows
        x = data.flatten()
        n = (len(x) // self.window) * self.window
        if n == 0:
            return
        x = x[:n].reshape(-1, self.window)
        t = torch.from_numpy(x.astype(np.float32))
        self.model.train()
        for step in range(self.train_steps):
            idx = np.random.randint(0, t.size(0), size=min(8, t.size(0)))
            batch = t[idx]
            recon, mu, logvar, _ = self.model(batch)
            recon_loss = ((recon - batch) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.trained = True

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        if not self.available:
            return data.tobytes(), {'mode': 'fallback'}
        if not self.trained:
            self._fit_impl(data)
        x = data.flatten().astype(np.float32)
        n = (len(x) // self.window) * self.window
        if n == 0:
            return b"", {'empty': True}
        xw = x[:n].reshape(-1, self.window)
        t = torch.from_numpy(xw)
        self.model.eval()
        with torch.no_grad():
            recon, mu, logvar, z = self.model(t)
        # Quantize latent using uncertainty (larger var -> fewer bits)
        var = torch.exp(logvar)
        inv_uncertainty = 1 / (var + 1e-6)
        normalized = (z * inv_uncertainty.sqrt()).float()
        latent = normalized.numpy().astype(np.float16)
        meta = {
            'latent_shape': list(latent.shape),
            'window': self.window,
            'latent_dim': self.latent_dim,
            'beta': self.beta,
        }
        return latent.tobytes(), meta

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        if not self.available or metadata.get('mode') == 'fallback':
            return np.frombuffer(compressed, dtype=np.float32)
        if not compressed:
            return np.array([])
        shape = metadata['latent_shape']
        latent = np.frombuffer(compressed, dtype=np.float16).reshape(shape)
        t = torch.from_numpy(latent.astype(np.float32))
        # Decode with decoder only path (approximate)
        with torch.no_grad():
            recon = self.model.dec(t)
        rec = recon.numpy().astype(np.float32).reshape(-1)
        return rec

    def stream_chunk(self, chunk: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        return self.compress(chunk)


def create_vae_compressor(**kwargs) -> VAECompressor:
    return VAECompressor(**kwargs)
