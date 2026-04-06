"""
Tests for the 4 new BCI compression algorithms added based on 2025-2026 research.
Covers: RVQCompressor, DSCAECompressor, LLCSpikeCompressor, DiffusionCompressor.
"""

import numpy as np
import pytest


# ============================================================
# Helpers
# ============================================================

RNG = np.random.default_rng(2025)


def make_eeg(n_ch: int = 8, n_samp: int = 512) -> np.ndarray:
    """Synthetic EEG with bandlimited oscillations."""
    t = np.linspace(0, 2, n_samp)
    data = np.zeros((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        freq_alpha = 10.0 + RNG.uniform(-1, 1)
        freq_beta = 20.0 + RNG.uniform(-2, 2)
        data[ch] = (
            float(RNG.standard_normal()) * np.sin(2 * np.pi * freq_alpha * t).astype(np.float32)
            + 0.5 * float(RNG.standard_normal()) * np.sin(2 * np.pi * freq_beta * t).astype(np.float32)
            + 0.2 * RNG.standard_normal(n_samp).astype(np.float32)
        )
    return data


def make_lfp(n_ch: int = 16, n_samp: int = 1024) -> np.ndarray:
    """Synthetic LFP: slow oscillations + noise."""
    t = np.linspace(0, 1, n_samp)
    data = np.zeros((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        data[ch] = (
            np.sin(2 * np.pi * 4.0 * t).astype(np.float32)
            + 0.3 * RNG.standard_normal(n_samp).astype(np.float32)
        )
    return data


def make_spike_train(n_ch: int = 4, n_samp: int = 500, rate: float = 0.05) -> np.ndarray:
    """Sparse binary spike train."""
    return (RNG.random((n_ch, n_samp)) < rate).astype(np.int32)


# ============================================================
# 1. RVQ Compressor (BrainCodec-style)
# ============================================================

class TestRVQCompressor:
    """Tests for RVQCompressor (BrainCodec ICLR 2025, 64x EEG)."""

    @pytest.fixture
    def compressor(self):
        from bci_compression.algorithms.rvq_compressor import RVQCompressor
        return RVQCompressor(n_residuals=4, codebook_size=64, segment_len=64)

    def test_compress_decompress_multichannel(self, compressor):
        data = make_eeg(n_ch=4, n_samp=256)
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.shape == data.shape
        assert recon.dtype == np.float32 or recon.dtype == data.dtype

    def test_compress_decompress_1d(self, compressor):
        data = make_eeg(n_ch=1, n_samp=256)[0]
        assert data.ndim == 1
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.ndim == 1
        assert recon.shape == data.shape

    def test_metadata_keys(self, compressor):
        data = make_eeg(n_ch=2, n_samp=128)
        _, meta = compressor.compress(data)
        assert 'compression_ratio' in meta
        assert 'n_channels' in meta
        assert 'n_samples' in meta

    def test_compression_ratio_positive(self, compressor):
        data = make_eeg(n_ch=4, n_samp=256)
        _, meta = compressor.compress(data)
        assert meta['compression_ratio'] > 1.0

    def test_line_length_loss_callable(self):
        from bci_compression.algorithms.rvq_compressor import line_length_loss
        a = np.sin(np.linspace(0, 2 * np.pi, 100)).astype(np.float32)
        b = a + 0.01 * np.random.randn(100).astype(np.float32)
        loss = line_length_loss(a, b)
        assert isinstance(loss, float)
        assert loss >= 0

    @pytest.mark.parametrize("n_ch,n_samp", [(1, 128), (8, 512), (16, 256)])
    def test_various_shapes(self, n_ch, n_samp):
        from bci_compression.algorithms.rvq_compressor import RVQCompressor
        comp = RVQCompressor(n_residuals=2, codebook_size=32, segment_len=64)
        data = make_eeg(n_ch=n_ch, n_samp=n_samp)
        compressed, meta = comp.compress(data)
        recon = comp.decompress(compressed, meta)
        assert recon.shape == data.shape

    def test_get_compression_ratio_estimate(self, compressor):
        ratio = compressor.get_compression_ratio_estimate()
        assert ratio > 1.0


# ============================================================
# 2. DS-CAE Compressor (RAMAN paper)
# ============================================================

class TestDSCAECompressor:
    """Tests for DSCAECompressor (RAMAN arXiv 2504.06996, 150x LFP)."""

    @pytest.fixture
    def compressor(self):
        from bci_compression.algorithms.cae_compression import DSCAECompressor
        return DSCAECompressor(target_ratio=16, latent_dim=4, apply_pruning=True)

    def test_compress_decompress_multichannel(self, compressor):
        data = make_lfp(n_ch=8, n_samp=512)
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.shape == data.shape

    def test_compress_decompress_1d(self, compressor):
        data = make_lfp(n_ch=1, n_samp=256)[0]
        assert data.ndim == 1
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.ndim == 1
        assert recon.shape == data.shape

    def test_metadata_keys(self, compressor):
        data = make_lfp(n_ch=4, n_samp=256)
        _, meta = compressor.compress(data)
        for key in ('compression_ratio', 'n_channels', 'n_samples',
                    'parameter_count', 'achieved_sparsity'):
            assert key in meta, f"Missing key: {key}"

    def test_achieved_sparsity(self, compressor):
        data = make_lfp(n_ch=4, n_samp=256)
        _, meta = compressor.compress(data)
        # Pruning enabled, should have some sparsity
        assert 0 <= meta['achieved_sparsity'] <= 1.0

    def test_no_pruning(self):
        from bci_compression.algorithms.cae_compression import DSCAECompressor
        comp = DSCAECompressor(target_ratio=8, apply_pruning=False)
        data = make_lfp(n_ch=4, n_samp=256)
        _, meta = comp.compress(data)
        assert meta['achieved_sparsity'] == 0.0

    def test_sndr_estimate(self, compressor):
        data = make_lfp(n_ch=4, n_samp=256)
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        sndr = compressor.sndr_estimate(data, recon)
        assert isinstance(sndr, float)

    def test_balanced_stochastic_prune(self):
        from bci_compression.algorithms.cae_compression import (
            DepthwiseSeparableConv, balanced_stochastic_prune
        )
        layer = DepthwiseSeparableConv(4, 8, kernel=3, stride=1)
        sp = balanced_stochastic_prune(layer, sparsity=0.5)
        assert 0.0 <= sp <= 1.0

    def test_compression_ratio_positive(self, compressor):
        data = make_lfp(n_ch=8, n_samp=512)
        _, meta = compressor.compress(data)
        assert meta['compression_ratio'] > 1.0


# ============================================================
# 3. LLC-Spike Compressor (CLEM)
# ============================================================

class TestLLCSpikeCompressor:
    """Tests for LLCSpikeCompressor (IEEE TIP 2025, lossless spike trains)."""

    @pytest.fixture
    def compressor(self):
        from bci_compression.algorithms.llc_spike_compressor import LLCSpikeCompressor
        return LLCSpikeCompressor(frame_size=16, max_count=8)

    def test_lossless_multichannel(self, compressor):
        """Core requirement: bit-exact reconstruction."""
        spikes = make_spike_train(n_ch=4, n_samp=320, rate=0.05)
        compressed, meta = compressor.compress(spikes)
        recon = compressor.decompress(compressed, meta)
        assert recon.shape == spikes.shape
        assert np.array_equal(recon, spikes), "Spike compressor must be lossless!"

    def test_lossless_1d(self, compressor):
        spikes_1d = make_spike_train(n_ch=1, n_samp=320)[0]
        assert spikes_1d.ndim == 1
        compressed, meta = compressor.compress(spikes_1d)
        recon = compressor.decompress(compressed, meta)
        assert recon.ndim == 1
        assert np.array_equal(recon, spikes_1d), "1D spike compression must be lossless!"

    def test_metadata_lossless_flag(self, compressor):
        spikes = make_spike_train(n_ch=2, n_samp=160)
        _, meta = compressor.compress(spikes)
        assert meta.get('lossless') is True

    def test_compression_ratio_positive(self, compressor):
        spikes = make_spike_train(n_ch=4, n_samp=320, rate=0.05)
        _, meta = compressor.compress(spikes)
        assert meta['compression_ratio'] > 0.5  # lossless may not always exceed 1x

    def test_spike_aggregator(self):
        from bci_compression.algorithms.llc_spike_compressor import SpikeAggregator
        agg = SpikeAggregator(frame_size=8)
        train = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int32)
        frames, orig_len = agg.aggregate(train)
        assert len(frames) == 2
        assert int(frames[0]) == 3  # first 8: 1+0+1+0+0+1+0+0
        assert int(frames[1]) == 4  # second 8: 1+1+0+0+0+0+1+1

    def test_clem_fit_encode_decode(self):
        from bci_compression.algorithms.llc_spike_compressor import CLEMEntropyModel
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 5, size=64).astype(np.int32)
        model = CLEMEntropyModel(max_count=8, smoothing=1.0)
        model.fit(frames)
        encoded = model.encode_frames(frames)
        decoded = model.decode_frames(encoded, len(frames))
        assert np.array_equal(decoded, frames), "CLEM encode/decode must match!"

    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.10])
    def test_lossless_at_various_rates(self, rate):
        from bci_compression.algorithms.llc_spike_compressor import LLCSpikeCompressor
        comp = LLCSpikeCompressor(frame_size=16, max_count=16)
        spikes = make_spike_train(n_ch=2, n_samp=256, rate=rate)
        compressed, meta = comp.compress(spikes)
        recon = comp.decompress(compressed, meta)
        assert np.array_equal(recon, spikes), f"Must be lossless at rate {rate}"


# ============================================================
# 4. Diffusion Compressor (EEGCiD)
# ============================================================

class TestDiffusionCompressor:
    """Tests for DiffusionCompressor (EEGCiD, extreme-ratio EEG compression)."""

    @pytest.fixture
    def compressor(self):
        from bci_compression.algorithms.diffusion_compressor import DiffusionCompressor
        return DiffusionCompressor(latent_ratio=16, n_diff_steps=5, sampling_rate=256.0)

    def test_compress_decompress_multichannel(self, compressor):
        data = make_eeg(n_ch=4, n_samp=256)
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.shape == data.shape

    def test_compress_decompress_1d(self, compressor):
        data = make_eeg(n_ch=1, n_samp=256)[0]
        assert data.ndim == 1
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        assert recon.ndim == 1
        assert recon.shape == data.shape

    def test_metadata_keys(self, compressor):
        data = make_eeg(n_ch=4, n_samp=256)
        _, meta = compressor.compress(data)
        for key in ('compression_ratio', 'n_channels', 'n_samples', 'latent_ratio'):
            assert key in meta, f"Missing key: {key}"

    def test_compression_ratio_high(self, compressor):
        """Diffusion compressor should achieve high compression ratio."""
        data = make_eeg(n_ch=4, n_samp=512)
        _, meta = compressor.compress(data)
        assert meta['compression_ratio'] > 5.0  # expect extreme compression

    def test_set_inference_steps(self, compressor):
        compressor.set_inference_steps(3)
        assert compressor.n_diff_steps == 3

    def test_reconstruction_quality_metrics(self, compressor):
        data = make_eeg(n_ch=2, n_samp=256)
        compressed, meta = compressor.compress(data)
        recon = compressor.decompress(compressed, meta)
        metrics = compressor.get_reconstruction_quality_estimate(data, recon)
        assert 'snr_db' in metrics
        assert 'rms_error' in metrics
        assert 'spectral_snr_alpha_db' in metrics

    def test_score_network_denoise(self):
        from bci_compression.algorithms.diffusion_compressor import EEGScoreNetwork
        net = EEGScoreNetwork(sampling_rate=256.0, n_steps=5)
        x = RNG.standard_normal((4, 256)).astype(np.float32)
        denoised = net.denoise_step(x, t_idx=2, noise_level=0.5)
        assert denoised.shape == x.shape

    @pytest.mark.parametrize("latent_ratio", [8, 32, 64])
    def test_various_latent_ratios(self, latent_ratio):
        from bci_compression.algorithms.diffusion_compressor import DiffusionCompressor
        comp = DiffusionCompressor(latent_ratio=latent_ratio, n_diff_steps=3)
        data = make_eeg(n_ch=2, n_samp=256)
        compressed, meta = comp.compress(data)
        recon = comp.decompress(compressed, meta)
        assert recon.shape == data.shape

    def test_reverse_diffuse(self):
        from bci_compression.algorithms.diffusion_compressor import EEGScoreNetwork
        net = EEGScoreNetwork(sampling_rate=256.0, n_steps=5)
        latent = RNG.standard_normal((4, 16)).astype(np.float32)
        recon = net.reverse_diffuse(latent, target_shape=(4, 128), seed=0)
        assert recon.shape == (4, 128)


# ============================================================
# 5. __init__.py Integration Tests
# ============================================================

class TestAlgorithmsInit:
    """Verify the new algorithms are properly registered in __init__.py."""

    def test_rvq_compressor_importable(self):
        from bci_compression.algorithms import RVQCompressor
        assert RVQCompressor is not None

    def test_dscae_compressor_importable(self):
        from bci_compression.algorithms import DSCAECompressor
        assert DSCAECompressor is not None

    def test_llcspike_compressor_importable(self):
        from bci_compression.algorithms import LLCSpikeCompressor
        assert LLCSpikeCompressor is not None

    def test_diffusion_compressor_importable(self):
        from bci_compression.algorithms import DiffusionCompressor
        assert DiffusionCompressor is not None

    def test_feature_flags_present(self):
        import bci_compression.algorithms as alg
        assert hasattr(alg, '_has_rvq')
        assert hasattr(alg, '_has_cae')
        assert hasattr(alg, '_has_llcspike')
        assert hasattr(alg, '_has_diffusion')
        assert alg._has_rvq is True
        assert alg._has_cae is True
        assert alg._has_llcspike is True
        assert alg._has_diffusion is True
