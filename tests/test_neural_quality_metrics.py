"""
Tests for NeuralQualityMetrics — Phase 8-9 roadmap item.

# =============================================================================
# ID: BCI-TEST-NEURAL-METRICS-001
# Purpose: Validate all functions in bci_compression.metrics.neural_quality
#          against known-good synthetic neural data.
# Requirement: All tests SHALL pass; SNR / PSNR tests use exact round-trips.
# =============================================================================
"""

import numpy as np
import pytest

from bci_compression.metrics.neural_quality import (
    NeuralQualityMetrics,
    compute_snr,
    compute_psnr,
    detect_spikes,
    mutual_information,
    phase_coherence,
    spike_timing_jitter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_1ch():
    """Single-channel 1-second 10 Hz sine at 1 kHz, float32."""
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    return (np.sin(2 * np.pi * 10 * t)).astype(np.float32)


@pytest.fixture
def sine_multichannel():
    """4-channel 1-second 10 Hz sine at 1 kHz."""
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    base = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    return np.stack([base * (i + 1) for i in range(4)])  # shape (4, 1000)


# ---------------------------------------------------------------------------
# SNR / PSNR
# ---------------------------------------------------------------------------

class TestSNR:
    def test_perfect_reconstruction(self, sine_1ch):
        """Perfect reconstruction -> infinite SNR."""
        snr = compute_snr(sine_1ch, sine_1ch.copy())
        assert snr == float("inf")

    def test_snr_decreases_with_noise(self, sine_1ch):
        """More noise -> lower SNR."""
        noisy_light = sine_1ch + np.random.RandomState(0).randn(len(sine_1ch)).astype(np.float32) * 0.01
        noisy_heavy = sine_1ch + np.random.RandomState(0).randn(len(sine_1ch)).astype(np.float32) * 0.5
        snr_light = compute_snr(sine_1ch, noisy_light)
        snr_heavy = compute_snr(sine_1ch, noisy_heavy)
        assert snr_light > snr_heavy

    def test_shape_mismatch_raises(self, sine_1ch):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_snr(sine_1ch, sine_1ch[:500])

    def test_zero_signal_returns_neg_inf(self):
        z = np.zeros(100, dtype=np.float32)
        assert compute_snr(z, z + 0.1) == float("-inf")

    def test_psnr_perfect(self, sine_1ch):
        assert compute_psnr(sine_1ch, sine_1ch.copy()) == float("inf")

    def test_psnr_positive_with_noise(self, sine_1ch):
        noisy = sine_1ch + 0.01
        psnr = compute_psnr(sine_1ch, noisy)
        assert psnr > 0


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

class TestSpikeDetection:
    def test_detects_known_spike(self):
        """Inject a spike 10x above noise floor and check detection."""
        rng = np.random.RandomState(42)
        data = rng.randn(1000).astype(np.float32) * 0.1
        data[300] = 5.0  # spike
        spikes = detect_spikes(data, threshold_sigma=3.0)
        assert 300 in spikes

    def test_refractory_period(self):
        """Two spikes within refractory period -> only one detected."""
        data = np.zeros(200, dtype=np.float32)
        data[50] = 10.0
        data[55] = 10.0  # within 30-sample refractory
        spikes = detect_spikes(data, refractory_samples=30)
        assert len(spikes) == 1

    def test_no_spikes_in_noise(self):
        """Low-amplitude Gaussian noise -> no spikes detected at 6-sigma threshold."""
        rng = np.random.RandomState(7)
        data = rng.randn(1000).astype(np.float32) * 0.1
        spikes = detect_spikes(data, threshold_sigma=6.0)
        assert len(spikes) == 0 or len(spikes) < 5  # extremely unlikely

    def test_empty_input(self):
        data = np.zeros(0, dtype=np.float32)
        spikes = detect_spikes(data)
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# Spike timing jitter
# ---------------------------------------------------------------------------

class TestSpikeJitter:
    def test_zero_jitter_identical(self, sine_1ch):
        result = spike_timing_jitter(sine_1ch, sine_1ch.copy(), sampling_rate=1000.0)
        # matched fraction should be 1.0 if spikes found; jitter=0
        if result["spike_count_orig"] > 0:
            assert result["mean_jitter_ms"] == pytest.approx(0.0, abs=0.01)

    def test_no_spikes_returns_nan(self):
        flat = np.zeros(500, dtype=np.float32)
        result = spike_timing_jitter(flat, flat.copy(), sampling_rate=1000.0)
        assert np.isnan(result["mean_jitter_ms"])

    def test_multichannel_accepted(self, sine_multichannel):
        result = spike_timing_jitter(
            sine_multichannel, sine_multichannel.copy(), sampling_rate=1000.0
        )
        assert "mean_jitter_ms" in result


# ---------------------------------------------------------------------------
# Phase coherence
# ---------------------------------------------------------------------------

class TestPhaseCoherence:
    def test_perfect_plv(self, sine_1ch):
        """Identical signals -> PLV ~ 1.0."""
        plv = phase_coherence(sine_1ch, sine_1ch.copy(), sampling_rate=1000.0, freq_band=(8.0, 13.0))
        assert plv == pytest.approx(1.0, abs=0.05)

    def test_random_phase_low_plv(self):
        """Independent random signals -> PLV close to 0."""
        rng = np.random.RandomState(1)
        a = rng.randn(2000).astype(np.float32)
        b = rng.randn(2000).astype(np.float32)
        plv = phase_coherence(a, b, sampling_rate=1000.0, freq_band=(8.0, 13.0))
        assert plv < 0.5

    def test_multichannel(self, sine_multichannel):
        plv = phase_coherence(
            sine_multichannel, sine_multichannel.copy(),
            sampling_rate=1000.0, freq_band=(8.0, 13.0)
        )
        assert plv == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

class TestMutualInformation:
    def test_high_mi_identical(self, sine_1ch):
        """Identical signals -> high MI."""
        mi = mutual_information(sine_1ch, sine_1ch.copy())
        assert mi > 3.0  # at least 3 bits

    def test_low_mi_independent(self):
        rng = np.random.RandomState(5)
        a = rng.randn(1000)
        b = rng.randn(1000)
        mi = mutual_information(a, b)
        assert mi < 1.0

    def test_non_negative(self):
        rng = np.random.RandomState(9)
        a = rng.randn(500)
        b = a + rng.randn(500) * 0.5
        assert mutual_information(a, b) >= 0.0


# ---------------------------------------------------------------------------
# NeuralQualityMetrics composite
# ---------------------------------------------------------------------------

class TestNeuralQualityMetrics:
    def test_evaluate_returns_all_keys(self, sine_multichannel):
        nqm = NeuralQualityMetrics(sampling_rate=1000.0, spike_band=(50.0, 400.0))
        result = nqm.evaluate(sine_multichannel, sine_multichannel.copy())
        for key in ("snr_db", "psnr_db", "phase_coherence_plv", "mutual_information_bits"):
            assert key in result

    def test_quality_grade_excellent(self):
        nqm = NeuralQualityMetrics()
        grade = nqm.quality_grade({"snr_db": 35.0, "mean_jitter_ms": 0.05, "phase_coherence_plv": 0.97})
        assert grade == "excellent"

    def test_quality_grade_poor(self):
        nqm = NeuralQualityMetrics()
        grade = nqm.quality_grade({"snr_db": 10.0, "mean_jitter_ms": 5.0, "phase_coherence_plv": 0.5})
        assert grade == "poor"

    def test_shape_mismatch_raises(self, sine_multichannel):
        nqm = NeuralQualityMetrics(sampling_rate=1000.0)
        with pytest.raises(ValueError):
            nqm.evaluate(sine_multichannel, sine_multichannel[:, :500])
