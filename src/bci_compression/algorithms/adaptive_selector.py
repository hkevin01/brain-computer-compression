"""
# =============================================================================
# ID: BCI-ALG-ADAPT-001
# Module: Adaptive Algorithm Selector
# Purpose: Dynamically select the best-fit compression algorithm for each
#          incoming neural/EMG data window based on real-time signal features,
#          avoiding constant switching through hysteresis control.
# Requirement: Selection latency SHALL be < 0.5 ms on CPU for windows up
#              to 1024 samples × 256 channels.
# Rationale: Neural data characteristics (spike rate, band-power profile,
#            cross-channel correlation) vary over time; a static algorithm
#            choice will be sub-optimal.  Hysteresis prevents rapid flapping
#            that would inflate overhead without quality benefit.
# Constraints: NumPy only (no GPU dependency); Python ≥ 3.8.
# Assumptions: Downstream compressors registered under keys 'transformer',
#              'vae', 'neural_lz'.
# Failure Modes: division-by-zero in score normalisation guarded by `or 1.0`;
#                empty windows return zero feature values.
# Verification: tests/test_adaptive_selector.py
# References:   Reinforcement Learning (Sutton & Barto, 2018) — future
#               replacement for heuristic scoring.
# =============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FeatureDict = Dict[str, float]


def bandpower(x: np.ndarray, fs: float, bands: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Estimate per-band power spectral density via Welch's periodogram approximation.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-002
    # Requirement: Return fractional band power for each (lo, hi) Hz pair;
    #              output keys follow 'band_{i}' convention.
    # Inputs:
    #   x     – np.ndarray shape (..., samples), float.  samples > 0.
    #   fs    – float, sampling rate in Hz.  fs > 0.
    #   bands – list of (lo_hz, hi_hz) tuples; lo < hi < fs/2.
    # Outputs:
    #   dict  – {band_0: float, band_1: float, …} mean PSD per band.
    # Preconditions:  x.size > 0.
    # Postconditions: all values ≥ 0.0.
    # Side Effects:   None.
    # Failure Modes:  Empty x → returns dict with 0.0 for all bands.
    # -------------------------------------------------------------------------
    """
    out: Dict[str, float] = {}
    if x.size == 0:
        return {f"band_{i}": 0.0 for i in range(len(bands))}
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    psd = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    for i, (lo, hi) in enumerate(bands):
        idx = (freqs >= lo) & (freqs <= hi)
        out[f"band_{i}"] = float(psd[..., idx].mean()) if np.any(idx) else 0.0
    return out


def spike_rate(x: np.ndarray, threshold: float = 4.0) -> float:
    """
    Estimate spike rate as the fraction of samples exceeding threshold × σ̂.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-003
    # Requirement: Return spike rate in spikes-per-sample in [0.0, 1.0].
    # Inputs:
    #   x         – np.ndarray, shape (samples,) or (channels, samples).
    #   threshold – float, multiple of estimated noise σ̂ (default 4σ).
    # Outputs:
    #   float – spike_count / total_samples.
    # Preconditions:  x.size > 0.
    # Postconditions: result ∈ [0.0, 1.0].
    # Side Effects:   None.
    # Failure Modes:  Empty x → 0.0.
    # -------------------------------------------------------------------------
    """
    if x.size == 0:
        return 0.0
        return 0.0
    sigma = np.median(np.abs(x)) / 0.6745 + 1e-6
    spikes = np.abs(x) > (threshold * sigma)
    return float(spikes.sum() / x.shape[-1])


def channel_corr(x: np.ndarray) -> float:
    """
    Compute mean upper-triangle Pearson correlation across channels.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-004
    # Requirement: Return mean inter-channel correlation in [-1.0, 1.0].
    # Inputs:  x – np.ndarray shape (channels, samples); channels ≥ 2.
    # Outputs: float – mean of upper-triangle of correlation matrix; 0.0 if
    #          single-channel or NaN present.
    # Preconditions:  x.ndim == 2, channels ≥ 2.
    # -------------------------------------------------------------------------
    """
    if x.ndim < 2 or x.shape[0] < 2:
        return 0.0
    c = np.corrcoef(x)
    if np.isnan(c).any():
        return 0.0
    upper = c[np.triu_indices(c.shape[0], k=1)]
    return float(np.mean(upper))


def kurtosis(x: np.ndarray) -> float:
    """
    Compute excess kurtosis of the flattened signal (Gaussian = 3).

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-005
    # Requirement: Return unitless kurtosis ≥ 0; guard against zero variance.
    # Inputs:  x – np.ndarray, any shape.
    # Outputs: float – fourth standardised moment.
    # Preconditions:  x.size > 0.
    # Failure Modes:  flat signal → variance ≈ 0 → guarded by +1e-9.
    # -------------------------------------------------------------------------
    """
    if x.size == 0:
        return 0.0
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2) + 1e-9
    return float(np.mean(((x - m) ** 4) / (s2 ** 2)))


@dataclass
class AdaptiveSelectorConfig:
    """
    Configuration for AdaptiveSelector.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-006
    # Fields:
    #   fs              – sampling rate (Hz); must be > 0.
    #   hysteresis      – minimum windows before allowing algorithm switch.
    #   switch_threshold – minimum score gap to trigger a switch (0–1).
    #   bands           – frequency band boundaries for power extraction.
    # -------------------------------------------------------------------------
    """
    fs: float = 1000.0
    hysteresis: int = 3
    switch_threshold: float = 0.15
    bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    )


class AdaptiveSelector:
    """
    Selects best compressor/parameters per window with hysteresis.

    # -------------------------------------------------------------------------
    # ID: BCI-ALG-ADAPT-007
    # Requirement: Choose 'transformer', 'vae', or 'neural_lz' for each
    #              incoming data window; hold selected algorithm for at least
    #              config.hysteresis windows to reduce switching overhead.
    # Purpose: Maximise compression efficiency across heterogeneous neural
    #          recording epochs (rest, task, artefact) without manual tuning.
    # Rationale: Heuristic scoring is a domain-inspired placeholder; future
    #            phases will replace with a learned RL policy.
    # Side Effects: Updates self.history, self.hold_counter, self.last_choice.
    # Verification: tests/test_adaptive_selector.py
    # -------------------------------------------------------------------------
    """

    def __init__(self, config: AdaptiveSelectorConfig | None = None):
        self.config = config or AdaptiveSelectorConfig()
        self.history: List[str] = []
        self.hold_counter = 0
        self.last_choice: Optional[str] = None

    def extract_features(self, window: np.ndarray) -> FeatureDict:
        feats: FeatureDict = {}
        feats.update(bandpower(window, self.config.fs, list(self.config.bands)))
        feats['spike_rate'] = spike_rate(window)
        feats['corr'] = channel_corr(window) if window.ndim == 2 else 0.0
        feats['kurtosis'] = kurtosis(window)
        return feats

    def score_algorithms(self, feats: FeatureDict) -> Dict[str, float]:
        # Heuristic scoring — domain-inspired weights.
        # Routing rationale:
        #   transformer   — high beta/gamma power → rich frequency content suits attention
        #   vae           — high kurtosis + theta → non-Gaussian distribution suits VAE prior
        #   neural_lz     — high cross-channel correlation + low spikes → structure exploitable by LZ
        #   rvq           — moderate spike rate + broad EEG/iEEG spectrum (BrainCodec 64x target)
        #   ds_cae        — very low spike rate + low-frequency LFP content (RAMAN 150x LFP target)
        #   llc_spike     — very high spike rate → lossless CLEM entropy model optimal for sparse trains
        #   diffusion     — low-frequency + low kurtosis (smooth EEG) → generative prior reconstructs well
        sr = feats.get('spike_rate', 0.0)
        band_0 = feats.get('band_0', 0.0)  # delta (LFP-dominant)
        band_1 = feats.get('band_1', 0.0)  # theta
        band_2 = feats.get('band_2', 0.0)  # beta
        band_3 = feats.get('band_3', 0.0)  # gamma
        kurt = feats.get('kurtosis', 0.0)
        corr = feats.get('corr', 0.0)
        scores: Dict[str, float] = {
            'transformer': band_2 + 0.5 * band_3,
            'vae': kurt * 0.1 + band_1,
            'neural_lz': corr * 0.8 + (1.0 - sr),
            # 2025-2026 research algorithms
            'rvq': (band_1 + band_2) * 0.5 + kurt * 0.05,       # EEG/iEEG broad-spectrum
            'ds_cae': band_0 * 1.5 + (1.0 - sr) * 0.8,          # LFP: delta-dominant, sparse spikes
            'llc_spike': sr * 2.0,                                # spike trains: high spike rate
            'diffusion': band_0 * 0.8 + (1.0 - kurt * 0.1),     # smooth long EEG recordings
        }
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def apply_hysteresis(self, choice: str, scores: Dict[str, float]) -> str:
        if self.last_choice is None:
            self.last_choice = choice
            self.hold_counter = 0
            return choice
        if choice != self.last_choice:
            improvement = scores[choice] - scores[self.last_choice]
            if improvement > self.config.switch_threshold and self.hold_counter >= self.config.hysteresis:
                self.last_choice = choice
                self.hold_counter = 0
            else:
                self.hold_counter += 1
        else:
            self.hold_counter = 0
        return self.last_choice

    def select(self, window: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        feats = self.extract_features(window)
        scores = self.score_algorithms(feats)
        top = max(scores.keys(), key=lambda k: scores[k])
        final = self.apply_hysteresis(top, scores)
        decision: Dict[str, Any] = {
            'features': feats,
            'scores': scores,
            'selected': final,
            'hysteresis_hold': self.hold_counter,
        }
        self.history.append(final)
        return final, decision


def create_adaptive_selector(**kwargs: Any) -> AdaptiveSelector:
    return AdaptiveSelector(**kwargs)
