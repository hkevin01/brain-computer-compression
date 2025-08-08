"""Adaptive algorithm selector for windowed neural/EMG data.

Provides lightweight, real‑time friendly feature extraction and heuristic
scoring across available compressors (e.g. transformer, VAE, neural_lz).
Includes hysteresis logic to avoid rapid flapping between algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FeatureDict = Dict[str, float]


def bandpower(x: np.ndarray, fs: float, bands: List[Tuple[float, float]]) -> Dict[str, float]:
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
    if x.size == 0:
        return 0.0
    sigma = np.median(np.abs(x)) / 0.6745 + 1e-6
    spikes = np.abs(x) > (threshold * sigma)
    return float(spikes.sum() / x.shape[-1])


def channel_corr(x: np.ndarray) -> float:
    if x.ndim < 2 or x.shape[0] < 2:
        return 0.0
    c = np.corrcoef(x)
    if np.isnan(c).any():
        return 0.0
    upper = c[np.triu_indices(c.shape[0], k=1)]
    return float(np.mean(upper))


def kurtosis(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2) + 1e-9
    return float(np.mean(((x - m) ** 4) / (s2 ** 2)))


@dataclass
class AdaptiveSelectorConfig:
    fs: float = 1000.0
    hysteresis: int = 3
    switch_threshold: float = 0.15
    bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    )


class AdaptiveSelector:
    """Selects best compressor/parameters per window with hysteresis.

    The scoring heuristics are placeholders and should be replaced or augmented
    with learned models / reinforcement learning in future phases.
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
        # Simple heuristic scoring — domain‑inspired weights
        scores: Dict[str, float] = {
            'transformer': feats.get('band_2', 0.0) + 0.5 * feats.get('band_3', 0.0),  # beta/gamma rich
            'vae': feats.get('kurtosis', 0.0) * 0.1 + feats.get('band_1', 0.0),  # non-Gaussian + theta
            'neural_lz': feats.get('corr', 0.0) * 0.8 + (1 - feats.get('spike_rate', 0.0)),
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
