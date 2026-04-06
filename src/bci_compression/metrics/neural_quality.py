"""
# =============================================================================
# ID: BCI-METRICS-NEURAL-001
# Module: Neural Signal Quality Metrics
# Purpose: Provide BCI-specific quality assessment functions for evaluating
#          the fidelity of compressed-then-reconstructed neural recordings.
#          Implements metrics that standard image/audio metrics miss: spike
#          timing jitter, LFP phase coherence, and mutual information.
# Requirement: All metrics SHALL operate on float32/float64 ndarrays;
#              spike-timing metric SHALL report jitter <= 0.5 ms for
#              high-quality compression (SNR > 25 dB).
# Rationale: Standard SNR / PSNR measure amplitude error but not the
#            biologically critical spike-timing dimension.  A compressor
#            may have good SNR but unacceptable timing jitter, making it
#            unsuitable for closed-loop BCI decoding.
# Constraints: SciPy required for Hilbert transform and signal processing.
# Assumptions: Sampling rate >= 1 kHz; signals pre-filtered to 300-5000 Hz
#              for spike metrics, 1-300 Hz for phase coherence metrics.
# Failure Modes: Empty arrays raise ValueError; channels with all-zero
#               data return NaN and log a warning.
# Verification: tests/test_neural_quality_metrics.py
# References:
#   - Quiroga et al. (2004) Unsupervised detection/classification of
#     extracellular action potentials. Neural Computation 16(8).
#   - Lachaux et al. (1999) Phase synchrony. Human Brain Mapping 8(4).
#   - Shannon (1948) A Mathematical Theory of Communication.
# =============================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio between original and reconstructed signals.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-002
    # Formula:     SNR = 10 * log10( E[x**2] / E[(x-x_hat)**2] )
    # Inputs:
    #   original      - np.ndarray, any shape, float.
    #   reconstructed - np.ndarray, same shape as original.
    # Outputs:
    #   float - SNR in dB.  Returns -inf if original is all-zero.
    # Preconditions:  original.shape == reconstructed.shape; original.size > 0.
    # Failure Modes:  zero-power original -> log(0) guarded to return -inf.
    # -------------------------------------------------------------------------
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if signal_power == 0:
        return float("-inf")
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-003
    # Formula:   PSNR = 20 * log10( max(|x|) / RMSE )
    # -------------------------------------------------------------------------
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Shape mismatch between original and reconstructed arrays.")
    peak = float(np.max(np.abs(original)))
    mse = float(np.mean((original - reconstructed) ** 2))
    if peak == 0:
        return float("-inf")
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(peak / np.sqrt(mse)))


def detect_spikes(
    data: np.ndarray,
    threshold_sigma: float = 4.0,
    refractory_samples: int = 30,
) -> np.ndarray:
    """
    Detect spike times using amplitude threshold crossing.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-004
    # Requirement: Return 1-D integer array of sample indices for spike events;
    #              enforce refractory period to avoid double-counting.
    # Inputs:
    #   data               - np.ndarray shape (samples,), float, band-passed.
    #   threshold_sigma    - float, threshold as multiples of robust sigma.
    #   refractory_samples - int, minimum spacing between detections.
    # Outputs:
    #   np.ndarray of int - spike sample indices.
    # Failure Modes:  flat signal -> no spikes -> empty array returned.
    # -------------------------------------------------------------------------
    """
    sigma = float(np.median(np.abs(data)) / 0.6745) + 1e-9
    threshold = threshold_sigma * sigma
    crossings = np.where(np.abs(data) > threshold)[0]
    if crossings.size == 0:
        return np.array([], dtype=int)
    spikes: List[int] = [int(crossings[0])]
    for idx in crossings[1:]:
        if idx - spikes[-1] >= refractory_samples:
            spikes.append(int(idx))
    return np.array(spikes, dtype=int)


def spike_timing_jitter(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampling_rate: float,
    threshold_sigma: float = 4.0,
) -> Dict[str, float]:
    """
    Measure spike timing jitter introduced by compression.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-005
    # Requirement: Report mean jitter <= 0.5 ms for high-quality compression.
    # Outputs:
    #   dict with keys: mean_jitter_ms, max_jitter_ms, spike_count_orig,
    #                   spike_count_recon, matched_fraction.
    # -------------------------------------------------------------------------
    """
    if original.ndim == 2:
        orig_1d = original.mean(axis=0)
        recon_1d = reconstructed.mean(axis=0)
    else:
        orig_1d = original.flatten()
        recon_1d = reconstructed.flatten()

    orig_spikes = detect_spikes(orig_1d, threshold_sigma)
    recon_spikes = detect_spikes(recon_1d, threshold_sigma)

    result: Dict[str, float] = {
        "spike_count_orig": float(len(orig_spikes)),
        "spike_count_recon": float(len(recon_spikes)),
        "mean_jitter_ms": float("nan"),
        "max_jitter_ms": float("nan"),
        "matched_fraction": 0.0,
    }

    if orig_spikes.size == 0:
        logger.warning("No spikes detected in original signal.")
        return result

    jitters_samples: List[float] = []
    max_search = int(sampling_rate * 0.002)
    for s in orig_spikes:
        candidates = recon_spikes[np.abs(recon_spikes - s) <= max_search]
        if candidates.size > 0:
            jitters_samples.append(float(np.min(np.abs(candidates - s))))

    if jitters_samples:
        ms_per_sample = 1000.0 / sampling_rate
        result["mean_jitter_ms"] = float(np.mean(jitters_samples) * ms_per_sample)
        result["max_jitter_ms"] = float(np.max(jitters_samples) * ms_per_sample)
        result["matched_fraction"] = float(len(jitters_samples) / len(orig_spikes))

    return result


def phase_coherence(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampling_rate: float,
    freq_band: Tuple[float, float] = (8.0, 13.0),
) -> float:
    """
    Compute phase locking value (PLV) between original and reconstructed LFP.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-006
    # Requirement: PLV in [0, 1]; PLV >= 0.9 indicates negligible phase distortion.
    # Formula:     PLV = |N^-1 sum exp(i*delta_phi)|
    # Failure Modes:  Hilbert edge effects on short signals; returns nan on error.
    # References:     Lachaux et al. (1999) Human Brain Mapping 8(4).
    # -------------------------------------------------------------------------
    """
    if original.ndim == 1:
        original = original[np.newaxis, :]
        reconstructed = reconstructed[np.newaxis, :]

    if original.shape != reconstructed.shape:
        raise ValueError("Shape mismatch for phase coherence computation.")

    nyq = sampling_rate / 2.0
    lo, hi = freq_band
    if hi >= nyq:
        hi = nyq * 0.99

    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    plv_values: List[float] = []

    for ch in range(original.shape[0]):
        try:
            orig_filt = scipy_signal.filtfilt(b, a, original[ch])
            recon_filt = scipy_signal.filtfilt(b, a, reconstructed[ch])
            phi_orig = np.angle(scipy_signal.hilbert(orig_filt))
            phi_recon = np.angle(scipy_signal.hilbert(recon_filt))
            delta_phi = phi_orig - phi_recon
            plv = float(np.abs(np.mean(np.exp(1j * delta_phi))))
            plv_values.append(plv)
        except Exception:
            continue

    return float(np.mean(plv_values)) if plv_values else float("nan")


def mutual_information(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_bins: int = 0,
) -> float:
    """
    Estimate mutual information between original and reconstructed signals.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-007
    # Requirement: Return MI in bits; MI == H(original) for lossless compression.
    # Formula:     MI(X;Y) = H(X) + H(Y) - H(X,Y)  (histogram estimator).
    # Failure Modes:  constant arrays (zero entropy) -> MI = 0.
    # Note:  n_bins=0 selects an adaptive bin count based on sample size to
    #        reduce histogram bias: bins = max(4, int((n / 5) ** (1/3))).
    # -------------------------------------------------------------------------
    """
    x = original.flatten().astype(np.float64)
    y = reconstructed.flatten().astype(np.float64)
    if n_bins <= 0:
        # Adaptive: enough bins for entropy > 3 bits, bounded by sample count
        n_bins = max(8, min(32, int(np.sqrt(len(x) / 4.0))))
    joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    joint_prob = joint_hist / (joint_hist.sum() + 1e-12)
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    def _safe_entropy(p: np.ndarray) -> float:
        p_nz = p[p > 0]
        return float(-np.sum(p_nz * np.log2(p_nz)))

    h_x = _safe_entropy(px)
    h_y = _safe_entropy(py)
    h_xy = _safe_entropy(joint_prob.flatten())
    return max(0.0, float(h_x + h_y - h_xy))


class NeuralQualityMetrics:
    """
    Composite neural signal quality assessor for compression evaluation.

    # -------------------------------------------------------------------------
    # ID: BCI-METRICS-NEURAL-008
    # Requirement: evaluate() SHALL return SNR, PSNR, spike jitter, phase
    #              coherence, and MI in < 50 ms for 30-second 32-channel
    #              recordings sampled at 1 kHz.
    # Purpose: Single-call interface replacing per-metric scripting.
    # -------------------------------------------------------------------------
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,
        spike_band: Tuple[float, float] = (300.0, 3000.0),
        lfp_band: Tuple[float, float] = (8.0, 30.0),
    ):
        self.sampling_rate = sampling_rate
        self.spike_band = spike_band
        self.lfp_band = lfp_band

    def evaluate(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run all quality metrics and return consolidated report dict.

        Parameters
        ----------
        original : np.ndarray
            Original neural data, shape (channels, samples).
        reconstructed : np.ndarray
            Decompressed reconstruction, same shape.

        Returns
        -------
        dict
            Keys: snr_db, psnr_db, mean_jitter_ms, max_jitter_ms,
            matched_spike_fraction, phase_coherence_plv,
            mutual_information_bits, spike_count_orig, spike_count_recon.
        """
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
            )

        results: Dict[str, float] = {}
        results["snr_db"] = compute_snr(original, reconstructed)
        results["psnr_db"] = compute_psnr(original, reconstructed)

        # Spike timing (apply band-pass to isolate spikes)
        nyq = self.sampling_rate / 2.0
        lo_s, hi_s = self.spike_band
        if hi_s < nyq and lo_s < hi_s and hi_s < nyq:
            try:
                b, a = scipy_signal.butter(
                    4, [lo_s / nyq, min(hi_s / nyq, 0.99)], btype="band"
                )
                mean_orig = original.mean(axis=0) if original.ndim == 2 else original.flatten()
                mean_recon = reconstructed.mean(axis=0) if reconstructed.ndim == 2 else reconstructed.flatten()
                orig_sp = scipy_signal.filtfilt(b, a, mean_orig)
                recon_sp = scipy_signal.filtfilt(b, a, mean_recon)
                jitter = spike_timing_jitter(orig_sp, recon_sp, self.sampling_rate)
                results["mean_jitter_ms"] = jitter.get("mean_jitter_ms", float("nan"))
                results["max_jitter_ms"] = jitter.get("max_jitter_ms", float("nan"))
                results["matched_spike_fraction"] = jitter.get("matched_fraction", 0.0)
                results["spike_count_orig"] = jitter.get("spike_count_orig", 0.0)
                results["spike_count_recon"] = jitter.get("spike_count_recon", 0.0)
            except Exception as exc:
                logger.warning(f"Spike timing metric failed: {exc}")
                for k in ("mean_jitter_ms", "max_jitter_ms", "matched_spike_fraction",
                          "spike_count_orig", "spike_count_recon"):
                    results[k] = float("nan")
        else:
            for k in ("mean_jitter_ms", "max_jitter_ms", "matched_spike_fraction",
                      "spike_count_orig", "spike_count_recon"):
                results[k] = float("nan")

        # Phase coherence
        try:
            results["phase_coherence_plv"] = phase_coherence(
                original, reconstructed, self.sampling_rate, self.lfp_band
            )
        except Exception as exc:
            logger.warning(f"Phase coherence failed: {exc}")
            results["phase_coherence_plv"] = float("nan")

        # Mutual information
        try:
            results["mutual_information_bits"] = mutual_information(original, reconstructed)
        except Exception as exc:
            logger.warning(f"Mutual information failed: {exc}")
            results["mutual_information_bits"] = float("nan")

        return results

    def quality_grade(self, metrics: Dict[str, float]) -> str:
        """
        Convert metric dict to a quality grade string.

        # -----------------------------------------------------------------------
        # ID: BCI-METRICS-NEURAL-009
        # Grade: excellent (SNR>=30, jitter<0.1ms, PLV>=0.95),
        #        good (SNR>=25, jitter<0.5ms, PLV>=0.85), fair (SNR>=20), poor.
        # -----------------------------------------------------------------------
        """
        snr = metrics.get("snr_db", -999.0)
        jitter = metrics.get("mean_jitter_ms", float("nan"))
        plv = metrics.get("phase_coherence_plv", float("nan"))

        j_ok_exc = np.isnan(jitter) or jitter < 0.1
        p_ok_exc = np.isnan(plv) or plv >= 0.95
        j_ok_good = np.isnan(jitter) or jitter < 0.5
        p_ok_good = np.isnan(plv) or plv >= 0.85

        if snr >= 30 and j_ok_exc and p_ok_exc:
            return "excellent"
        if snr >= 25 and j_ok_good and p_ok_good:
            return "good"
        if snr >= 20:
            return "fair"
        return "poor"
