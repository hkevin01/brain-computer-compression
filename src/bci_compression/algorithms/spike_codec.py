"""Spike-centric codec for neural action potentials.

Pipeline:
 1. Detect threshold crossings (robust MAD-based sigma)
 2. Extract fixed-length snippets around peak
 3. Perform (or update) PCA basis over snippets
 4. Project snippets -> low-dimensional coefficients
 5. Quantize coefficients + peak amplitude + width
 6. Pack events as compact binary stream

Metrics computed (if optional ground truth spike times provided via metadata or argument):
  - detection_f1
  - timing_jitter_ms (median absolute timing error)
  - isi_divergence (symmetric KL of ISI distributions)

This module focuses on efficient event representation; waveform reconstruction
is approximate (optional) and not required for many BCI pipelines which operate
on spike times & features directly.
"""
from __future__ import annotations
import numpy as np
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from .core import BaseCompressor, Config, CompressionError

# ----------------------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------------------


@dataclass
class SpikeEvent:
    channel: int
    t_sample: int
    peak: float
    width: float
    coeffs: np.ndarray  # PCA coeffs (float)

# ----------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------


def mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-9


def extract_snippets(data: np.ndarray, indices: np.ndarray, win_pre: int, win_post: int) -> np.ndarray:
    snippets = []
    n = data.shape[-1]
    for idx in indices:
        start = max(0, idx - win_pre)
        end = min(n, idx + win_post)
        if idx - win_pre >= 0 and idx + win_post <= n:
            snip = data[idx - win_pre: idx + win_post]
        else:
            snip = np.pad(data[start:end], (max(0, win_pre - idx), max(0, idx + win_post - n)), mode='constant')
        snippets.append(snip)
    return np.stack(snippets, axis=0) if snippets else np.empty((0, win_pre + win_post))


def compute_width(wave: np.ndarray, peak_index: int) -> float:
    peak_val = wave[peak_index]
    half = 0.5 * peak_val
    left = peak_index
    while left > 0 and wave[left] > half:
        left -= 1
    right = peak_index
    while right < len(wave) - 1 and wave[right] > half:
        right += 1
    return float(right - left)


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-12
    q = q + 1e-12
    p /= p.sum()
    q /= q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))

# ----------------------------------------------------------------------------------
# Spike Codec
# ----------------------------------------------------------------------------------


class SpikeCodec(BaseCompressor):
    """Spike-centric compressor producing event stream.

    Compressed format layout (little endian):
      Header:
        magic (4 bytes) b'SPK1'
        uint16: version (1)
        uint16: snippet_len
        uint16: n_components (PCA)
        uint16: quant_bits
        uint32: n_events
      PCA basis (float32) shape (n_components, snippet_len)
      Events (variable): per event
        uint16 channel
        uint32 t_sample
        int16 peak_q
        uint16 width_q
        coeffs: int16 * n_components
      Scales (float32 * (n_components + 2)) appended at end for dequantization:
        peak_scale, width_scale, coeff_scales[n_components]
    """

    def __init__(self,
                 threshold: float = 4.0,
                 win_pre: int = 16,
                 win_post: int = 24,
                 n_components: int = 3,
                 quant_bits: int = 12,
                 adaptive: bool = True,
                 config: Optional[Config] = None):
        super().__init__(name="spike_codec", config=config)
        self.threshold = threshold
        self.win_pre = win_pre
        self.win_post = win_post
        self.snippet_len = win_pre + win_post
        self.n_components = n_components
        self.quant_bits = quant_bits
        self.adaptive = adaptive
        self._pca_basis: Optional[np.ndarray] = None  # (n_components, snippet_len)
        self._mean_wave: Optional[np.ndarray] = None

    # ---------------- Internal PCA helpers ----------------
    def _fit_pca(self, snippets: np.ndarray) -> None:
        if snippets.shape[0] == 0:
            raise CompressionError("No snippets to fit PCA")
        mean = snippets.mean(axis=0, keepdims=True)
        centered = snippets - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:self.n_components]
        self._pca_basis = basis
        self._mean_wave = mean.squeeze(0)

    def _project(self, snippets: np.ndarray) -> np.ndarray:
        if self._pca_basis is None:
            self._fit_pca(snippets)
        centered = snippets - self._mean_wave
        return centered @ self._pca_basis.T

    def _reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        if self._pca_basis is None or self._mean_wave is None:
            raise CompressionError("PCA basis not fitted")
        return coeffs @ self._pca_basis + self._mean_wave

    # ---------------- Compression ----------------
    def _detect_spikes_channel(self, x: np.ndarray) -> np.ndarray:
        sigma = mad_sigma(x)
        thr = self.threshold * sigma
        crossings = np.where(x > thr)[0]
        keep = []
        last = -self.snippet_len
        for idx in crossings:
            if idx - last < self.snippet_len:
                if keep and x[idx] > x[keep[-1]]:
                    keep[-1] = idx
                continue
            keep.append(idx)
            last = idx
        return np.array(keep, dtype=np.int32)

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        if data.ndim == 1:
            data = data[None, :]
        n_channels, n_samples = data.shape
        events: List[SpikeEvent] = []
        for ch in range(n_channels):
            idxs = self._detect_spikes_channel(data[ch])
            if idxs.size == 0:
                continue
            snippets = extract_snippets(data[ch], idxs, self.win_pre, self.win_post)
            if snippets.shape[0] == 0:
                continue
            coeffs = self._project(snippets)
            peaks = snippets.max(axis=1)
            peak_pos = snippets.argmax(axis=1)
            widths = [compute_width(wave, pidx) for wave, pidx in zip(snippets, peak_pos)]
            for i, t in enumerate(idxs):
                events.append(SpikeEvent(channel=ch, t_sample=int(t), peak=float(peaks[i]), width=float(widths[i]), coeffs=coeffs[i]))
        if not events:
            return b'', {'events': 0, 'pca_components': 0}
        peak_vals = np.array([e.peak for e in events])
        width_vals = np.array([e.width for e in events])
        coeff_matrix = np.stack([e.coeffs for e in events], axis=0)
        peak_scale = peak_vals.max() or 1.0
        width_scale = width_vals.max() or 1.0
        coeff_scales = np.maximum(np.abs(coeff_matrix).max(axis=0), 1e-6)
        qmax = 2 ** (self.quant_bits - 1) - 1
        peak_q = np.round(peak_vals / peak_scale * qmax).astype(np.int16)
        width_q = np.round(width_vals / width_scale * qmax).astype(np.uint16)
        coeff_q = np.round(coeff_matrix / coeff_scales * qmax).astype(np.int16)
        buf = bytearray()
        buf.extend(b'SPK1')
        header = struct.pack('<HHHHI', 1, self.snippet_len, self.n_components, self.quant_bits, len(events))
        buf.extend(header)
        if self._pca_basis is None:
            raise CompressionError("PCA basis missing")
        buf.extend(self._pca_basis.astype(np.float32).tobytes())
        for i, ev in enumerate(events):
            buf.extend(struct.pack('<H I h H', ev.channel, ev.t_sample, peak_q[i], width_q[i]))
            buf.extend(coeff_q[i].tobytes())
        buf.extend(struct.pack('<f f', peak_scale, width_scale))
        buf.extend(coeff_scales.astype(np.float32).tobytes())
        metadata: Dict[str, Any] = {
            'events': len(events),
            'pca_components': self.n_components,
            'snippet_len': self.snippet_len,
        }
        return bytes(buf), metadata

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        if not compressed:
            return np.empty((0,), dtype=np.float32)
        view = memoryview(compressed)
        if view[:4].tobytes() != b'SPK1':
            raise CompressionError("Invalid magic header for spike codec")
        offset = 4
        version, snippet_len, n_components, quant_bits, n_events = struct.unpack_from('<HHHHI', view, offset)
        offset += struct.calcsize('<HHHHI')
        basis_size = n_components * snippet_len * 4
        _ = np.frombuffer(view[offset:offset + basis_size], dtype=np.float32).reshape(n_components, snippet_len)
        offset += basis_size
        events = []
        for _i in range(n_events):
            ch, t_sample, peak_q, width_q = struct.unpack_from('<H I h H', view, offset)
            offset += struct.calcsize('<H I h H')
            coeff_q = np.frombuffer(view[offset:offset + n_components * 2], dtype=np.int16)
            offset += n_components * 2
            events.append((ch, t_sample, peak_q, width_q, coeff_q))
        peak_scale, width_scale = struct.unpack_from('<f f', view, offset)
        offset += struct.calcsize('<f f')
        _ = np.frombuffer(view[offset:offset + n_components * 4], dtype=np.float32)
        spike_times = []
        spike_channels = []
        for ch, t, _p_q, _w_q, _c_q in events:
            spike_times.append(t)
            spike_channels.append(ch)
        return np.array(list(zip(spike_channels, spike_times)), dtype=np.int32)

    # ---------------- Metrics (external helper) ----------------
    def compute_metrics(self, detected_times: np.ndarray, gt_times: Optional[np.ndarray], fs: float) -> Dict[str, float]:
        if gt_times is None or gt_times.size == 0:
            return {'detection_f1': 0.0, 'timing_jitter_ms': 0.0, 'isi_divergence': 0.0}
        tolerance = int(0.001 * fs)  # 1 ms
        det = np.sort(detected_times)
        gt = np.sort(gt_times)
        i = j = 0
        matches = 0
        jitters = []
        while i < len(det) and j < len(gt):
            dt = det[i] - gt[j]
            if abs(dt) <= tolerance:
                matches += 1
                jitters.append(abs(dt) / fs * 1000.0)
                i += 1
                j += 1
            elif det[i] < gt[j]:
                i += 1
            else:
                j += 1
        precision = matches / len(det) if len(det) else 0.0
        recall = matches / len(gt) if len(gt) else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        isi_det = np.diff(det) if len(det) > 1 else np.array([])
        isi_gt = np.diff(gt) if len(gt) > 1 else np.array([])
        if isi_det.size == 0 or isi_gt.size == 0:
            isi_div = 0.0
        else:
            bins = np.linspace(0, max(isi_det.max(), isi_gt.max()), 32)
            p, _ = np.histogram(isi_det, bins=bins)
            q, _ = np.histogram(isi_gt, bins=bins)
            isi_div = symmetric_kl(p.astype(float), q.astype(float))
        return {
            'detection_f1': float(f1),
            'timing_jitter_ms': float(np.median(jitters) if jitters else 0.0),
            'isi_divergence': float(isi_div),
        }


def create_spike_codec(**kwargs: Any) -> SpikeCodec:  # type: ignore[name-defined]
    return SpikeCodec(**kwargs)
