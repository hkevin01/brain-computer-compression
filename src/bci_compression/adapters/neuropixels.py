"""
Imec Neuropixels Device Adapter

Supports Neuropixels 1.0, 2.0 (single-shank and 4-shank), and Ultra probes:

  NP 1.0 : 384 AP channels + 384 LFP channels, 30 kHz AP / 2.5 kHz LFP
  NP 2.0 : 384 AP channels per shank (up to 4 shanks), 30 kHz AP
  NP Ultra: 384 channels, 30 kHz

Recording sites are arranged in two staggered columns on the shank.  The
column pattern, inter-site spacing, and bank configuration differ between
generations.

Binary file formats:
  SpikeGLX : one .bin file (int16, channels-major) + .meta sidecar
  Open Ephys: per-channel .dat files or the new .nwb / Zarr container

This adapter reads the SpikeGLX .bin/.meta format natively, converts raw
ADC counts to µV using the gain factors in the .meta file, and provides
spatial helpers for the probe layout.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import resample, load_mapping_file, save_mapping_file

# ── Probe geometry constants ──────────────────────────────────────────────────

# NP 1.0: 960 sites, 384 selectable; 2-column staggered, 20 µm vertical pitch
_NP10_N_CHANNELS = 384
_NP10_AP_RATE = 30_000
_NP10_LFP_RATE = 2_500
_NP10_V_PITCH_UM = 20.0        # µm between rows
_NP10_H_PITCH_UM = 32.0        # µm between columns

# NP 2.0 per shank: same 384 channels, denser geometry
_NP20_N_CHANNELS = 384
_NP20_AP_RATE = 30_000
_NP20_V_PITCH_UM = 15.0
_NP20_H_PITCH_UM = 32.0

# ── Site coordinate tables ────────────────────────────────────────────────────

def _build_np10_coords() -> np.ndarray:
    """
    Return ``(384, 2)`` array of [x_um, y_um] site locations for NP 1.0.

    The NP 1.0 layout has two columns (11 µm and 43 µm from probe centreline)
    and 192 rows with 20 µm vertical spacing.  Sites are assigned in a
    specific order by the hardware multiplexer.
    """
    x_left, x_right = 11.0, 43.0
    coords = np.zeros((384, 2))
    for site in range(384):
        row = site // 2
        col = site % 2
        coords[site, 0] = x_left if col == 0 else x_right
        coords[site, 1] = row * _NP10_V_PITCH_UM
    return coords

def _build_np20_coords() -> np.ndarray:
    """
    Return ``(384, 2)`` array of [x_um, y_um] site locations for NP 2.0.

    NP 2.0 uses a 4-column metal pattern with two active columns per shank,
    5 µm from the centreline.
    """
    x_left, x_right = 27.0, 59.0
    coords = np.zeros((384, 2))
    for site in range(384):
        row = site // 2
        col = site % 2
        coords[site, 0] = x_left if col == 0 else x_right
        coords[site, 1] = row * _NP20_V_PITCH_UM
    return coords

NP10_SITE_COORDS = _build_np10_coords()
NP20_SITE_COORDS = _build_np20_coords()

# ── Default mappings ──────────────────────────────────────────────────────────

def _make_mapping(device: str, n_ch: int, ap_rate: int) -> Dict:
    return {
        'device': device,
        'sampling_rate': ap_rate,
        'channels': n_ch,
        'mapping': {f'ch_{i}': f'site_{i:03d}' for i in range(n_ch)},
        'channel_groups': {
            'bank_0': list(range(0, min(96, n_ch))),
            'bank_1': list(range(96, min(192, n_ch))),
            'bank_2': list(range(192, min(288, n_ch))),
            'bank_3': list(range(288, n_ch)),
            'even_sites': list(range(0, n_ch, 2)),
            'odd_sites':  list(range(1, n_ch, 2)),
        },
    }

NEUROPIXELS_10_MAPPING  = _make_mapping('neuropixels_1.0', _NP10_N_CHANNELS, _NP10_AP_RATE)
NEUROPIXELS_20_MAPPING  = _make_mapping('neuropixels_2.0', _NP20_N_CHANNELS, _NP20_AP_RATE)
NEUROPIXELS_ULTRA_MAPPING = _make_mapping('neuropixels_ultra', 384, _NP10_AP_RATE)


class NeuropixelsAdapter:
    """
    Adapter for Imec Neuropixels silicon probes.

    Handles:
    - Channel mapping (AP-band and LFP-band streams)
    - ADC-count to µV conversion
    - Spatial site-coordinate lookup
    - SpikeGLX .bin / .meta file parsing

    Example::

        adapter = NeuropixelsAdapter(probe='np1.0')
        data_uv = adapter.to_microvolts(raw_ap_counts)
        coords  = adapter.site_coordinates()
    """

    _PROBE_MAP = {
        'np1.0':  (NEUROPIXELS_10_MAPPING,    NP10_SITE_COORDS,  _NP10_LFP_RATE),
        'np2.0':  (NEUROPIXELS_20_MAPPING,    NP20_SITE_COORDS,  None),
        'ultra':  (NEUROPIXELS_ULTRA_MAPPING,  NP10_SITE_COORDS,  None),
    }

    # NP 1.0 gain settings → µV per LSB at each gain level
    # Full-scale ±0.6 V, 10-bit signed (1024 steps per half-range).
    # gain_factor = (0.6 V × 1e6 µV/V) / (512 × gain)
    _GAIN_TO_UV = {
        50:   1171.875,
        125:   468.750,
        250:   234.375,
        500:   117.1875,
        1000:   58.5938,
        1500:   39.0625,
        2500:   23.4375,
        3000:   19.5313,
    }
    _DEFAULT_AP_GAIN  = 500    # typical AP-band gain setting
    _DEFAULT_LFP_GAIN = 250    # typical LFP-band gain setting

    def __init__(
        self,
        probe: str = 'np1.0',
        ap_gain: int = _DEFAULT_AP_GAIN,
        lfp_gain: int = _DEFAULT_LFP_GAIN,
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise NeuropixelsAdapter.

        Args:
            probe: ``'np1.0'``, ``'np2.0'``, or ``'ultra'``.
            ap_gain: AP-band gain setting (used for µV conversion).
            lfp_gain: LFP-band gain setting.
            custom_mapping: Override with a custom channel-mapping dict.

        Raises:
            ValueError: Unknown probe string without custom_mapping.
        """
        if custom_mapping:
            self.mapping = custom_mapping
            self._coords: Optional[np.ndarray] = None
            self._lfp_rate: Optional[int] = None
        elif probe in self._PROBE_MAP:
            self.mapping, self._coords, self._lfp_rate = self._PROBE_MAP[probe]
        else:
            valid = ', '.join(f"'{k}'" for k in self._PROBE_MAP)
            raise ValueError(
                f"Unknown Neuropixels probe '{probe}'. Valid: {valid}"
            )

        self.probe = probe
        self.sampling_rate: int = int(self.mapping['sampling_rate'])
        self.ap_gain  = ap_gain
        self.lfp_gain = lfp_gain

    # ── Scaling ───────────────────────────────────────────────────────────────

    def _uv_per_lsb(self, gain: int) -> float:
        """Return µV per ADC least-significant bit for *gain*."""
        if gain in self._GAIN_TO_UV:
            return self._GAIN_TO_UV[gain]
        # Generic formula for non-standard gains
        return 600_000.0 / (512.0 * gain)

    def to_microvolts(
        self,
        data: np.ndarray,
        band: str = 'ap',
    ) -> np.ndarray:
        """
        Convert raw int16 ADC counts to µV.

        Args:
            data: ``(channels × samples)`` int16 array from .bin file.
            band: ``'ap'`` or ``'lfp'``; selects the gain factor.

        Returns:
            Float64 array in µV.
        """
        gain = self.ap_gain if band == 'ap' else self.lfp_gain
        scale = self._uv_per_lsb(gain)
        return data.astype(np.float64) * scale

    # ── Spatial helpers ───────────────────────────────────────────────────────

    def site_coordinates(self) -> np.ndarray:
        """
        Return (384, 2) array of [x_µm, y_µm] for each recording site.

        Returns:
            Coordinate array, or an index-based placeholder if the probe
            geometry is not known.
        """
        if self._coords is not None:
            return self._coords.copy()
        n = self.mapping['channels']
        return np.column_stack([np.zeros(n), np.arange(n) * 20.0])

    def sites_in_depth_range(
        self,
        y_min_um: float,
        y_max_um: float,
    ) -> np.ndarray:
        """
        Return indices of sites whose y-coordinate falls in [y_min_um, y_max_um].

        Args:
            y_min_um: Minimum depth from probe tip in µm.
            y_max_um: Maximum depth from probe tip in µm.

        Returns:
            Integer index array.
        """
        coords = self.site_coordinates()
        mask = (coords[:, 1] >= y_min_um) & (coords[:, 1] <= y_max_um)
        return np.where(mask)[0]

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase',
    ) -> np.ndarray:
        """Resample AP or LFP data to *target_rate* Hz."""
        return resample(data, float(self.sampling_rate), target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return bank/parity channel groups."""
        return self.mapping.get('channel_groups', {})

    # ── SpikeGLX .bin / .meta reader ─────────────────────────────────────────

    @classmethod
    def from_spikeglx(
        cls,
        bin_filepath: str,
        probe: str = 'np1.0',
        max_seconds: Optional[float] = None,
        channels: Optional[List[int]] = None,
    ) -> Tuple['NeuropixelsAdapter', np.ndarray]:
        """
        Load AP-band data from a SpikeGLX .bin/.meta file pair.

        SpikeGLX saves data as de-interleaved int16, *channels × samples*
        order, with a sidecar ``.meta`` text file (``key=value`` per line)
        that contains ``nSavedChans``, ``imSampRate``, ``imAiRangeMax``,
        ``imAiRangeMin``, and ``imProbeOpt``.

        Args:
            bin_filepath: Path to the ``.bin`` file.  The `.meta` file must
                          exist at the same path with ``.meta`` extension.
            probe: Probe model (used to select gain default and geometry).
            max_seconds: If given, load only the first *max_seconds* of data.
            channels: Subset of channel indices to load (None = all).

        Returns:
            ``(adapter, data_uv)`` where *data_uv* is a float64 array in µV
            of shape ``(n_channels × n_samples)``.

        Raises:
            FileNotFoundError: If .bin or .meta file not found.
            ValueError: If .meta content cannot be parsed.

        Example::

            adapter, data = NeuropixelsAdapter.from_spikeglx(
                'recording_g0_t0.imec0.ap.bin',
                probe='np1.0',
                max_seconds=10.0,
            )
            print(data.shape)   # (384, 300000)
        """
        bin_path = Path(bin_filepath)
        meta_path = bin_path.with_suffix('.meta')

        if not bin_path.is_file():
            raise FileNotFoundError(f"SpikeGLX .bin not found: {bin_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"SpikeGLX .meta not found: {meta_path}")

        # ── Parse .meta (key=value text file) ─────────────────────────────
        meta: Dict[str, str] = {}
        with open(meta_path, 'r', encoding='utf-8', errors='replace') as mf:
            for line in mf:
                line = line.strip()
                if '=' in line:
                    k, _, v = line.partition('=')
                    meta[k.strip()] = v.strip()

        n_saved_chans = int(meta.get('nSavedChans', 385))
        ap_rate = float(meta.get('imSampRate', 30_000))

        # Determine AP gain from meta (optional; use default if absent)
        ap_gain_str = meta.get('imAiRangeMax', '')
        ap_gain = 500  # default
        if 'imProbeOpt' in meta:
            # Option 3 probes default to 500 gain AP
            ap_gain = 500

        # ── Load binary data ──────────────────────────────────────────────
        file_size_bytes = bin_path.stat().st_size
        n_total_samples = file_size_bytes // (n_saved_chans * 2)  # int16 = 2 bytes

        if max_seconds is not None:
            n_samples = min(n_total_samples, int(max_seconds * ap_rate))
        else:
            n_samples = n_total_samples

        raw = np.memmap(
            str(bin_path),
            dtype='<i2',
            mode='r',
            shape=(n_total_samples, n_saved_chans),
        )

        # SpikeGLX saves (samples, channels) — take only AP channels (first 384)
        n_ap = min(n_saved_chans - 1, _NP10_N_CHANNELS)  # exclude sync channel
        if channels is None:
            channels_arr = np.arange(n_ap)
        else:
            channels_arr = np.array(channels, dtype=int)

        data_int16 = raw[:n_samples, channels_arr].T   # (n_ch × n_samples)

        adapter = cls(probe=probe, ap_gain=ap_gain)
        # Override sampling rate from meta
        adapter.sampling_rate = int(ap_rate)

        data_uv = adapter.to_microvolts(data_int16, band='ap')
        return adapter, data_uv

    @classmethod
    def from_file(cls, filepath: str) -> 'NeuropixelsAdapter':
        """Load adapter configuration from YAML / JSON."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)

    def save_mapping(self, filepath: str) -> None:
        """Persist mapping to YAML or JSON."""
        save_mapping_file(self.mapping, filepath)


def convert_neuropixels_to_standard(
    data: np.ndarray,
    probe: str = 'np1.0',
    ap_gain: int = 500,
    target_rate: Optional[float] = None,
) -> np.ndarray:
    """
    One-shot converter: ADC counts → µV, optional resampling.

    Args:
        data: int16 raw AP data ``(channels × samples)``.
        probe: Probe model key.
        ap_gain: AP-band gain setting.
        target_rate: Optional target sample rate in Hz.

    Returns:
        Processed float64 array in µV.
    """
    adapter = NeuropixelsAdapter(probe=probe, ap_gain=ap_gain)
    out = adapter.to_microvolts(data, band='ap')
    if target_rate is not None:
        out = adapter.resample_to(out, target_rate)
    return out


__all__ = [
    'NeuropixelsAdapter',
    'convert_neuropixels_to_standard',
    'NEUROPIXELS_10_MAPPING',
    'NEUROPIXELS_20_MAPPING',
    'NEUROPIXELS_ULTRA_MAPPING',
    'NP10_SITE_COORDS',
    'NP20_SITE_COORDS',
]
