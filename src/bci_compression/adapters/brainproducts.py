"""
Brain Products BrainVision Adapter

Supports the Brain Products BrainVision file format and hardware:

 BrainAmp  : up to 128 EEG channels, 5000 Hz max
 actiCHamp : 32 / 64 / 128 / 160 active electrode channels, 25 kHz max
 LiveAmp   : 8 / 16 / 32 wireless channels, 1000 Hz

File format  : BrainVision Header/Data/Marker triplet
  .vhdr — ASCII header  (key=value sections + channel metadata)
  .eeg  — binary data   (multiplexed INT_16 or FLOAT_32)
  .vmrk — ASCII markers (one event per line)

ADC data is stored multiplexed (all channels for sample 0, then sample 1,
…) as signed 16-bit integers.  Each channel has a per-channel resolution
factor (µV/bit) listed in the .vhdr [Channel Infos] section.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import resample, load_mapping_file, save_mapping_file

# ── Pre-defined mappings ──────────────────────────────────────────────────────

def _make_bp_mapping(device: str, n_ch: int, fs: float, ch_names: List[str]) -> Dict:
    return {
        'device': device,
        'sampling_rate': fs,
        'channels': n_ch,
        'mapping': {f'ch_{i}': name for i, name in enumerate(ch_names)},
        'channel_groups': {
            'all': list(range(n_ch)),
            'first_half': list(range(n_ch // 2)),
            'second_half': list(range(n_ch // 2, n_ch)),
        },
    }


# Standard 10-20 electrode labels for common cap sizes
_32_CH_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',  'FC5',
    'FC1', 'FC2', 'FC6', 'T7',  'C3',  'Cz',  'C4',  'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10','P7',  'P3',
    'Pz',  'P4',  'P8',  'PO9', 'O1',  'Oz',  'O2',  'PO10',
]

_64_CH_NAMES = _32_CH_NAMES + [
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F5',  'F1',  'F2',
    'F6',  'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5',
    'C1',  'C2',  'C6',  'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
    'P5',  'P1',  'P2',  'P6',  'PO7', 'PO3', 'POz', 'PO4',
]

_128_CH_NAMES = _64_CH_NAMES + [
    'PO8',  'Iz',  'I1',  'I2',  'IO',   'VEOG', 'HEOG', 'Ref',
    'Fp1h', 'Fp2h','F9',  'F10', 'AFF1', 'AFF2', 'AFF5h','AFF6h',
    'FFT7h','FFT8h','FFC3h','FFC4h','FFC5h','FFC6h','FTT7h','FTT8h',
    'FC4h', 'FC3h', 'CFC3h','CFC4h','CFC5h','CFC6h','CCP3h','CCP4h',
    'CPP3h','CPP4h','CPP5h','CPP6h','PPO1h','PPO2h','P9',   'P10',
    'PPO9h','PPO10h','POO1','POO2','POO9h','POO10h','OI1h','OI2h',
    'I1h', 'I2h', 'AFp1','AFp2','AFF1h','AFF2h','AFF9h','AFF10h',
    'FFT9h','FFT10h','FFC1h','FFC2h','FCC3h','FCC4h','FCC5h','FCC6h',
]

BRAINAMP_32CH_MAPPING   = _make_bp_mapping('brainamp_32ch',  32, 500.0,  _32_CH_NAMES)
BRAINAMP_64CH_MAPPING   = _make_bp_mapping('brainamp_64ch',  64, 500.0,  _64_CH_NAMES)
BRAINAMP_128CH_MAPPING  = _make_bp_mapping('brainamp_128ch', 128, 500.0, _128_CH_NAMES)
ACTICHAMP_32CH_MAPPING  = _make_bp_mapping('actichamp_32ch',  32, 1000.0, _32_CH_NAMES)
ACTICHAMP_64CH_MAPPING  = _make_bp_mapping('actichamp_64ch',  64, 1000.0, _64_CH_NAMES)
ACTICHAMP_128CH_MAPPING = _make_bp_mapping('actichamp_128ch',128, 1000.0, _128_CH_NAMES)


class BrainProductsAdapter:
    """
    Adapter for Brain Products BrainVision recording systems.

    Can be initialised from:
    - A preset device name (``brainamp_32ch`` etc.)
    - A .vhdr file read via :meth:`from_vhdr_file`
    - An arbitrary custom mapping dict

    Example::

        # From .vhdr file
        adapter, data = BrainProductsAdapter.from_vhdr_file('rec.vhdr')
        data_1khz = adapter.resample_to(data, 1000)

        # From preset
        adapter = BrainProductsAdapter(device='brainamp_64ch')
    """

    _DEVICE_MAP = {
        'brainamp_32ch':    BRAINAMP_32CH_MAPPING,
        'brainamp_64ch':    BRAINAMP_64CH_MAPPING,
        'brainamp_128ch':   BRAINAMP_128CH_MAPPING,
        'actichamp_32ch':   ACTICHAMP_32CH_MAPPING,
        'actichamp_64ch':   ACTICHAMP_64CH_MAPPING,
        'actichamp_128ch':  ACTICHAMP_128CH_MAPPING,
    }

    def __init__(
        self,
        device: str = 'brainamp_64ch',
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise BrainProductsAdapter.

        Args:
            device: Preset key; one of the keys in ``_DEVICE_MAP``.
            custom_mapping: Full mapping override dict.

        Raises:
            ValueError: Unknown device without custom_mapping.
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device in self._DEVICE_MAP:
            self.mapping = self._DEVICE_MAP[device]
        else:
            valid = ', '.join(f"'{k}'" for k in self._DEVICE_MAP)
            raise ValueError(
                f"Unknown Brain Products device '{device}'. Valid: {valid}"
            )

        self.device = device
        self.sampling_rate: float = float(self.mapping['sampling_rate'])
        # Per-channel resolution in µV/bit (may be set from .vhdr)
        self._resolutions: Optional[np.ndarray] = None

    # ── Core ─────────────────────────────────────────────────────────────────

    def convert(
        self,
        data: np.ndarray,
        apply_scaling: bool = True,
    ) -> np.ndarray:
        """
        Scale raw INT_16 samples to µV using per-channel resolutions.

        When loaded from a .vhdr file the per-channel resolution values are
        stored internally; otherwise a default of 0.1 µV/bit is assumed.

        Args:
            data: Raw data ``(channels × samples)``, integer dtype.
            apply_scaling: Apply µV scaling (set False to skip).

        Returns:
            Float64 array in µV.
        """
        out = data.astype(np.float64)
        if not apply_scaling:
            return out

        n_ch = out.shape[0]
        if self._resolutions is not None and len(self._resolutions) == n_ch:
            out *= self._resolutions[:, np.newaxis]
        else:
            out *= 0.1   # default 0.1 µV / bit
        return out

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase',
    ) -> np.ndarray:
        """Resample *data* to *target_rate* Hz."""
        return resample(data, self.sampling_rate, target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return channel groups dict."""
        return self.mapping.get('channel_groups', {})

    def get_channel_names(self) -> List[str]:
        """Return channel names in index order."""
        m = self.mapping['mapping']
        return [m[f'ch_{i}'] for i in range(len(m))]

    # ── .vhdr / .eeg parser ───────────────────────────────────────────────────

    @classmethod
    def from_vhdr_file(
        cls,
        vhdr_filepath: str,
        max_seconds: Optional[float] = None,
        channels: Optional[List[int]] = None,
    ) -> Tuple['BrainProductsAdapter', np.ndarray]:
        """
        Parse a BrainVision .vhdr + .eeg file pair and return adapter + data.

        The .vhdr is an ASCII INI-style file with these relevant sections:

        - **[Common Infos]** — ``NumberOfChannels``, ``SamplingInterval`` (µs),
          ``DataFile`` (path to .eeg), ``DataFormat`` (BINARY / ASCII),
          ``DataOrientation`` (MULTIPLEXED / VECTORIZED)
        - **[Binary Infos]** — ``BinaryFormat`` (INT_16 or IEEE_FLOAT_32)
        - **[Channel Infos]** — one line per channel:
          ``Ch<n>=name,ref,resolution,unit``

        Args:
            vhdr_filepath: Path to the ``.vhdr`` header file.
            max_seconds: Load at most this many seconds of data.
            channels: Subset of channel indices to load.

        Returns:
            ``(adapter, data_uv)`` where ``data_uv`` is float64 µV,
            shape ``(n_channels × n_samples)``.

        Raises:
            FileNotFoundError: If .vhdr or .eeg is missing.
            ValueError: If header content cannot be parsed.

        Example::

            adapter, data = BrainProductsAdapter.from_vhdr_file('rec.vhdr')
            print(adapter.sampling_rate, data.shape)
        """
        vhdr_path = Path(vhdr_filepath)
        if not vhdr_path.is_file():
            raise FileNotFoundError(f".vhdr not found: {vhdr_path}")

        # ── Parse header ──────────────────────────────────────────────────
        section: str = ''
        n_channels: int = 0
        sampling_interval_us: float = 200.0   # default → 5000 Hz
        data_filename: str = ''
        binary_format: str = 'INT_16'
        data_orientation: str = 'MULTIPLEXED'
        ch_names: Dict[int, str] = {}
        ch_resolutions: Dict[int, float] = {}

        with open(vhdr_path, 'r', encoding='utf-8', errors='replace') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith(';'):
                    continue
                if line.startswith('['):
                    section = line.lstrip('[').rstrip(']').strip()
                    continue
                if '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()

                if section == 'Common Infos':
                    if key == 'NumberOfChannels':
                        n_channels = int(val)
                    elif key == 'SamplingInterval':
                        sampling_interval_us = float(val)
                    elif key == 'DataFile':
                        data_filename = val
                    elif key == 'DataOrientation':
                        data_orientation = val.upper()

                elif section == 'Binary Infos':
                    if key == 'BinaryFormat':
                        binary_format = val.upper()

                elif section == 'Channel Infos':
                    # Key is 'Ch1', 'Ch2', … (1-based)
                    ch_idx = int(key[2:]) - 1
                    parts = val.split(',')
                    ch_names[ch_idx] = parts[0].strip() if parts else key
                    if len(parts) >= 3:
                        try:
                            ch_resolutions[ch_idx] = float(parts[2].strip())
                        except (ValueError, IndexError):
                            ch_resolutions[ch_idx] = 0.1

        if n_channels == 0:
            raise ValueError(f"No channel count found in {vhdr_path}")

        sampling_rate = 1_000_000.0 / sampling_interval_us

        # ── Locate .eeg binary ────────────────────────────────────────────
        if data_filename:
            eeg_path = vhdr_path.parent / data_filename
        else:
            eeg_path = vhdr_path.with_suffix('.eeg')

        if not eeg_path.is_file():
            raise FileNotFoundError(f".eeg not found: {eeg_path}")

        # ── Read binary data ──────────────────────────────────────────────
        if binary_format == 'IEEE_FLOAT_32' or 'FLOAT' in binary_format:
            dtype = np.dtype('<f4')
        else:
            dtype = np.dtype('<i2')   # INT_16

        raw_flat = np.fromfile(str(eeg_path), dtype=dtype)
        n_total_samples = len(raw_flat) // n_channels

        if max_seconds is not None:
            n_load = min(n_total_samples, int(max_seconds * sampling_rate))
        else:
            n_load = n_total_samples

        raw_flat = raw_flat[:n_load * n_channels]

        if data_orientation == 'VECTORIZED':
            # Vectorized: all samples of ch0, then ch1, …
            raw = raw_flat.reshape(n_channels, n_load)
        else:
            # Multiplexed (default): ch0_t0, ch1_t0, …, ch0_t1, ch1_t1, …
            raw = raw_flat.reshape(n_load, n_channels).T   # (n_ch × n_samp)

        if channels is not None:
            raw = raw[channels]
            actual_n_ch = len(channels)
            ch_names = {i: ch_names.get(channels[i], f'ch_{i}')
                        for i in range(actual_n_ch)}
            ch_resolutions = {i: ch_resolutions.get(channels[i], 0.1)
                              for i in range(actual_n_ch)}
        else:
            actual_n_ch = n_channels

        # ── Build adapter ─────────────────────────────────────────────────
        mapping_dict = {f'ch_{i}': ch_names.get(i, f'ch_{i}')
                        for i in range(actual_n_ch)}
        custom_mapping = {
            'device': 'brainproducts_vhdr',
            'sampling_rate': sampling_rate,
            'channels': actual_n_ch,
            'mapping': mapping_dict,
            'channel_groups': {
                'all': list(range(actual_n_ch)),
            },
            'vhdr_file': str(vhdr_path),
        }

        adapter = cls(custom_mapping=custom_mapping)
        resolutions_arr = np.array(
            [ch_resolutions.get(i, 0.1) for i in range(actual_n_ch)]
        )
        adapter._resolutions = resolutions_arr

        # Scale to µV
        data_uv = adapter.convert(raw, apply_scaling=True)
        return adapter, data_uv

    @classmethod
    def from_file(cls, filepath: str) -> 'BrainProductsAdapter':
        """Load adapter configuration from YAML / JSON."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)

    def save_mapping(self, filepath: str) -> None:
        """Persist mapping to YAML or JSON."""
        save_mapping_file(self.mapping, filepath)


def load_brainvision_data(
    vhdr_filepath: str,
    target_rate: Optional[float] = None,
) -> Tuple['BrainProductsAdapter', np.ndarray]:
    """
    Convenience wrapper: load BrainVision data and optionally resample.

    Args:
        vhdr_filepath: Path to the .vhdr file.
        target_rate: If given, resample to this Hz.

    Returns:
        ``(adapter, data_uv)`` where data_uv is float64 µV.
    """
    adapter, data = BrainProductsAdapter.from_vhdr_file(vhdr_filepath)
    if target_rate is not None:
        data = adapter.resample_to(data, target_rate)
    return adapter, data


__all__ = [
    'BrainProductsAdapter',
    'load_brainvision_data',
    'BRAINAMP_32CH_MAPPING',
    'BRAINAMP_64CH_MAPPING',
    'BRAINAMP_128CH_MAPPING',
    'ACTICHAMP_32CH_MAPPING',
    'ACTICHAMP_64CH_MAPPING',
    'ACTICHAMP_128CH_MAPPING',
]
