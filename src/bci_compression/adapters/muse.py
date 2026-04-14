"""
InteraXon Muse Device Adapter

Supports the Muse headband family:
- Muse 1   (2014–2016): 4 EEG channels, 220 Hz
- Muse 2   (2018)     : 4 EEG + PPG + accl/gyro, 256 Hz
- Muse S   (2020)     : same channels as Muse 2 but sleep-optimised, 256 Hz
- Muse 2016 (2016-18) : upgraded Muse 1, 220 Hz

EEG electrodes: TP9 (left ear), AF7 (left forehead), AF8 (right forehead),
                TP10 (right ear), plus internal references Fpz.

Data streams (LSL or BlueMUSE CSV):
  - EEG    : µV, 4 channels
  - PPG    : arbitrary units, 3 channels (Muse 2/S only)
  - Gyro   : degrees/s, 3 axes
  - Accel  : g, 3 axes
  - Alpha / Beta / Delta / Gamma / Theta absolute band powers
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import map_channels, resample, load_mapping_file, save_mapping_file

# ── Channel mappings ──────────────────────────────────────────────────────────

MUSE_1_MAPPING: Dict = {
    'device': 'muse_1',
    'sampling_rate': 220,
    'channels': 4,
    'eeg_scale_uv': 1.0,   # Muse SDK delivers data already in µV
    'reference': 'Fpz',
    'mapping': {
        'ch_0': 'TP9',
        'ch_1': 'AF7',
        'ch_2': 'AF8',
        'ch_3': 'TP10',
    },
    'channel_groups': {
        'temporal':  [0, 3],   # TP9, TP10
        'frontal':   [1, 2],   # AF7, AF8
        'left':      [0, 1],
        'right':     [2, 3],
    },
}

MUSE_2_MAPPING: Dict = {
    **MUSE_1_MAPPING,
    'device': 'muse_2',
    'sampling_rate': 256,
}

MUSE_S_MAPPING: Dict = {
    **MUSE_2_MAPPING,
    'device': 'muse_s',
}

MUSE_2016_MAPPING: Dict = {
    **MUSE_1_MAPPING,
    'device': 'muse_2016',
    'sampling_rate': 220,
}


class MuseAdapter:
    """
    Adapter for InteraXon Muse EEG headbands.

    Handles electrode mapping, optional band-power extraction,
    and resampling.  Can also parse Muse CSV files produced by
    BlueMUSE / Mind Monitor.

    Example::

        adapter = MuseAdapter(device='muse_2')
        data_filtered = adapter.convert(raw_eeg)
        data_1khz = adapter.resample_to(data_filtered, 1000)
    """

    _DEVICE_MAP = {
        'muse_1':    MUSE_1_MAPPING,
        'muse_2':    MUSE_2_MAPPING,
        'muse_s':    MUSE_S_MAPPING,
        'muse_2016': MUSE_2016_MAPPING,
    }

    # Column headers emitted by Mind Monitor CSV exports
    _MIND_MONITOR_EEG_COLS = [
        '/muse/eeg',       # combined label (older)
        'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10',  # Mind Monitor
    ]
    _MIND_MONITOR_EEG_INDIVIDUAL = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    def __init__(
        self,
        device: str = 'muse_2',
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise MuseAdapter.

        Args:
            device: ``'muse_1'``, ``'muse_2'``, ``'muse_s'``, or ``'muse_2016'``.
            custom_mapping: Override with a custom channel-mapping dict.

        Raises:
            ValueError: For unknown device strings without a custom_mapping.
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device in self._DEVICE_MAP:
            self.mapping = self._DEVICE_MAP[device]
        else:
            valid = ', '.join(f"'{k}'" for k in self._DEVICE_MAP)
            raise ValueError(
                f"Unknown Muse device '{device}'. Valid choices: {valid}"
            )

        self.device = device
        self.sampling_rate: float = self.mapping['sampling_rate']

    # ── Core conversion ───────────────────────────────────────────────────────

    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Apply channel mapping to raw EEG data.

        Args:
            data: ``(channels × samples)`` array in µV.
            apply_mapping: Reorder channels according to the device map.

        Returns:
            Mapped ``(channels × samples)`` array.
        """
        if not apply_mapping:
            return data
        return map_channels(data, self.mapping['mapping'])

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase',
    ) -> np.ndarray:
        """Resample *data* from the device rate to *target_rate* Hz."""
        return resample(data, self.sampling_rate, target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return named channel groups."""
        return self.mapping.get('channel_groups', {})

    def get_channel_names(self) -> List[str]:
        """Channel names in index order: TP9, AF7, AF8, TP10."""
        m = self.mapping['mapping']
        return [m[f'ch_{i}'] for i in range(len(m))]

    # ── Band-power utilities ──────────────────────────────────────────────────

    def compute_band_power(
        self,
        data: np.ndarray,
        band: Tuple[float, float],
    ) -> np.ndarray:
        """
        Compute mean absolute band power for each channel via Welch's method.

        Args:
            data: ``(channels × samples)`` EEG in µV.
            band: ``(low_hz, high_hz)`` frequency band.

        Returns:
            1-D array of shape ``(channels,)`` with power in µV².
        """
        from scipy.signal import welch

        n_ch = data.shape[0]
        powers = np.zeros(n_ch)
        for i in range(n_ch):
            freqs, psd = welch(
                data[i],
                fs=self.sampling_rate,
                nperseg=min(256, data.shape[1]),
            )
            mask = (freqs >= band[0]) & (freqs <= band[1])
            if mask.any():
                # Integrate PSD over the band (trapezoidal rule)
                powers[i] = np.trapz(psd[mask], freqs[mask])
        return powers

    # ── CSV file reader (Mind Monitor / BlueMUSE) ─────────────────────────────

    @classmethod
    def from_csv(
        cls,
        csv_filepath: str,
        device: str = 'muse_2',
    ) -> Tuple['MuseAdapter', np.ndarray, np.ndarray]:
        """
        Load EEG data from a Mind Monitor CSV export.

        Mind Monitor writes one row per sample with a ``TimeStamp`` column
        followed by per-electrode columns ``RAW_TP9``, ``RAW_AF7``,
        ``RAW_AF8``, ``RAW_TP10`` (all in µV).

        Args:
            csv_filepath: Path to the ``.csv`` file.
            device: Muse model; determines the nominal sampling rate.

        Returns:
            ``(adapter, timestamps, eeg_data)`` where *timestamps* is a 1-D
            array of UNIX seconds, and *eeg_data* is ``(4 × samples)`` µV.

        Example::

            adapter, t, eeg = MuseAdapter.from_csv('session.csv', 'muse_2')
            print(eeg.shape)  # (4, N)
        """
        adapter = cls(device=device)

        timestamps: List[float] = []
        rows: List[List[float]] = []

        with open(csv_filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty or header-less CSV: {csv_filepath}")

            # Locate EEG columns – Mind Monitor uses RAW_TP9 … RAW_TP10
            eeg_cols = [
                c for c in reader.fieldnames
                if c in cls._MIND_MONITOR_EEG_INDIVIDUAL
            ]
            ts_col = next(
                (c for c in reader.fieldnames
                 if 'time' in c.lower() or 'timestamp' in c.lower()),
                None,
            )

            if len(eeg_cols) < 4:
                raise ValueError(
                    f"Could not find the four EEG columns "
                    f"(RAW_TP9/AF7/AF8/TP10) in {csv_filepath}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                try:
                    if ts_col:
                        timestamps.append(float(row[ts_col]))
                    rows.append([float(row[c]) for c in eeg_cols])
                except (ValueError, KeyError):
                    continue  # skip malformed rows

        if not rows:
            raise ValueError(f"No valid EEG rows found in {csv_filepath}")

        eeg_data = np.array(rows, dtype=np.float64).T   # (4 × N)
        ts_array = np.array(timestamps) if timestamps else np.arange(eeg_data.shape[1]) / adapter.sampling_rate

        return adapter, ts_array, eeg_data

    def save_mapping(self, filepath: str) -> None:
        """Persist the channel mapping to YAML or JSON."""
        save_mapping_file(self.mapping, filepath)

    @classmethod
    def from_file(cls, filepath: str) -> 'MuseAdapter':
        """Load adapter configuration from a YAML / JSON file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)


def convert_muse_to_standard(
    data: np.ndarray,
    device: str = 'muse_2',
    target_rate: Optional[float] = None,
) -> np.ndarray:
    """
    One-shot converter for Muse EEG data.

    Args:
        data: ``(channels × samples)`` in µV.
        device: Muse device key.
        target_rate: Optional resampling target in Hz.

    Returns:
        Converted (and optionally resampled) data.
    """
    adapter = MuseAdapter(device=device)
    out = adapter.convert(data)
    if target_rate is not None:
        out = adapter.resample_to(out, target_rate)
    return out


__all__ = [
    'MuseAdapter',
    'convert_muse_to_standard',
    'MUSE_1_MAPPING',
    'MUSE_2_MAPPING',
    'MUSE_S_MAPPING',
    'MUSE_2016_MAPPING',
]
