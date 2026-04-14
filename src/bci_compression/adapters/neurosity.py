"""
Neurosity Crown / Shift Device Adapter

Supported devices:
- Crown : 8-channel EEG @ 256 Hz (research headset, 2020–present)
- Shift : 4-channel EEG @ 256 Hz (earlier prosumer headset)

Crown electrode placement (extended 10-20):
  CP3, C3, F5, PO3  (left hemisphere)
  PO4, F6, C4, CP4  (right hemisphere)

The Neurosity SDK streams data via gRPC / WebSocket.  This adapter works
with pre-recorded NumPy arrays; for live streaming use the official
``neurosity`` Python client (``pip install neurosity``).

Data from the Crown SDK is delivered in µV, 256 samples per second,
in batches depending on the callback rate.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import map_channels, resample, load_mapping_file, save_mapping_file

# ── Channel definitions ───────────────────────────────────────────────────────

NEUROSITY_CROWN_MAPPING: Dict = {
    'device': 'neurosity_crown',
    'sampling_rate': 256,
    'channels': 8,
    'reference': 'linked mastoids',
    'notch_hz': [50, 60],
    'mapping': {
        'ch_0': 'CP3',
        'ch_1': 'C3',
        'ch_2': 'F5',
        'ch_3': 'PO3',
        'ch_4': 'PO4',
        'ch_5': 'F6',
        'ch_6': 'C4',
        'ch_7': 'CP4',
    },
    'channel_groups': {
        'left_hemisphere':  [0, 1, 2, 3],   # CP3, C3, F5, PO3
        'right_hemisphere': [4, 5, 6, 7],   # PO4, F6, C4, CP4
        'motor_strip':      [1, 6],          # C3, C4
        'frontal':          [2, 5],          # F5, F6
        'parieto_occipital':[3, 4],          # PO3, PO4
        'central_parietal': [0, 7],          # CP3, CP4
    },
}

NEUROSITY_SHIFT_MAPPING: Dict = {
    'device': 'neurosity_shift',
    'sampling_rate': 256,
    'channels': 4,
    'reference': 'linked mastoids',
    'mapping': {
        'ch_0': 'C3',
        'ch_1': 'C4',
        'ch_2': 'Cz',
        'ch_3': 'Pz',
    },
    'channel_groups': {
        'motor': [0, 1],
        'midline': [2, 3],
        'left':  [0],
        'right': [1],
    },
}


class NeurosityAdapter:
    """
    Adapter for Neurosity Crown and Shift headsets.

    Encapsulates channel mapping, resampling, and common EEG band-power
    extraction used in BCI pipelines.

    Example::

        adapter = NeurosityAdapter(device='crown')
        mapped = adapter.convert(raw_eeg)          # reorder channels
        bands  = adapter.band_powers(mapped)       # {'alpha': ..., ...}
    """

    _DEVICE_MAP = {
        'crown': NEUROSITY_CROWN_MAPPING,
        'shift': NEUROSITY_SHIFT_MAPPING,
    }

    # Standard EEG frequency bands (Hz)
    BANDS: Dict[str, tuple] = {
        'delta': (0.5,  4.0),
        'theta': (4.0,  8.0),
        'alpha': (8.0, 13.0),
        'beta':  (13.0, 30.0),
        'gamma': (30.0, 100.0),
    }

    def __init__(
        self,
        device: str = 'crown',
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise NeurosityAdapter.

        Args:
            device: ``'crown'`` or ``'shift'``.
            custom_mapping: Override all settings with a custom dict.

        Raises:
            ValueError: For unknown device without custom_mapping.
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device in self._DEVICE_MAP:
            self.mapping = self._DEVICE_MAP[device]
        else:
            valid = ', '.join(f"'{k}'" for k in self._DEVICE_MAP)
            raise ValueError(
                f"Unknown Neurosity device '{device}'. Valid: {valid}"
            )

        self.device = device
        self.sampling_rate: float = self.mapping['sampling_rate']

    # ── Core ─────────────────────────────────────────────────────────────────

    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Apply channel reordering.

        Args:
            data: ``(channels × samples)`` in µV.
            apply_mapping: When True, apply the electrode map.

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
        """Resample *data* to *target_rate* Hz."""
        return resample(data, self.sampling_rate, target_rate, method=method)

    # ── Band-power extraction ─────────────────────────────────────────────────

    def band_powers(
        self,
        data: np.ndarray,
        bands: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-channel mean band power using Welch's PSD estimator.

        Args:
            data: ``(channels × samples)`` EEG in µV.
            bands: Band definition dict ``{name: (low_hz, high_hz)}``.
                   Defaults to the five standard EEG bands.

        Returns:
            Dict mapping band name to ``(channels,)`` power array in µV².

        Example::

            result = adapter.band_powers(eeg)
            alpha_power = result['alpha']   # shape (8,) for Crown
        """
        from scipy.signal import welch

        if bands is None:
            bands = self.BANDS

        n_ch, n_samp = data.shape
        nperseg = min(256, n_samp)
        result: Dict[str, np.ndarray] = {}

        for band_name, (low, high) in bands.items():
            band_power = np.zeros(n_ch)
            for i in range(n_ch):
                freqs, psd = welch(
                    data[i], fs=self.sampling_rate, nperseg=nperseg
                )
                mask = (freqs >= low) & (freqs <= high)
                if mask.any():
                    band_power[i] = float(np.trapz(psd[mask], freqs[mask]))
            result[band_name] = band_power

        return result

    def focus_index(self, data: np.ndarray) -> float:
        """
        Compute a simple focus index = beta / (alpha + theta) across all channels.

        This mirrors the metric used in the Neurosity SDK's ``focus`` status.

        Args:
            data: ``(channels × samples)`` EEG in µV.

        Returns:
            Scalar focus index (higher → more focused).
        """
        bp = self.band_powers(data, {
            'alpha': self.BANDS['alpha'],
            'beta':  self.BANDS['beta'],
            'theta': self.BANDS['theta'],
        })
        denom = np.mean(bp['alpha']) + np.mean(bp['theta'])
        if denom == 0:
            return 0.0
        return float(np.mean(bp['beta']) / denom)

    def calm_index(self, data: np.ndarray) -> float:
        """
        Compute a simple calm index = alpha / (beta + gamma).

        Args:
            data: ``(channels × samples)`` EEG in µV.

        Returns:
            Scalar calm index (higher → calmer).
        """
        bp = self.band_powers(data, {
            'alpha': self.BANDS['alpha'],
            'beta':  self.BANDS['beta'],
            'gamma': self.BANDS['gamma'],
        })
        denom = np.mean(bp['beta']) + np.mean(bp['gamma'])
        if denom == 0:
            return 0.0
        return float(np.mean(bp['alpha']) / denom)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return the channel-group dictionary."""
        return self.mapping.get('channel_groups', {})

    def get_channel_names(self) -> List[str]:
        """Return channel names in index order."""
        m = self.mapping['mapping']
        return [m[f'ch_{i}'] for i in range(len(m))]

    def save_mapping(self, filepath: str) -> None:
        """Save mapping to YAML or JSON."""
        save_mapping_file(self.mapping, filepath)

    @classmethod
    def from_file(cls, filepath: str) -> 'NeurosityAdapter':
        """Load adapter from a YAML / JSON mapping file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)


def convert_neurosity_to_standard(
    data: np.ndarray,
    device: str = 'crown',
    target_rate: Optional[float] = None,
) -> np.ndarray:
    """
    Convert Neurosity data to standardised format, optionally resampling.

    Args:
        data: ``(channels × samples)`` in µV.
        device: ``'crown'`` or ``'shift'``.
        target_rate: Optional resample target Hz.

    Returns:
        Processed data array.
    """
    adapter = NeurosityAdapter(device=device)
    out = adapter.convert(data)
    if target_rate is not None:
        out = adapter.resample_to(out, target_rate)
    return out


__all__ = [
    'NeurosityAdapter',
    'convert_neurosity_to_standard',
    'NEUROSITY_CROWN_MAPPING',
    'NEUROSITY_SHIFT_MAPPING',
]
