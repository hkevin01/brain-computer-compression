"""
Emotiv Device Adapter

Supports all current Emotiv headsets:
- EPOC    : 14-channel EEG @ 128 Hz   (original & EPOC X)
- EPOC+   : 14-channel EEG @ 256 Hz   (higher sample rate variant)
- INSIGHT : 5-channel EEG  @ 128 Hz
- FLEX    : up to 32-channel EEG @ 256 Hz (configurable montage)

All Emotiv EEG channels are placed according to the extended 10-20
international system.  Raw ADC values can optionally be converted to
microvolts using the published scale factor (0.51 µV/LSB for EPOC/EPOC+).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import map_channels, resample, load_mapping_file, save_mapping_file

# ── Channel definitions ───────────────────────────────────────────────────────

# EPOC / EPOC X — 14 EEG + 2 reference (CMS/DRL)
EMOTIV_EPOC_14CH_MAPPING: Dict = {
    'device': 'emotiv_epoc_14ch',
    'sampling_rate': 128,
    'channels': 14,
    'adc_scale_uv': 0.51,        # µV per raw LSB
    'reference': 'CMS/DRL',
    'mapping': {
        'ch_0':  'AF3',
        'ch_1':  'F7',
        'ch_2':  'F3',
        'ch_3':  'FC5',
        'ch_4':  'T7',
        'ch_5':  'P7',
        'ch_6':  'O1',
        'ch_7':  'O2',
        'ch_8':  'P8',
        'ch_9':  'T8',
        'ch_10': 'FC6',
        'ch_11': 'F4',
        'ch_12': 'F8',
        'ch_13': 'AF4',
    },
    'channel_groups': {
        'frontal':   [0, 2, 11, 12, 13],   # AF3, F3, F4, F8, AF4
        'temporal':  [1, 4, 9],             # F7, T7, T8
        'parietal':  [5, 8],                # P7, P8
        'occipital': [6, 7],                # O1, O2
        'motor':     [3, 10],               # FC5, FC6
    },
}

# EPOC+ — same electrode layout but 256 Hz
EMOTIV_EPOC_PLUS_MAPPING: Dict = {
    **EMOTIV_EPOC_14CH_MAPPING,
    'device': 'emotiv_epoc_plus_14ch',
    'sampling_rate': 256,
}

# INSIGHT — 5 EEG channels
EMOTIV_INSIGHT_5CH_MAPPING: Dict = {
    'device': 'emotiv_insight_5ch',
    'sampling_rate': 128,
    'channels': 5,
    'adc_scale_uv': 0.51,
    'reference': 'CMS/DRL',
    'mapping': {
        'ch_0': 'AF3',
        'ch_1': 'AF4',
        'ch_2': 'T7',
        'ch_3': 'T8',
        'ch_4': 'Pz',
    },
    'channel_groups': {
        'frontal':  [0, 1],
        'temporal': [2, 3],
        'parietal': [4],
    },
}

# FLEX — 32-channel configurable; default uses full 10-20 layout
_FLEX_CH_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',  'FC5',
    'FC1', 'FC2', 'FC6', 'M1',  'T7',  'C3',  'Cz',  'C4',
    'T8',  'M2',  'CP5', 'CP1', 'CP2', 'CP6', 'P7',  'P3',
    'Pz',  'P4',  'P8',  'POz', 'O1',  'Oz',  'O2',  'AF4',
]

EMOTIV_FLEX_32CH_MAPPING: Dict = {
    'device': 'emotiv_flex_32ch',
    'sampling_rate': 256,
    'channels': 32,
    'adc_scale_uv': 0.51,
    'reference': 'CMS/DRL',
    'mapping': {f'ch_{i}': name for i, name in enumerate(_FLEX_CH_NAMES)},
    'channel_groups': {
        'frontal':   [0, 1, 2, 3, 4, 5, 6],
        'central':   [7, 8, 9, 10, 12, 13, 14, 15, 16],
        'parietal':  [18, 19, 20, 21, 22, 23, 24, 25, 26],
        'occipital': [27, 28, 29, 30, 31],
    },
}


class EmotivAdapter:
    """
    Adapter for Emotiv EEG headsets.

    Supports EPOC, EPOC+, INSIGHT, and FLEX devices.  Handles electrode
    mapping, optional ADC-to-µV scaling, and resampling.

    Example::

        adapter = EmotivAdapter(device='epoc_14ch')
        scaled = adapter.to_microvolts(raw_adc_data)
        resampled = adapter.resample_to(scaled, target_rate=256)
    """

    _DEVICE_MAP = {
        'epoc_14ch':   EMOTIV_EPOC_14CH_MAPPING,
        'epoc_plus':   EMOTIV_EPOC_PLUS_MAPPING,
        'insight_5ch': EMOTIV_INSIGHT_5CH_MAPPING,
        'flex_32ch':   EMOTIV_FLEX_32CH_MAPPING,
    }

    def __init__(
        self,
        device: str = 'epoc_14ch',
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise EmotivAdapter.

        Args:
            device: One of ``'epoc_14ch'``, ``'epoc_plus'``,
                    ``'insight_5ch'``, ``'flex_32ch'``.
            custom_mapping: Override all settings with a custom dict.

        Raises:
            ValueError: If *device* is not a recognised key and no
                        *custom_mapping* is supplied.
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device in self._DEVICE_MAP:
            self.mapping = self._DEVICE_MAP[device]
        else:
            valid = ', '.join(f"'{k}'" for k in self._DEVICE_MAP)
            raise ValueError(
                f"Unknown Emotiv device '{device}'. Valid choices: {valid}"
            )

        self.device = device
        self.sampling_rate: float = self.mapping['sampling_rate']
        self.adc_scale_uv: float = self.mapping.get('adc_scale_uv', 0.51)

    # ── Core conversion ───────────────────────────────────────────────────────

    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Reorder channels to match the standard electrode layout.

        Args:
            data: Raw data array ``(channels × samples)``.
            apply_mapping: When *True*, apply the electrode-index mapping.

        Returns:
            Remapped ``(channels × samples)`` array.
        """
        if not apply_mapping:
            return data
        return map_channels(data, self.mapping['mapping'])

    def to_microvolts(self, data: np.ndarray) -> np.ndarray:
        """
        Convert raw ADC integers to microvolts.

        Emotiv devices output 14-bit (EPOC) or 16-bit (EPOC+/FLEX) signed
        integers.  The published scale factor is 0.51 µV per LSB.

        Args:
            data: Raw ADC data ``(channels × samples)``, integer dtype.

        Returns:
            Floating-point data in µV.
        """
        return data.astype(np.float64) * self.adc_scale_uv

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase',
    ) -> np.ndarray:
        """Resample *data* to *target_rate* Hz."""
        return resample(data, self.sampling_rate, target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return the channel-group dictionary for this device."""
        return self.mapping.get('channel_groups', {})

    def get_channel_names(self) -> List[str]:
        """Return channel names in channel-index order."""
        m = self.mapping['mapping']
        return [m[f'ch_{i}'] for i in range(len(m))]

    def save_mapping(self, filepath: str) -> None:
        """Serialise the current mapping to a YAML or JSON file."""
        save_mapping_file(self.mapping, filepath)

    @classmethod
    def from_file(cls, filepath: str) -> 'EmotivAdapter':
        """Load adapter configuration from a YAML / JSON mapping file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)

    @classmethod
    def from_flex_labels(
        cls,
        channel_labels: List[str],
        sampling_rate: float = 256.0,
    ) -> 'EmotivAdapter':
        """
        Build a FLEX adapter for a custom channel selection.

        Args:
            channel_labels: Ordered list of 10-20 electrode names.
            sampling_rate: Acquisition rate in Hz (default 256).

        Returns:
            EmotivAdapter configured for the supplied montage.

        Example::

            adapter = EmotivAdapter.from_flex_labels(
                ['Fp1', 'Fp2', 'C3', 'C4', 'Pz'], sampling_rate=256
            )
        """
        n = len(channel_labels)
        custom_mapping = {
            'device': 'emotiv_flex_custom',
            'sampling_rate': sampling_rate,
            'channels': n,
            'adc_scale_uv': 0.51,
            'mapping': {f'ch_{i}': lbl for i, lbl in enumerate(channel_labels)},
            'channel_groups': {'all': list(range(n))},
        }
        return cls(custom_mapping=custom_mapping)


def convert_emotiv_to_standard(
    data: np.ndarray,
    device: str = 'epoc_14ch',
    target_rate: Optional[float] = None,
    scale_to_uv: bool = True,
) -> np.ndarray:
    """
    One-shot converter: map channels, optionally scale to µV, optionally resample.

    Args:
        data: Raw data ``(channels × samples)``.
        device: Emotiv device key (see :class:`EmotivAdapter`).
        target_rate: If given, resample to this rate in Hz.
        scale_to_uv: Apply ADC-to-µV scaling (default ``True``).

    Returns:
        Processed ``(channels × samples)`` array.

    Example::

        data_uv = convert_emotiv_to_standard(raw, 'epoc_14ch', target_rate=256)
    """
    adapter = EmotivAdapter(device=device)
    out = adapter.convert(data)
    if scale_to_uv:
        out = adapter.to_microvolts(out)
    if target_rate is not None:
        out = adapter.resample_to(out, target_rate)
    return out


__all__ = [
    'EmotivAdapter',
    'convert_emotiv_to_standard',
    'EMOTIV_EPOC_14CH_MAPPING',
    'EMOTIV_EPOC_PLUS_MAPPING',
    'EMOTIV_INSIGHT_5CH_MAPPING',
    'EMOTIV_FLEX_32CH_MAPPING',
]
