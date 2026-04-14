"""
Blackrock Device Adapter

Provides converters and mappings for Blackrock Microsystems devices:
- Neuroport arrays (96-128 channels)
- Cerebus systems (up to 256 channels)
- Utah array configurations
"""

from typing import Dict, Optional, List
import numpy as np
from . import map_channels, resample, load_mapping_file, save_mapping_file


# Blackrock Neuroport 96-channel Utah array mapping
BLACKROCK_NEUROPORT_96CH_MAPPING = {
    'device': 'blackrock_neuroport_96ch',
    'sampling_rate': 30000,
    'channels': 96,
    'array_layout': 'utah_96',
    'mapping': {
        **{f'ch_{i}': f'electrode_{i+1:03d}' for i in range(96)}
    },
    'channel_groups': {
        'grid_row_0': list(range(0, 10)),
        'grid_row_1': list(range(10, 20)),
        'grid_row_2': list(range(20, 30)),
        'grid_row_3': list(range(30, 40)),
        'grid_row_4': list(range(40, 50)),
        'grid_row_5': list(range(50, 60)),
        'grid_row_6': list(range(60, 70)),
        'grid_row_7': list(range(70, 80)),
        'grid_row_8': list(range(80, 90)),
        'grid_row_9': list(range(90, 96)),
        'motor_cortex': list(range(0, 48)),  # Example motor region
        'sensory_cortex': list(range(48, 96)),  # Example sensory region
    }
}

# Blackrock Cerebus 128-channel configuration
BLACKROCK_CEREBUS_128CH_MAPPING = {
    'device': 'blackrock_cerebus_128ch',
    'sampling_rate': 30000,
    'channels': 128,
    'array_layout': 'dual_utah',
    'mapping': {
        **{f'ch_{i}': f'array1_electrode_{i+1:03d}' for i in range(96)},
        **{f'ch_{i}': f'array2_electrode_{i-95:03d}' for i in range(96, 128)},
    },
    'channel_groups': {
        'array_1': list(range(0, 96)),
        'array_2': list(range(96, 128)),
        'array_1_motor': list(range(0, 48)),
        'array_1_sensory': list(range(48, 96)),
        'array_2_all': list(range(96, 128)),
    }
}


class BlackrockAdapter:
    """
    Adapter for Blackrock Microsystems neural recording data.

    Supports:
    - Neuroport arrays (96 channels @ 30kHz)
    - Cerebus systems (128+ channels @ 30kHz)
    - Utah array configurations
    - NEV/NSx file formats

    Example:
        >>> adapter = BlackrockAdapter(device='neuroport_96ch')
        >>> standardized = adapter.convert(raw_data)
        >>> downsampled = adapter.resample_to(standardized, target_rate=1000)
    """

    def __init__(self, device: str = 'neuroport_96ch', custom_mapping: Optional[Dict] = None):
        """
        Initialize Blackrock adapter.

        Args:
            device: Device type ('neuroport_96ch' or 'cerebus_128ch')
            custom_mapping: Optional custom mapping dictionary
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device == 'neuroport_96ch':
            self.mapping = BLACKROCK_NEUROPORT_96CH_MAPPING
        elif device == 'cerebus_128ch':
            self.mapping = BLACKROCK_CEREBUS_128CH_MAPPING
        else:
            raise ValueError(f"Unknown device type: {device}")

        self.device = device
        self.sampling_rate = self.mapping['sampling_rate']
        self.array_layout = self.mapping.get('array_layout', 'unknown')

    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Convert Blackrock raw data to standardized format.

        Args:
            data: Raw Blackrock data (channels x samples)
            apply_mapping: Whether to apply electrode mapping

        Returns:
            Standardized neural data
        """
        if not apply_mapping:
            return data

        return map_channels(data, self.mapping['mapping'])

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase'
    ) -> np.ndarray:
        """
        Resample data to target sampling rate.

        Blackrock systems typically record at 30kHz, often need downsampling.

        Args:
            data: Input data
            target_rate: Target sampling rate in Hz
            method: Resampling method ('fft' or 'polyphase')

        Returns:
            Resampled data
        """
        return resample(data, self.sampling_rate, target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Get channel groupings for this device."""
        return self.mapping.get('channel_groups', {})

    def get_array_layout(self) -> str:
        """Get the array layout type."""
        return self.array_layout

    def save_mapping(self, filepath: str) -> None:
        """Save current mapping to file."""
        save_mapping_file(self.mapping, filepath)

    @classmethod
    def from_file(cls, filepath: str) -> 'BlackrockAdapter':
        """Load adapter from mapping file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)

    @classmethod
    def from_nev_file(cls, nev_filepath: str) -> 'BlackrockAdapter':
        """
        Create adapter from NEV file metadata using pure-Python struct parsing.

        Parses the Blackrock NEV (Neural Event) binary format v2.1–v3.0:
        - Basic header (336 bytes): magic, version, timestamp resolution,
          sample resolution, creator comment
        - Extended headers (32 bytes each): NEUEVWAV packets yield channel
          IDs and electrode labels

        Args:
            nev_filepath: Path to NEV file (.nev)

        Returns:
            BlackrockAdapter configured from file metadata

        Raises:
            ValueError: If the file is not a valid Blackrock NEV file
            FileNotFoundError: If nev_filepath does not exist

        Example:
            >>> adapter = BlackrockAdapter.from_nev_file('recording.nev')
            >>> print(adapter.sampling_rate, adapter.mapping['channels'])
        """
        import struct
        import os

        if not os.path.isfile(nev_filepath):
            raise FileNotFoundError(f"NEV file not found: {nev_filepath}")

        with open(nev_filepath, 'rb') as fid:
            # ── Basic Header (336 bytes total) ──────────────────────────────
            # Bytes 0-7: File type identifier
            magic = fid.read(8)
            if magic != b'NEURALEV':
                raise ValueError(
                    f"Not a Blackrock NEV file (expected b'NEURALEV', "
                    f"got {magic!r})"
                )

            # Bytes 8-9: File spec [major, minor] (1 byte each)
            major, minor = struct.unpack('<BB', fid.read(2))

            # Bytes 10-11: Additional flags (uint16)
            fid.seek(2, 1)

            # Bytes 12-15: Total bytes in all headers combined (uint32)
            total_header_bytes, = struct.unpack('<I', fid.read(4))

            # Bytes 16-19: Bytes per data packet (uint32)
            fid.seek(4, 1)

            # Bytes 20-23: Time resolution of timestamps in Hz (uint32)
            timestamp_resolution, = struct.unpack('<I', fid.read(4))

            # Bytes 24-27: Time resolution of samples (spike waveform) in Hz (uint32)
            sample_resolution, = struct.unpack('<I', fid.read(4))

            # Bytes 28-43: Time origin (SYSTEMTIME struct, 16 bytes) — skip
            fid.seek(16, 1)

            # Bytes 44-75: Application name, null-padded 32 bytes — skip
            fid.seek(32, 1)

            # Bytes 76-331: Comment string, null-padded 256 bytes
            comment_raw = fid.read(256)
            comment = comment_raw.decode('ascii', errors='replace').rstrip('\x00').strip()

            # Bytes 332-335: Processor address (uint32) — skip
            fid.seek(4, 1)

            # ── Extended Headers (32 bytes each) ────────────────────────────
            ext_header_bytes = total_header_bytes - 336
            n_ext_headers = max(0, ext_header_bytes // 32)

            channels: Dict[str, str] = {}

            for _ in range(n_ext_headers):
                # 8-byte packet identifier string
                pkt_id_raw = fid.read(8)
                pkt_id = pkt_id_raw.decode('ascii', errors='replace').rstrip('\x00')
                # 24-byte payload
                payload = fid.read(24)

                if pkt_id == 'NEUEVWAV' and len(payload) >= 14:
                    electrode_id, = struct.unpack('<H', payload[0:2])
                    if electrode_id > 0:
                        ch_idx = electrode_id - 1
                        channels[f'ch_{ch_idx}'] = f'electrode_{electrode_id:03d}'

                elif pkt_id == 'NEUEVLBL' and len(payload) >= 18:
                    # Electrode label packet: electrode ID + 16-char label
                    electrode_id, = struct.unpack('<H', payload[0:2])
                    label = payload[2:18].decode('ascii', errors='replace').rstrip('\x00')
                    if electrode_id > 0 and label:
                        ch_key = f'ch_{electrode_id - 1}'
                        # Labels override the default electrode_XXX name
                        channels[ch_key] = label

        # Fall back to default 96-channel Neuroport layout if no NEUEVWAV found
        if not channels:
            channels = {f'ch_{i}': f'electrode_{i + 1:03d}' for i in range(96)}

        n_channels = len(channels)
        # Build contiguous integer-keyed groups
        all_indices = list(range(n_channels))

        custom_mapping = {
            'device': f'blackrock_nev_v{major}.{minor}',
            'sampling_rate': int(sample_resolution) if sample_resolution else 30000,
            'channels': n_channels,
            'array_layout': 'nev_file',
            'file_version': f'{major}.{minor}',
            'timestamp_resolution': int(timestamp_resolution),
            'comment': comment,
            'mapping': channels,
            'channel_groups': {
                'all': all_indices,
                'first_half': all_indices[: n_channels // 2],
                'second_half': all_indices[n_channels // 2:],
            },
        }

        return cls(custom_mapping=custom_mapping)


def convert_blackrock_to_standard(
    data: np.ndarray,
    device: str = 'neuroport_96ch',
    target_rate: Optional[float] = None
) -> np.ndarray:
    """
    Quick converter for Blackrock data to standardized format.

    Args:
        data: Raw Blackrock data (channels x samples)
        device: Device type ('neuroport_96ch' or 'cerebus_128ch')
        target_rate: Optional target sampling rate for resampling

    Returns:
        Standardized (and optionally resampled) neural data

    Example:
        >>> # Downsample from 30kHz to 1kHz
        >>> standard_data = convert_blackrock_to_standard(
        ...     raw_data, 'neuroport_96ch', target_rate=1000
        ... )
    """
    adapter = BlackrockAdapter(device=device)
    converted = adapter.convert(data)

    if target_rate:
        converted = adapter.resample_to(converted, target_rate)

    return converted


# Export public API
__all__ = [
    'BlackrockAdapter',
    'convert_blackrock_to_standard',
    'BLACKROCK_NEUROPORT_96CH_MAPPING',
    'BLACKROCK_CEREBUS_128CH_MAPPING',
]
