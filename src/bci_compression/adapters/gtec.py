"""
g.tec Medical Engineering Device Adapter

Supported hardware:
  g.USBamp   : 16 / 32 channels, up to 38400 Hz, USB amplifier
  g.HIamp    : 32 / 64 / 128 / 256 channels, up to 38400 Hz, high-density
  g.Nautilus : 8 / 16 / 32 wireless dry-electrode EEG, 250 / 500 Hz
  g.SAHARA   : CMOS wireless dry electrodes (16 ch @ 256 Hz)

g.tec systems are widely used for P300, SSVEP, and motor imagery BCIs.
Channel labels follow the 10-20 international system.

g.tec data can be exported in several formats.  This adapter works with:
  - Raw NumPy arrays (any g.tec system)
  - GDF files (exported via g.BSanalyze / g.tec MATLAB toolbox)
    — GDF is a binary format with a 256-byte fixed header + per-channel
      header blocks followed by raw float / int ADC data.

For live streaming from g.tec hardware, use the official g.NautilusAPI
Python interface or the g.tec MATLAB / Simulink driver.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import resample, load_mapping_file, save_mapping_file

# ── Channel-name tables ───────────────────────────────────────────────────────

_16_CH = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
    'C1', 'Cz',  'C2',  'C4',  'C6',  'CP3', 'CPz','CP4',
]

_32_CH = _16_CH + [
    'CP5', 'CP1', 'CP2', 'CP6', 'P7',  'P5',  'P3',  'P1',
    'Pz',  'P2',  'P4',  'P6',  'P8',  'PO7', 'PO3', 'POz',
]

_64_CH = _32_CH + [
    'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz',  'Fp1', 'Fpz',
    'Fp2', 'AF7', 'AF3','AFz','AF4','AF8',  'F7',  'F5',
    'F3',  'F1',  'F2', 'F4', 'F6', 'F8',  'FT7', 'FT8',
    'T7',  'T8',  'TP7','TP8','P9', 'P10', 'TP9', 'TP10',
]

_128_CH = _64_CH + [
    'I1',  'I2',  'IO',  'FC5', 'FC6', 'FT9', 'FT10','T9',
    'T10', 'TP9h','TP10h','P9h','P10h','PO9h','PO10h','M1',
    'M2',  'Nz',  'AFF1','AFF2','AFF5h','AFF6h','FFT7h','FFT8h',
    'FFC3h','FFC4h','FFC5h','FFC6h','FCC3h','FCC4h','FCC5h','FCC6h',
    'CCP3h','CCP4h','CCP5h','CCP6h','CPP3h','CPP4h','CPP5h','CPP6h',
    'PPO1h','PPO2h','PPO5h','PPO6h','POO1h','POO2h','POO5h','POO6h',
    'OI1h','OI2h','AFp1','AFp2','FFT9h','FFT10h','FFC1h','FFC2h',
    'FCC1h','FCC2h','CCP1h','CCP2h','CPP1h','CPP2h','PPO9h','PPO10h',
]

_256_CH = _128_CH + [f'E{i+1}' for i in range(128)]


def _make_mapping(device: str, n_ch: int, fs: float, names: List[str]) -> Dict:
    return {
        'device': device,
        'sampling_rate': fs,
        'channels': n_ch,
        'mapping': {f'ch_{i}': name for i, name in enumerate(names)},
        'channel_groups': {
            'all': list(range(n_ch)),
            'frontal':  [i for i, n in enumerate(names) if n.startswith(('Fp','AF','F'))],
            'central':  [i for i, n in enumerate(names) if n.startswith(('FC','C','CP'))],
            'parietal': [i for i, n in enumerate(names) if n.startswith('P')],
            'occipital':[i for i, n in enumerate(names) if n.startswith(('O','PO'))],
        },
    }


GTEC_USBAMP_16CH_MAPPING  = _make_mapping('gtec_usbamp_16ch',  16, 256.0,  _16_CH)
GTEC_USBAMP_32CH_MAPPING  = _make_mapping('gtec_usbamp_32ch',  32, 256.0,  _32_CH)
GTEC_HIAMP_32CH_MAPPING   = _make_mapping('gtec_hiamp_32ch',   32, 256.0,  _32_CH)
GTEC_HIAMP_64CH_MAPPING   = _make_mapping('gtec_hiamp_64ch',   64, 256.0,  _64_CH)
GTEC_HIAMP_128CH_MAPPING  = _make_mapping('gtec_hiamp_128ch', 128, 256.0,  _128_CH)
GTEC_HIAMP_256CH_MAPPING  = _make_mapping('gtec_hiamp_256ch', 256, 256.0,  _256_CH)
GTEC_NAUTILUS_8CH_MAPPING = _make_mapping('gtec_nautilus_8ch',  8, 250.0,
                                          ['Fp1','Fp2','C3','C4','P7','P8','O1','O2'])
GTEC_NAUTILUS_32CH_MAPPING= _make_mapping('gtec_nautilus_32ch', 32, 500.0, _32_CH)


class GTecAdapter:
    """
    Adapter for g.tec Medical Engineering EEG / BCI systems.

    Handles channel mapping, resampling, and optional ERD/ERS detection
    for motor-imagery paradigms common in g.tec research setups.

    Example::

        adapter = GTecAdapter(device='gtec_hiamp_64ch')
        data_uv = adapter.convert(raw_data)
        erd = adapter.compute_erd(data_uv, band=(8, 13), baseline_sec=2.0)
    """

    _DEVICE_MAP = {
        'usbamp_16ch':   GTEC_USBAMP_16CH_MAPPING,
        'usbamp_32ch':   GTEC_USBAMP_32CH_MAPPING,
        'hiamp_32ch':    GTEC_HIAMP_32CH_MAPPING,
        'hiamp_64ch':    GTEC_HIAMP_64CH_MAPPING,
        'hiamp_128ch':   GTEC_HIAMP_128CH_MAPPING,
        'hiamp_256ch':   GTEC_HIAMP_256CH_MAPPING,
        'nautilus_8ch':  GTEC_NAUTILUS_8CH_MAPPING,
        'nautilus_32ch': GTEC_NAUTILUS_32CH_MAPPING,
    }

    def __init__(
        self,
        device: str = 'hiamp_64ch',
        custom_mapping: Optional[Dict] = None,
    ) -> None:
        """
        Initialise GTecAdapter.

        Args:
            device: Preset key (see _DEVICE_MAP).
            custom_mapping: Override with a custom mapping dict.

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
                f"Unknown g.tec device '{device}'. Valid: {valid}"
            )
        self.device = device
        self.sampling_rate: float = float(self.mapping['sampling_rate'])

    # ── Core ─────────────────────────────────────────────────────────────────

    def convert(self, data: np.ndarray) -> np.ndarray:
        """
        Return data as float64 (g.tec drivers deliver µV directly).

        Args:
            data: ``(channels × samples)`` array.

        Returns:
            Float64 copy of *data*.
        """
        return data.astype(np.float64)

    def resample_to(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'polyphase',
    ) -> np.ndarray:
        """Resample *data* to *target_rate* Hz."""
        return resample(data, self.sampling_rate, target_rate, method=method)

    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Return named channel groups."""
        return self.mapping.get('channel_groups', {})

    def get_channel_names(self) -> List[str]:
        """Return channel names in index order."""
        m = self.mapping['mapping']
        return [m[f'ch_{i}'] for i in range(len(m))]

    # ── ERD / ERS analysis ────────────────────────────────────────────────────

    def compute_erd(
        self,
        data: np.ndarray,
        band: Tuple[float, float] = (8.0, 13.0),
        baseline_sec: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Event-Related (De)Synchronisation percentage.

        ERD(%) = [ (A – R) / R ] × 100

        where A = band power in the active window per channel, and
              R = band power in the baseline (first *baseline_sec* seconds).

        Args:
            data: ``(channels × samples)`` EEG in µV.
            band: ``(low_hz, high_hz)`` frequency range.
            baseline_sec: Duration of the baseline epoch at the start.

        Returns:
            1-D array ``(channels,)`` of ERD% values.
            Negative values → desynchronisation (ERD).
            Positive values → synchronisation (ERS).
        """
        from scipy.signal import butter, filtfilt

        n_baseline = int(baseline_sec * self.sampling_rate)
        n_ch = data.shape[0]

        # Band-pass filter
        b, a = butter(4, [band[0] / (self.sampling_rate / 2),
                          band[1] / (self.sampling_rate / 2)],
                      btype='bandpass')
        filtered = filtfilt(b, a, data, axis=1)

        # Envelope via squaring + sliding RMS
        power = filtered ** 2
        baseline_pwr = np.mean(power[:, :n_baseline], axis=1)   # (n_ch,)
        active_pwr   = np.mean(power[:, n_baseline:], axis=1)   # (n_ch,)

        with np.errstate(divide='ignore', invalid='ignore'):
            erd = np.where(
                baseline_pwr > 0,
                (active_pwr - baseline_pwr) / baseline_pwr * 100.0,
                0.0,
            )
        return erd

    # ── GDF file reader ───────────────────────────────────────────────────────

    @classmethod
    def from_gdf_file(
        cls,
        gdf_filepath: str,
        max_seconds: Optional[float] = None,
    ) -> Tuple['GTecAdapter', np.ndarray, np.ndarray]:
        """
        Parse a GDF (General Data Format) v1.25 / v2.0 file.

        GDF is a binary format used by g.tec's g.BSanalyze and BioSig:
        - Fixed 256-byte file header (version, patient info, recording info)
        - Per-channel header records (256 bytes each)
        - Raw data records (float32 or int16 multiplexed by channel)

        Args:
            gdf_filepath: Path to ``.gdf`` file.
            max_seconds: Load at most this many seconds.

        Returns:
            ``(adapter, timestamps_s, data_uv)`` where timestamps are in
            seconds from recording start and data is float64 µV
            ``(n_channels × n_samples)``.

        Raises:
            FileNotFoundError: If file not found.
            ValueError: If magic bytes indicate a non-GDF file.
        """
        gdf_path = Path(gdf_filepath)
        if not gdf_path.is_file():
            raise FileNotFoundError(f"GDF file not found: {gdf_path}")

        with open(gdf_path, 'rb') as fid:
            # ── Fixed header (256 bytes) ───────────────────────────────────
            version_raw = fid.read(8)
            version_str = version_raw.decode('ascii', errors='replace').strip()
            if not version_str.startswith('GDF'):
                raise ValueError(
                    f"Not a GDF file (header: {version_str!r}): {gdf_path}"
                )

            fid.seek(88, 1)   # skip patient info (88 bytes)
            # Recording info
            startdate_raw = fid.read(8)  # startdate as float64 (unused here)

            # Header length in 256-byte blocks (uint64, little-endian at offset 184)
            fid.seek(184)
            header_blocks, = struct.unpack('<Q', fid.read(8))

            # Reserved (44 bytes) — skip but stay at offset 192
            fid.seek(192)
            n_data_records, = struct.unpack('<q', fid.read(8))  # int64
            # Duration of one data record in seconds (2 × uint32: num/den at 200)
            fid.seek(200)
            dur_num, dur_den = struct.unpack('<II', fid.read(8))
            record_duration = dur_num / max(1, dur_den)          # seconds
            n_channels, = struct.unpack('<I', fid.read(4))

            # ── Per-channel header (n × 256 bytes starting at 256) ─────────
            fid.seek(256)
            # GDF v2: channels labels are 16-byte ASCII strings
            ch_labels_raw = fid.read(16 * n_channels)
            ch_labels = [ch_labels_raw[i*16:(i+1)*16].decode(
                'ascii', errors='replace').rstrip('\x00').strip()
                for i in range(n_channels)]

            # Transducer type (80 bytes each) — skip
            fid.seek(256 + 16 * n_channels + 80 * n_channels)
            # Physical dimension code (8 bytes each) — skip
            fid.seek(8 * n_channels, 1)
            # Physical min (float64 × n)
            phys_min = np.frombuffer(fid.read(8 * n_channels), dtype='<f8')
            # Physical max (float64 × n)
            phys_max = np.frombuffer(fid.read(8 * n_channels), dtype='<f8')
            # Digital min (int64 × n)
            dig_min = np.frombuffer(fid.read(8 * n_channels), dtype='<i8')
            # Digital max (int64 × n)
            dig_max = np.frombuffer(fid.read(8 * n_channels), dtype='<i8')
            # Skip prefilter (68 bytes each)
            fid.seek(68 * n_channels, 1)
            # Samples per record (uint32 × n)
            spr_raw = fid.read(4 * n_channels)
            samples_per_record = np.frombuffer(spr_raw, dtype='<u4')
            # Data type (uint32 × n)
            dtype_codes = np.frombuffer(fid.read(4 * n_channels), dtype='<u4')

        # ── Determine sampling rate ─────────────────────────────────────────
        # fs = samples_per_record[0] / record_duration
        fs = float(samples_per_record[0]) / record_duration if record_duration > 0 else 256.0

        # ── Scale factors ────────────────────────────────────────────────────
        dig_range = (dig_max - dig_min).astype(np.float64)
        phys_range = (phys_max - phys_min).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(dig_range != 0, phys_range / dig_range, 1.0)
        offset = phys_min - scale * dig_min.astype(np.float64)

        # ── Read data ────────────────────────────────────────────────────────
        header_bytes = int(header_blocks) * 256
        total_ch_samps = int(np.sum(samples_per_record))

        # Number of records to load
        if max_seconds is not None and record_duration > 0:
            n_records = min(
                int(n_data_records),
                max(1, int(np.ceil(max_seconds / record_duration)))
            )
        else:
            n_records = int(n_data_records)

        # GDF data types: code 3 = int16, code 16 = float32, code 17 = float64
        _DTYPE_MAP = {3: '<i2', 16: '<f4', 17: '<f8', 2: '<u2', 4: '<u4'}
        read_dtype = _DTYPE_MAP.get(int(dtype_codes[0]), '<i2')
        bytes_per_sample = np.dtype(read_dtype).itemsize

        total_samples_to_read = total_ch_samps * n_records
        raw_flat = np.fromfile(
            str(gdf_path),
            dtype=np.dtype(read_dtype),
            count=total_samples_to_read,
            offset=header_bytes,
        )

        # Reshape to (records, channels, samples_per_record)
        raw = raw_flat[:n_records * total_ch_samps].reshape(n_records, n_channels, -1)
        # Merge records along time axis → (n_channels, total_samples)
        raw = raw.transpose(1, 0, 2).reshape(n_channels, -1).astype(np.float64)

        # Apply scale/offset to each channel
        data_phys = raw * scale[:, np.newaxis] + offset[:, np.newaxis]

        # ── Build adapter ─────────────────────────────────────────────────────
        custom_mapping = {
            'device': f'gtec_gdf_v{version_str[-4:] if len(version_str) >= 4 else ""}',
            'sampling_rate': fs,
            'channels': n_channels,
            'mapping': {f'ch_{i}': ch_labels[i] or f'ch_{i}'
                        for i in range(n_channels)},
            'channel_groups': {'all': list(range(n_channels))},
        }
        adapter = cls(custom_mapping=custom_mapping)

        n_samples_total = raw.shape[1]
        timestamps = np.arange(n_samples_total) / fs
        return adapter, timestamps, data_phys

    @classmethod
    def from_file(cls, filepath: str) -> 'GTecAdapter':
        """Load adapter configuration from YAML / JSON."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)

    def save_mapping(self, filepath: str) -> None:
        """Persist mapping to YAML or JSON."""
        save_mapping_file(self.mapping, filepath)


def convert_gtec_to_standard(
    data: np.ndarray,
    device: str = 'hiamp_64ch',
    target_rate: Optional[float] = None,
) -> np.ndarray:
    """
    Convert g.tec data to standard float64 format, optionally resampling.

    Args:
        data: ``(channels × samples)`` in µV.
        device: g.tec device preset key.
        target_rate: Optional resample target Hz.

    Returns:
        Processed float64 array.
    """
    adapter = GTecAdapter(device=device)
    out = adapter.convert(data)
    if target_rate is not None:
        out = adapter.resample_to(out, target_rate)
    return out


__all__ = [
    'GTecAdapter',
    'convert_gtec_to_standard',
    'GTEC_USBAMP_16CH_MAPPING',
    'GTEC_USBAMP_32CH_MAPPING',
    'GTEC_HIAMP_32CH_MAPPING',
    'GTEC_HIAMP_64CH_MAPPING',
    'GTEC_HIAMP_128CH_MAPPING',
    'GTEC_HIAMP_256CH_MAPPING',
    'GTEC_NAUTILUS_8CH_MAPPING',
    'GTEC_NAUTILUS_32CH_MAPPING',
]
