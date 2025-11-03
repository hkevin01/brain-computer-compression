"""
BCI Device Adapters - Multi-System Electrode Mapping & Signal Processing

This module provides utilities to make compression algorithms portable across
different BCI acquisition systems (NeuroPort, Blackrock, Intan, OpenBCI, etc.).

Key Features:
- Electrode/channel mapping between different naming conventions
- Resampling with anti-aliasing for rate normalization
- Channel grouping for spatial filtering and compression
- Calibration metadata management
"""

from typing import Dict, List, Optional, Union, Literal
import numpy as np
from scipy import signal
import yaml
import json


def map_channels(
    data: np.ndarray,
    mapping: Dict[str, Union[str, int]],
    input_format: Literal["index", "name"] = "index",
    output_format: Literal["index", "name"] = "index"
) -> np.ndarray:
    """
    Map channels from one naming/indexing convention to another.

    Args:
        data: Input neural data (channels x samples) or (samples x channels)
        mapping: Dictionary mapping source channels to target channels
                 e.g., {'ch_0': 'FP1', 'ch_1': 'FP2'} or {0: 8, 1: 9}
        input_format: Format of input channel identifiers ('index' or 'name')
        output_format: Format of output channel identifiers ('index' or 'name')

    Returns:
        Remapped neural data with channels reordered/renamed

    Example:
        >>> mapping = {'ch_0': 'FP1', 'ch_1': 'FP2', 'ch_2': 'F3'}
        >>> remapped = map_channels(data, mapping)
    """
    # Ensure data is channels x samples
    if data.shape[0] > data.shape[1]:
        data = data.T
        transposed = True
    else:
        transposed = False

    n_channels, n_samples = data.shape

    # Build index mapping
    if input_format == "index" and output_format == "index":
        # Direct index mapping
        index_map = {int(k.split('_')[1]) if isinstance(k, str) and k.startswith('ch_') else k: v
                     for k, v in mapping.items()}
    elif input_format == "name":
        # Name to index mapping (reverse lookup)
        index_map = {i: list(mapping.keys()).index(k)
                     for i, k in enumerate(mapping.keys())}
    else:
        # Default: preserve order
        index_map = {i: i for i in range(n_channels)}

    # Apply mapping
    remapped_data = np.zeros_like(data)
    for src_idx in range(min(n_channels, len(index_map))):
        if src_idx in index_map:
            tgt_idx = index_map[src_idx]
            if isinstance(tgt_idx, int) and tgt_idx < n_channels:
                remapped_data[tgt_idx] = data[src_idx]
            else:
                remapped_data[src_idx] = data[src_idx]

    return remapped_data.T if transposed else remapped_data


def resample(
    data: np.ndarray,
    src_rate: float,
    dst_rate: float,
    method: Literal["fft", "polyphase"] = "polyphase",
    axis: int = -1
) -> np.ndarray:
    """
    Resample neural data with anti-aliasing filter.

    Args:
        data: Input neural data (any shape, resampling along specified axis)
        src_rate: Source sampling rate in Hz
        dst_rate: Destination sampling rate in Hz
        method: Resampling method ('fft' or 'polyphase')
        axis: Axis along which to resample (default: last axis)

    Returns:
        Resampled neural data

    Example:
        >>> # Downsample from 30kHz to 10kHz
        >>> resampled = resample(data, 30000, 10000, method='polyphase')
    """
    if src_rate == dst_rate:
        return data

    # Calculate resampling ratio
    ratio = dst_rate / src_rate
    num_samples = data.shape[axis]
    new_num_samples = int(np.round(num_samples * ratio))

    if method == "fft":
        # FFT-based resampling (higher quality, slower)
        return signal.resample(data, new_num_samples, axis=axis)
    elif method == "polyphase":
        # Polyphase filtering (faster, good quality)
        # Calculate up/down factors
        from math import gcd
        up = int(dst_rate)
        down = int(src_rate)
        g = gcd(up, down)
        up //= g
        down //= g

        return signal.resample_poly(data, up, down, axis=axis)
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def apply_channel_groups(
    data: np.ndarray,
    groups: Dict[str, List[int]],
    reducer: Literal["mean", "median", "first", "concat"] = "mean"
) -> Dict[str, np.ndarray]:
    """
    Apply logical channel grouping for spatial filtering and compression.

    Args:
        data: Input neural data (channels x samples)
        groups: Dictionary mapping group names to channel indices
                e.g., {'motor_strip': [8,9,10,11], 'emg_pair_1': [30,31]}
        reducer: How to combine channels within a group
                 - 'mean': average across channels
                 - 'median': median across channels
                 - 'first': use first channel only
                 - 'concat': concatenate channels

    Returns:
        Dictionary mapping group names to processed data arrays

    Example:
        >>> groups = {'motor_strip': [8,9,10,11], 'emg_pair': [30,31]}
        >>> grouped = apply_channel_groups(data, groups, reducer='mean')
        >>> motor_data = grouped['motor_strip']  # (1 x samples)

    Note:
        Assumes data is in channels x samples format. If your data is
        samples x channels, transpose it first.
    """
    result = {}

    for group_name, channel_indices in groups.items():
        # Extract channels for this group
        group_data = data[channel_indices, :]

        # Apply reducer
        if reducer == "mean":
            result[group_name] = np.mean(group_data, axis=0, keepdims=True)
        elif reducer == "median":
            result[group_name] = np.median(group_data, axis=0, keepdims=True)
        elif reducer == "first":
            result[group_name] = group_data[0:1, :]
        elif reducer == "concat":
            result[group_name] = group_data
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

    return result


def load_mapping_file(filepath: str) -> Dict:
    """
    Load electrode mapping from YAML or JSON file.

    Args:
        filepath: Path to mapping file (.yaml, .yml, or .json)

    Returns:
        Dictionary containing mapping configuration

    Example mapping file (YAML):
        ```yaml
        device: openbci_32
        sampling_rate: 250
        mapping:
          ch_0: FP1
          ch_1: FP2
          ch_2: F3
        channel_groups:
          motor_strip: [8,9,10,11]
          emg_pair_1: [30,31]
        ```
    """
    if filepath.endswith(('.yaml', '.yml')):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def save_mapping_file(mapping: Dict, filepath: str) -> None:
    """
    Save electrode mapping to YAML or JSON file.

    Args:
        mapping: Mapping configuration dictionary
        filepath: Output file path (.yaml, .yml, or .json)
    """
    if filepath.endswith(('.yaml', '.yml')):
        with open(filepath, 'w') as f:
            yaml.dump(mapping, f, default_flow_style=False)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(mapping, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def apply_calibration(
    data: np.ndarray,
    calibration: Dict[str, Union[float, List[float], np.ndarray]],
    channels: Optional[List[int]] = None
) -> np.ndarray:
    """
    Apply per-channel calibration (scaling, offsets, bad channel masking).

    Args:
        data: Input neural data (channels x samples)
        calibration: Dictionary with 'scale', 'offset', 'bad_channels' keys
        channels: Optional list of channel indices to calibrate (default: all)

    Returns:
        Calibrated neural data

    Example calibration dict:
        ```python
        calibration = {
            'scale': [1.0, 1.2, 0.95, ...],  # per-channel gain
            'offset': [0.0, -0.5, 0.3, ...],  # per-channel DC offset
            'bad_channels': [5, 12]  # channels to zero out
        }
        ```
    """
    # Ensure data is channels x samples
    if data.shape[0] > data.shape[1]:
        data = data.T
        transposed = True
    else:
        transposed = False

    calibrated = data.copy()
    n_channels = data.shape[0]

    if channels is None:
        channels = list(range(n_channels))

    # Apply scaling
    if 'scale' in calibration:
        scale = np.array(calibration['scale'])
        if scale.size == 1:
            scale = np.full(n_channels, scale)
        calibrated[channels] *= scale[channels].reshape(-1, 1)

    # Apply offset
    if 'offset' in calibration:
        offset = np.array(calibration['offset'])
        if offset.size == 1:
            offset = np.full(n_channels, offset)
        calibrated[channels] += offset[channels].reshape(-1, 1)

    # Mask bad channels
    if 'bad_channels' in calibration:
        bad_channels = calibration['bad_channels']
        calibrated[bad_channels] = 0.0

    return calibrated.T if transposed else calibrated


# Export public API
__all__ = [
    'map_channels',
    'resample',
    'apply_channel_groups',
    'load_mapping_file',
    'save_mapping_file',
    'apply_calibration',
]
