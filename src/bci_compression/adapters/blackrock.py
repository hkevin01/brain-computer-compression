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
        Create adapter from NEV file metadata.
        
        Note: Requires neo or blackrock_io library for NEV parsing.
        This is a placeholder for future implementation.
        
        Args:
            nev_filepath: Path to NEV file
        
        Returns:
            BlackrockAdapter configured from file metadata
        """
        # TODO: Implement NEV file parsing
        raise NotImplementedError("NEV file parsing requires neo library")


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
