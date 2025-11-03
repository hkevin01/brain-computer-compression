"""
OpenBCI Device Adapter

Provides converters and mappings for OpenBCI Cyton (8-channel) and 
Daisy (16-channel) boards to standard 10-20 electrode positions.
"""

from typing import Dict, Optional
import numpy as np
from . import map_channels, resample, load_mapping_file, save_mapping_file


# Standard OpenBCI Cyton 8-channel mapping (10-20 system)
OPENBCI_CYTON_8CH_MAPPING = {
    'device': 'openbci_cyton_8ch',
    'sampling_rate': 250,
    'channels': 8,
    'mapping': {
        'ch_0': 'Fp1',
        'ch_1': 'Fp2',
        'ch_2': 'C3',
        'ch_3': 'C4',
        'ch_4': 'P7',
        'ch_5': 'P8',
        'ch_6': 'O1',
        'ch_7': 'O2',
    },
    'channel_groups': {
        'frontal': [0, 1],
        'central': [2, 3],
        'parietal': [4, 5],
        'occipital': [6, 7],
    }
}

# OpenBCI Cyton + Daisy 16-channel mapping
OPENBCI_DAISY_16CH_MAPPING = {
    'device': 'openbci_daisy_16ch',
    'sampling_rate': 250,
    'channels': 16,
    'mapping': {
        'ch_0': 'Fp1',
        'ch_1': 'Fp2',
        'ch_2': 'C3',
        'ch_3': 'C4',
        'ch_4': 'P7',
        'ch_5': 'P8',
        'ch_6': 'O1',
        'ch_7': 'O2',
        'ch_8': 'F7',
        'ch_9': 'F8',
        'ch_10': 'F3',
        'ch_11': 'F4',
        'ch_12': 'T7',
        'ch_13': 'T8',
        'ch_14': 'P3',
        'ch_15': 'P4',
    },
    'channel_groups': {
        'frontal': [0, 1, 8, 9, 10, 11],
        'central': [2, 3],
        'temporal': [4, 5, 12, 13],
        'parietal': [14, 15],
        'occipital': [6, 7],
    }
}


class OpenBCIAdapter:
    """
    Adapter for OpenBCI device data to standardized format.
    
    Example:
        >>> adapter = OpenBCIAdapter(device='cyton_8ch')
        >>> standardized = adapter.convert(raw_data)
        >>> resampled = adapter.resample_to(standardized, target_rate=1000)
    """
    
    def __init__(self, device: str = 'cyton_8ch', custom_mapping: Optional[Dict] = None):
        """
        Initialize OpenBCI adapter.
        
        Args:
            device: Device type ('cyton_8ch' or 'daisy_16ch')
            custom_mapping: Optional custom mapping dictionary
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device == 'cyton_8ch':
            self.mapping = OPENBCI_CYTON_8CH_MAPPING
        elif device == 'daisy_16ch':
            self.mapping = OPENBCI_DAISY_16CH_MAPPING
        else:
            raise ValueError(f"Unknown device type: {device}")
        
        self.device = device
        self.sampling_rate = self.mapping['sampling_rate']
    
    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Convert OpenBCI raw data to standardized format.
        
        Args:
            data: Raw OpenBCI data (channels x samples)
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
        
        Args:
            data: Input data
            target_rate: Target sampling rate in Hz
            method: Resampling method ('fft' or 'polyphase')
        
        Returns:
            Resampled data
        """
        return resample(data, self.sampling_rate, target_rate, method=method)
    
    def get_channel_groups(self) -> Dict[str, list]:
        """Get channel groupings for this device."""
        return self.mapping.get('channel_groups', {})
    
    def save_mapping(self, filepath: str) -> None:
        """Save current mapping to file."""
        save_mapping_file(self.mapping, filepath)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'OpenBCIAdapter':
        """Load adapter from mapping file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)


def convert_openbci_to_standard(
    data: np.ndarray,
    device: str = 'cyton_8ch',
    target_rate: Optional[float] = None
) -> np.ndarray:
    """
    Quick converter for OpenBCI data to standardized format.
    
    Args:
        data: Raw OpenBCI data (channels x samples)
        device: Device type ('cyton_8ch' or 'daisy_16ch')
        target_rate: Optional target sampling rate for resampling
    
    Returns:
        Standardized (and optionally resampled) neural data
    
    Example:
        >>> standard_data = convert_openbci_to_standard(raw_data, 'cyton_8ch', target_rate=1000)
    """
    adapter = OpenBCIAdapter(device=device)
    converted = adapter.convert(data)
    
    if target_rate:
        converted = adapter.resample_to(converted, target_rate)
    
    return converted


# Export public API
__all__ = [
    'OpenBCIAdapter',
    'convert_openbci_to_standard',
    'OPENBCI_CYTON_8CH_MAPPING',
    'OPENBCI_DAISY_16CH_MAPPING',
]
