"""
Intan Technologies Device Adapter

Provides converters and mappings for Intan RHD and RHS systems:
- RHD2000 series (up to 512 channels)
- RHS series (stimulation + recording)
- Custom headstage configurations
"""

from typing import Dict, Optional, List
import numpy as np
from . import map_channels, resample, load_mapping_file, save_mapping_file


# Intan RHD2132 32-channel headstage
INTAN_RHD2132_32CH_MAPPING = {
    'device': 'intan_rhd2132_32ch',
    'sampling_rate': 20000,
    'channels': 32,
    'headstage_type': 'RHD2132',
    'mapping': {
        **{f'ch_{i}': f'amp_ch_{i:02d}' for i in range(32)}
    },
    'channel_groups': {
        'headstage_1': list(range(0, 16)),
        'headstage_2': list(range(16, 32)),
        'odd_channels': list(range(0, 32, 2)),
        'even_channels': list(range(1, 32, 2)),
    }
}

# Intan RHD2164 64-channel headstage
INTAN_RHD2164_64CH_MAPPING = {
    'device': 'intan_rhd2164_64ch',
    'sampling_rate': 20000,
    'channels': 64,
    'headstage_type': 'RHD2164',
    'mapping': {
        **{f'ch_{i}': f'amp_ch_{i:02d}' for i in range(64)}
    },
    'channel_groups': {
        'headstage_1': list(range(0, 32)),
        'headstage_2': list(range(32, 64)),
        'quad_1': list(range(0, 16)),
        'quad_2': list(range(16, 32)),
        'quad_3': list(range(32, 48)),
        'quad_4': list(range(48, 64)),
    }
}

# Intan RHS multi-channel stim/recording
INTAN_RHS_128CH_MAPPING = {
    'device': 'intan_rhs_128ch',
    'sampling_rate': 30000,
    'channels': 128,
    'headstage_type': 'RHS2116',
    'stim_capable': True,
    'mapping': {
        **{f'ch_{i}': f'stim_amp_ch_{i:03d}' for i in range(128)}
    },
    'channel_groups': {
        'bank_A': list(range(0, 32)),
        'bank_B': list(range(32, 64)),
        'bank_C': list(range(64, 96)),
        'bank_D': list(range(96, 128)),
    }
}


class IntanAdapter:
    """
    Adapter for Intan Technologies recording systems.
    
    Supports:
    - RHD2000 series amplifiers (up to 512 channels)
    - RHS series stim/recording systems
    - Multiple headstage configurations
    - .rhd and .rhs file formats
    
    Example:
        >>> adapter = IntanAdapter(device='rhd2132_32ch')
        >>> standardized = adapter.convert(raw_data)
        >>> downsampled = adapter.resample_to(standardized, target_rate=1000)
    """
    
    def __init__(self, device: str = 'rhd2132_32ch', custom_mapping: Optional[Dict] = None):
        """
        Initialize Intan adapter.
        
        Args:
            device: Device type ('rhd2132_32ch', 'rhd2164_64ch', 'rhs_128ch')
            custom_mapping: Optional custom mapping dictionary
        """
        if custom_mapping:
            self.mapping = custom_mapping
        elif device == 'rhd2132_32ch':
            self.mapping = INTAN_RHD2132_32CH_MAPPING
        elif device == 'rhd2164_64ch':
            self.mapping = INTAN_RHD2164_64CH_MAPPING
        elif device == 'rhs_128ch':
            self.mapping = INTAN_RHS_128CH_MAPPING
        else:
            raise ValueError(f"Unknown device type: {device}")
        
        self.device = device
        self.sampling_rate = self.mapping['sampling_rate']
        self.headstage_type = self.mapping.get('headstage_type', 'unknown')
        self.stim_capable = self.mapping.get('stim_capable', False)
    
    def convert(self, data: np.ndarray, apply_mapping: bool = True) -> np.ndarray:
        """
        Convert Intan raw data to standardized format.
        
        Args:
            data: Raw Intan data (channels x samples)
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
    
    def get_channel_groups(self) -> Dict[str, List[int]]:
        """Get channel groupings for this device."""
        return self.mapping.get('channel_groups', {})
    
    def get_headstage_type(self) -> str:
        """Get the headstage type."""
        return self.headstage_type
    
    def is_stim_capable(self) -> bool:
        """Check if device supports stimulation."""
        return self.stim_capable
    
    def save_mapping(self, filepath: str) -> None:
        """Save current mapping to file."""
        save_mapping_file(self.mapping, filepath)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'IntanAdapter':
        """Load adapter from mapping file."""
        mapping = load_mapping_file(filepath)
        return cls(custom_mapping=mapping)
    
    @classmethod
    def from_rhd_file(cls, rhd_filepath: str) -> 'IntanAdapter':
        """
        Create adapter from RHD file metadata.
        
        Note: Requires intanutil library for RHD parsing.
        This is a placeholder for future implementation.
        
        Args:
            rhd_filepath: Path to RHD file
        
        Returns:
            IntanAdapter configured from file metadata
        """
        # TODO: Implement RHD file parsing
        raise NotImplementedError("RHD file parsing requires intanutil library")


def convert_intan_to_standard(
    data: np.ndarray,
    device: str = 'rhd2132_32ch',
    target_rate: Optional[float] = None
) -> np.ndarray:
    """
    Quick converter for Intan data to standardized format.
    
    Args:
        data: Raw Intan data (channels x samples)
        device: Device type ('rhd2132_32ch', 'rhd2164_64ch', 'rhs_128ch')
        target_rate: Optional target sampling rate for resampling
    
    Returns:
        Standardized (and optionally resampled) neural data
    
    Example:
        >>> # Downsample from 20kHz to 1kHz
        >>> standard_data = convert_intan_to_standard(
        ...     raw_data, 'rhd2132_32ch', target_rate=1000
        ... )
    """
    adapter = IntanAdapter(device=device)
    converted = adapter.convert(data)
    
    if target_rate:
        converted = adapter.resample_to(converted, target_rate)
    
    return converted


# Export public API
__all__ = [
    'IntanAdapter',
    'convert_intan_to_standard',
    'INTAN_RHD2132_32CH_MAPPING',
    'INTAN_RHD2164_64CH_MAPPING',
    'INTAN_RHS_128CH_MAPPING',
]
