"""
BCI Data Adapter

Adapts data from different BCI systems to standardized format for compression.
Handles:
- Resampling to target sampling rates
- Channel selection and reordering
- Voltage scaling and normalization
- Data type conversion
"""

import numpy as np
from typing import Optional, Tuple, Union
from .system_profiles import BCISystemProfile, get_system_profile


class BCIDataAdapter:
    """
    Adapter for converting BCI data between different system formats.
    
    Parameters
    ----------
    source_profile : Union[str, BCISystemProfile]
        Source system profile (name or profile object)
    target_profile : Optional[Union[str, BCISystemProfile]]
        Target system profile (if None, uses source profile)
    target_sampling_rate : Optional[int]
        Override target sampling rate
    normalize : bool
        Whether to normalize voltage to [-1, 1] range
    """
    
    def __init__(
        self,
        source_profile: Union[str, BCISystemProfile],
        target_profile: Optional[Union[str, BCISystemProfile]] = None,
        target_sampling_rate: Optional[int] = None,
        normalize: bool = False
    ):
        # Load profiles
        if isinstance(source_profile, str):
            self.source_profile = get_system_profile(source_profile)
        else:
            self.source_profile = source_profile
            
        if target_profile is None:
            self.target_profile = self.source_profile
        elif isinstance(target_profile, str):
            self.target_profile = get_system_profile(target_profile)
        else:
            self.target_profile = target_profile
        
        # Override sampling rate if specified
        if target_sampling_rate is not None:
            # Create modified profile with target sampling rate
            self.target_profile = BCISystemProfile(
                name=self.target_profile.name,
                num_channels=self.target_profile.num_channels,
                sampling_rate=target_sampling_rate,
                electrode_standard=self.target_profile.electrode_standard,
                bit_depth=self.target_profile.bit_depth,
                voltage_range=self.target_profile.voltage_range,
                recommended_compression=self.target_profile.recommended_compression,
                data_type=self.target_profile.data_type,
                manufacturer=self.target_profile.manufacturer,
                description=self.target_profile.description,
            )
        
        self.normalize = normalize
        
    def adapt(self, data: np.ndarray) -> np.ndarray:
        """
        Adapt data from source to target format.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with shape (channels, samples) or (samples,)
            
        Returns
        -------
        np.ndarray
            Adapted data
        """
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        adapted_data = data.copy()
        
        # Step 1: Voltage scaling
        adapted_data = self._scale_voltage(adapted_data)
        
        # Step 2: Resampling
        if self.source_profile.sampling_rate != self.target_profile.sampling_rate:
            adapted_data = self._resample(adapted_data)
        
        # Step 3: Normalization
        if self.normalize:
            adapted_data = self._normalize(adapted_data)
        
        # Step 4: Data type conversion
        adapted_data = self._convert_dtype(adapted_data)
        
        return adapted_data
    
    def _scale_voltage(self, data: np.ndarray) -> np.ndarray:
        """Scale voltage from source to target range."""
        source_min, source_max = self.source_profile.voltage_range
        target_min, target_max = self.target_profile.voltage_range
        
        # Convert to normalized 0-1 range based on source
        source_range = source_max - source_min
        normalized = (data - source_min) / source_range
        
        # Scale to target range
        target_range = target_max - target_min
        scaled = normalized * target_range + target_min
        
        return scaled
    
    def _resample(self, data: np.ndarray) -> np.ndarray:
        """
        Resample data to target sampling rate.
        
        Uses simple linear interpolation for efficiency.
        For production, consider using scipy.signal.resample_poly for better quality.
        """
        source_rate = self.source_profile.sampling_rate
        target_rate = self.target_profile.sampling_rate
        
        n_channels, n_samples = data.shape
        
        # Calculate target length
        target_length = int(n_samples * target_rate / source_rate)
        
        # Create interpolation indices
        source_indices = np.arange(n_samples)
        target_indices = np.linspace(0, n_samples - 1, target_length)
        
        # Interpolate each channel
        resampled = np.zeros((n_channels, target_length), dtype=data.dtype)
        for ch in range(n_channels):
            resampled[ch] = np.interp(target_indices, source_indices, data[ch])
        
        return resampled
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        min_val, max_val = self.target_profile.voltage_range
        voltage_range = max_val - min_val
        
        # Normalize to [-1, 1]
        normalized = 2 * (data - min_val) / voltage_range - 1
        
        return normalized
    
    def _convert_dtype(self, data: np.ndarray) -> np.ndarray:
        """Convert to appropriate data type based on bit depth."""
        bit_depth = self.target_profile.bit_depth
        
        if bit_depth <= 8:
            dtype = np.int8
        elif bit_depth <= 16:
            dtype = np.int16
        elif bit_depth <= 32:
            dtype = np.int32
        else:
            dtype = np.float32
        
        # Only convert if not already float (preserve float precision)
        if not np.issubdtype(data.dtype, np.floating):
            return data.astype(dtype)
        
        return data
    
    def get_compression_settings(self) -> dict:
        """
        Get recommended compression settings for target system.
        
        Returns
        -------
        dict
            Compression settings
        """
        return {
            'algorithm': self.target_profile.recommended_compression,
            'data_type': self.target_profile.data_type,
            'sampling_rate': self.target_profile.sampling_rate,
            'num_channels': self.target_profile.num_channels,
            'bit_depth': self.target_profile.bit_depth,
        }


def adapt_data(
    data: np.ndarray,
    source_system: str,
    target_system: Optional[str] = None,
    target_sampling_rate: Optional[int] = None,
    normalize: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to adapt data between BCI systems.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (channels, samples) or (samples,)
    source_system : str
        Name of source BCI system
    target_system : Optional[str]
        Name of target BCI system (if None, uses source)
    target_sampling_rate : Optional[int]
        Override target sampling rate
    normalize : bool
        Whether to normalize voltage to [-1, 1] range
        
    Returns
    -------
    adapted_data : np.ndarray
        Adapted data
    settings : dict
        Recommended compression settings
        
    Examples
    --------
    >>> import numpy as np
    >>> from bci_compression.formats import adapt_data
    >>> 
    >>> # Simulate OpenBCI data
    >>> data = np.random.randn(8, 2000)
    >>> 
    >>> # Adapt to standard format with 1kHz sampling
    >>> adapted, settings = adapt_data(
    ...     data,
    ...     source_system='openbci_8',
    ...     target_sampling_rate=1000
    ... )
    >>> 
    >>> print(f"Adapted shape: {adapted.shape}")
    >>> print(f"Recommended algorithm: {settings['algorithm']}")
    """
    adapter = BCIDataAdapter(
        source_profile=source_system,
        target_profile=target_system,
        target_sampling_rate=target_sampling_rate,
        normalize=normalize
    )
    
    adapted_data = adapter.adapt(data)
    settings = adapter.get_compression_settings()
    
    return adapted_data, settings
