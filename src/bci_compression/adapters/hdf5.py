"""
Generic HDF5 Adapter

Provides a flexible adapter for HDF5-format neural data files.
Supports custom field specifications and auto-detection.
"""

from typing import Dict, Optional, List, Union
import numpy as np
import h5py
from pathlib import Path
from . import map_channels, resample, load_mapping_file, save_mapping_file


class HDF5Adapter:
    """
    Generic adapter for HDF5 neural data files.
    
    Supports:
    - Custom field specifications
    - Auto-detection of data structure
    - Multiple datasets within one file
    - Metadata extraction
    
    Example:
        >>> adapter = HDF5Adapter.from_hdf5('data.h5', data_path='/neural/raw')
        >>> data = adapter.load_data()
        >>> resampled = adapter.resample_to(data, target_rate=1000)
    """
    
    def __init__(
        self,
        filepath: Optional[str] = None,
        data_path: str = '/data',
        sampling_rate: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize HDF5 adapter.
        
        Args:
            filepath: Path to HDF5 file
            data_path: HDF5 path to data array (e.g., '/neural/raw')
            sampling_rate: Sampling rate in Hz (auto-detected if possible)
            channel_names: Optional list of channel names
            metadata: Optional metadata dictionary
        """
        self.filepath = filepath
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.metadata = metadata or {}
        self._file_handle = None
    
    @classmethod
    def from_hdf5(
        cls,
        filepath: str,
        data_path: str = '/data',
        sampling_rate_path: Optional[str] = None,
        channel_names_path: Optional[str] = None
    ) -> 'HDF5Adapter':
        """
        Create adapter from HDF5 file with auto-detection.
        
        Args:
            filepath: Path to HDF5 file
            data_path: HDF5 path to data array
            sampling_rate_path: HDF5 path to sampling rate (optional)
            channel_names_path: HDF5 path to channel names (optional)
        
        Returns:
            HDF5Adapter configured from file
        
        Example:
            >>> adapter = HDF5Adapter.from_hdf5(
            ...     'recording.h5',
            ...     data_path='/neural/data',
            ...     sampling_rate_path='/neural/sampling_rate'
            ... )
        """
        with h5py.File(filepath, 'r') as f:
            # Extract sampling rate if path provided
            sampling_rate = None
            if sampling_rate_path and sampling_rate_path in f:
                sampling_rate = float(f[sampling_rate_path][()])
            
            # Extract channel names if path provided
            channel_names = None
            if channel_names_path and channel_names_path in f:
                channel_names = [name.decode() if isinstance(name, bytes) else str(name)
                               for name in f[channel_names_path][()]]
            
            # Extract metadata
            metadata = dict(f.attrs) if hasattr(f, 'attrs') else {}
        
        return cls(
            filepath=filepath,
            data_path=data_path,
            sampling_rate=sampling_rate,
            channel_names=channel_names,
            metadata=metadata
        )
    
    def load_data(
        self,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None,
        channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Load data from HDF5 file.
        
        Args:
            start_sample: Starting sample index (None = beginning)
            end_sample: Ending sample index (None = end)
            channels: List of channel indices to load (None = all)
        
        Returns:
            Neural data array (channels x samples)
        """
        with h5py.File(self.filepath, 'r') as f:
            if self.data_path not in f:
                raise ValueError(f"Data path '{self.data_path}' not found in HDF5 file")
            
            dataset = f[self.data_path]
            
            # Determine array shape
            if len(dataset.shape) == 2:
                # Assume channels x samples or samples x channels
                if dataset.shape[0] > dataset.shape[1]:
                    # Likely samples x channels, transpose
                    if channels is not None:
                        data = dataset[start_sample:end_sample, channels].T
                    else:
                        data = dataset[start_sample:end_sample, :].T
                else:
                    # Likely channels x samples
                    if channels is not None:
                        data = dataset[channels, start_sample:end_sample]
                    else:
                        data = dataset[:, start_sample:end_sample]
            else:
                raise ValueError(f"Unexpected data shape: {dataset.shape}")
            
            return data
    
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
        if self.sampling_rate is None:
            raise ValueError("Sampling rate not set. Cannot resample.")
        
        return resample(data, self.sampling_rate, target_rate, method=method)
    
    def get_info(self) -> Dict:
        """
        Get information about the HDF5 file structure.
        
        Returns:
            Dictionary with file info
        """
        with h5py.File(self.filepath, 'r') as f:
            info = {
                'filepath': self.filepath,
                'data_path': self.data_path,
                'sampling_rate': self.sampling_rate,
                'channel_names': self.channel_names,
                'metadata': self.metadata,
                'available_datasets': list(f.keys()),
            }
            
            if self.data_path in f:
                dataset = f[self.data_path]
                info['data_shape'] = dataset.shape
                info['data_dtype'] = str(dataset.dtype)
                info['data_size_mb'] = dataset.size * dataset.dtype.itemsize / (1024**2)
            
            return info
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets in HDF5 file.
        
        Returns:
            List of dataset paths
        """
        datasets = []
        
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        
        with h5py.File(self.filepath, 'r') as f:
            f.visititems(visit_func)
        
        return datasets
    
    def create_mapping_from_metadata(self) -> Dict:
        """
        Create a device mapping from HDF5 metadata.
        
        Returns:
            Mapping dictionary compatible with other adapters
        """
        mapping = {
            'device': 'hdf5_generic',
            'sampling_rate': self.sampling_rate,
            'data_path': self.data_path,
            'metadata': self.metadata,
        }
        
        if self.channel_names:
            mapping['mapping'] = {f'ch_{i}': name 
                                 for i, name in enumerate(self.channel_names)}
            mapping['channels'] = len(self.channel_names)
        
        return mapping
    
    def save_mapping(self, filepath: str) -> None:
        """Save current mapping to file."""
        mapping = self.create_mapping_from_metadata()
        save_mapping_file(mapping, filepath)


def load_hdf5_neural_data(
    filepath: str,
    data_path: str = '/data',
    sampling_rate: Optional[float] = None,
    target_rate: Optional[float] = None
) -> np.ndarray:
    """
    Quick loader for HDF5 neural data.
    
    Args:
        filepath: Path to HDF5 file
        data_path: HDF5 path to data array
        sampling_rate: Source sampling rate (required for resampling)
        target_rate: Optional target sampling rate for resampling
    
    Returns:
        Neural data array (and optionally resampled)
    
    Example:
        >>> data = load_hdf5_neural_data(
        ...     'recording.h5',
        ...     data_path='/neural/raw',
        ...     sampling_rate=30000,
        ...     target_rate=1000
        ... )
    """
    adapter = HDF5Adapter(
        filepath=filepath,
        data_path=data_path,
        sampling_rate=sampling_rate
    )
    
    data = adapter.load_data()
    
    if target_rate and sampling_rate:
        data = adapter.resample_to(data, target_rate)
    
    return data


# Export public API
__all__ = [
    'HDF5Adapter',
    'load_hdf5_neural_data',
]
