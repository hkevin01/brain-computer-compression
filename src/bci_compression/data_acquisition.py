"""
Data acquisition framework for neural recording devices.

This module provides interfaces for acquiring neural data from various
recording devices commonly used in BCI applications, including support
for real-time streaming and data buffering. Extended to support EMG data
acquisition from common EMG recording formats.
"""

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np

# EMG data format imports (with optional dependencies)
try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class NeuralDataSource(Protocol):
    """Protocol for neural data sources."""

    def start_acquisition(self) -> bool:
        """Start data acquisition."""
        ...

    def stop_acquisition(self) -> bool:
        """Stop data acquisition."""
        ...

    def read_data(self) -> Optional[np.ndarray]:
        """Read available data."""
        ...

    def get_sampling_rate(self) -> float:
        """Get the sampling rate."""
        ...


class BaseDataAcquisition(ABC):
    """
    Abstract base class for neural data acquisition systems.

    This class defines the interface for acquiring neural signals
    from various recording devices and provides common functionality
    for buffering and streaming.
    """

    def __init__(
        self,
        sampling_rate: float,
        n_channels: int,
        buffer_duration: float = 1.0
    ):
        """
        Initialize data acquisition system.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        n_channels : int
            Number of recording channels
        buffer_duration : float, default=1.0
            Buffer duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sampling_rate * buffer_duration)

        # Data buffer
        self.data_buffer = np.zeros((n_channels, self.buffer_size))
        self.buffer_index = 0
        self.is_acquiring = False

        # Threading
        self.acquisition_thread = None
        self.data_queue = Queue()

        # Callbacks
        self.data_callback: Optional[Callable[[np.ndarray], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the recording device."""

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the recording device."""

    @abstractmethod
    def _acquire_data_chunk(self) -> Optional[np.ndarray]:
        """Acquire a chunk of data from the device."""

    def set_data_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for new data."""
        self.data_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self.error_callback = callback

    def _acquisition_loop(self) -> None:
        """Main acquisition loop."""
        while self.is_acquiring:
            try:
                # Get new data chunk
                data_chunk = self._acquire_data_chunk()

                if data_chunk is not None:
                    # Add to buffer
                    chunk_size = data_chunk.shape[1]

                    # Handle buffer overflow
                    if self.buffer_index + chunk_size > self.buffer_size:
                        # Shift buffer
                        shift_amount = self.buffer_index + chunk_size - self.buffer_size
                        self.data_buffer[:, :-shift_amount] = self.data_buffer[:, shift_amount:]
                        self.buffer_index = self.buffer_size - chunk_size

                    # Add new data
                    end_idx = self.buffer_index + chunk_size
                    self.data_buffer[:, self.buffer_index:end_idx] = data_chunk
                    self.buffer_index = end_idx

                    # Trigger callback
                    if self.data_callback:
                        self.data_callback(data_chunk)

                    # Add to queue
                    self.data_queue.put(data_chunk)

                # Small sleep to prevent CPU overload
                time.sleep(0.001)

            except Exception as e:
                if self.error_callback:
                    self.error_callback(e)
                else:
                    print(f"Error in acquisition loop: {e}")

    def start_acquisition(self) -> bool:
        """Start data acquisition."""
        if self.is_acquiring:
            return True

        if not self.connect():
            return False

        self.is_acquiring = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self.acquisition_thread.start()

        print(f"Data acquisition started at {self.sampling_rate} Hz")
        return True

    def stop_acquisition(self) -> bool:
        """Stop data acquisition."""
        if not self.is_acquiring:
            return True

        self.is_acquiring = False

        if self.acquisition_thread:
            self.acquisition_thread.join()

        self.disconnect()
        print("Data acquisition stopped")
        return True

    def get_latest_data(self, duration: float) -> np.ndarray:
        """
        Get the most recent data from the buffer.

        Parameters
        ----------
        duration : float
            Duration of data to retrieve in seconds

        Returns
        -------
        np.ndarray
            Recent data with shape (channels, samples)
        """
        n_samples = int(duration * self.sampling_rate)
        n_samples = min(n_samples, self.buffer_index)

        if n_samples <= 0:
            return np.array([]).reshape(self.n_channels, 0)

        start_idx = max(0, self.buffer_index - n_samples)
        return self.data_buffer[:, start_idx:self.buffer_index].copy()

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information."""
        return {
            'buffer_size': self.buffer_size,
            'current_index': self.buffer_index,
            'fill_percentage': (self.buffer_index / self.buffer_size) * 100,
            'duration_filled': self.buffer_index / self.sampling_rate
        }


class SimulatedDataAcquisition(BaseDataAcquisition):
    """
    Simulated data acquisition for testing and development.

    This class generates synthetic neural signals that can be used
    for testing BCI algorithms without requiring actual hardware.
    """

    def __init__(
        self,
        sampling_rate: float = 30000.0,
        n_channels: int = 64,
        buffer_duration: float = 1.0,
        noise_level: float = 0.1,
        signal_amplitude: float = 100.0
    ):
        """
        Initialize simulated data acquisition.

        Parameters
        ----------
        sampling_rate : float, default=30000.0
            Sampling rate in Hz
        n_channels : int, default=64
            Number of simulated channels
        buffer_duration : float, default=1.0
            Buffer duration in seconds
        noise_level : float, default=0.1
            Noise level (standard deviation)
        signal_amplitude : float, default=100.0
            Signal amplitude in microvolts
        """
        super().__init__(sampling_rate, n_channels, buffer_duration)
        self.noise_level = noise_level
        self.signal_amplitude = signal_amplitude
        self.chunk_size = int(sampling_rate * 0.01)  # 10ms chunks

        # Signal generation parameters
        self.time_counter = 0
        self.phase_offsets = np.random.uniform(0, 2 * np.pi, n_channels)
        self.frequencies = np.random.uniform(8, 30, n_channels)  # 8-30 Hz

    def connect(self) -> bool:
        """Simulate connection to device."""
        print("Connected to simulated neural recording device")
        return True

    def disconnect(self) -> bool:
        """Simulate disconnection from device."""
        print("Disconnected from simulated device")
        return True

    def _acquire_data_chunk(self) -> Optional[np.ndarray]:
        """Generate simulated neural data chunk."""
        # Time vector for this chunk
        t_start = self.time_counter / self.sampling_rate
        t_end = (self.time_counter + self.chunk_size) / self.sampling_rate
        t = np.linspace(t_start, t_end, self.chunk_size, endpoint=False)

        # Generate signals for each channel
        data_chunk = np.zeros((self.n_channels, self.chunk_size))

        for ch in range(self.n_channels):
            # Base sinusoidal signal
            signal = self.signal_amplitude * np.sin(
                2 * np.pi * self.frequencies[ch] * t + self.phase_offsets[ch]
            )

            # Add harmonics
            signal += 0.3 * self.signal_amplitude * np.sin(
                4 * np.pi * self.frequencies[ch] * t + self.phase_offsets[ch]
            )

            # Add 1/f noise
            noise = np.random.normal(0, self.noise_level * self.signal_amplitude, self.chunk_size)

            # Combine signal and noise
            data_chunk[ch, :] = signal + noise

        self.time_counter += self.chunk_size
        return data_chunk


class FileDataAcquisition(BaseDataAcquisition):
    """
    File-based data acquisition for replaying recorded data.

    This class can replay neural data from files, useful for
    testing algorithms with real recorded data.
    """

    def __init__(
        self,
        file_path: str,
        sampling_rate: float,
        buffer_duration: float = 1.0,
        loop_data: bool = True,
        playback_speed: float = 1.0
    ):
        """
        Initialize file-based data acquisition.

        Parameters
        ----------
        file_path : str
            Path to the data file
        sampling_rate : float
            Sampling rate of the data
        buffer_duration : float, default=1.0
            Buffer duration in seconds
        loop_data : bool, default=True
            Whether to loop the data when reaching the end
        playback_speed : float, default=1.0
            Playback speed multiplier
        """
        self.file_path = file_path
        self.loop_data = loop_data
        self.playback_speed = playback_speed

        # Load data
        self.file_data = self._load_data_file()
        n_channels = self.file_data.shape[0]

        super().__init__(sampling_rate, n_channels, buffer_duration)

        self.file_index = 0
        self.chunk_size = int(sampling_rate * 0.01 * playback_speed)  # Adjusted for speed

    def _load_data_file(self) -> np.ndarray:
        """Load data from file."""
        try:
            if self.file_path.endswith('.npy'):
                return np.load(self.file_path)
            elif self.file_path.endswith('.npz'):
                data = np.load(self.file_path)
                # Assume the data is stored with key 'data'
                return data['data'] if 'data' in data else data[list(data.keys())[0]]
            else:
                # Try to load as text
                return np.loadtxt(self.file_path)
        except Exception as e:
            raise ValueError(f"Could not load data file {self.file_path}: {e}")

    def connect(self) -> bool:
        """Open file for reading."""
        print(f"Opened data file: {self.file_path}")
        print(f"Data shape: {self.file_data.shape}")
        return True

    def disconnect(self) -> bool:
        """Close file."""
        print("Closed data file")
        return True

    def _acquire_data_chunk(self) -> Optional[np.ndarray]:
        """Read chunk from file."""
        if self.file_index >= self.file_data.shape[1]:
            if self.loop_data:
                self.file_index = 0
            else:
                return None

        # Get chunk
        end_idx = min(self.file_index + self.chunk_size, self.file_data.shape[1])
        chunk = self.file_data[:, self.file_index:end_idx]
        self.file_index = end_idx

        # Simulate real-time delay
        expected_delay = chunk.shape[1] / (self.sampling_rate * self.playback_speed)
        time.sleep(expected_delay)

        return chunk


class DataAcquisitionManager:
    """
    Manager for coordinating multiple data acquisition sources.

    This class can handle multiple data sources and provide
    synchronized data streams for multi-device recordings.
    """

    def __init__(self):
        """Initialize data acquisition manager."""
        self.sources: Dict[str, BaseDataAcquisition] = {}
        self.master_callback: Optional[Callable] = None
        self.is_synchronized = False

    def add_source(self, name: str, source: BaseDataAcquisition) -> None:
        """Add a data acquisition source."""
        self.sources[name] = source
        source.set_data_callback(self._handle_source_data)

    def remove_source(self, name: str) -> bool:
        """Remove a data acquisition source."""
        if name in self.sources:
            self.sources[name].stop_acquisition()
            del self.sources[name]
            return True
        return False

    def set_master_callback(self, callback: Callable) -> None:
        """Set callback for synchronized data."""
        self.master_callback = callback

    def _handle_source_data(self, data: np.ndarray) -> None:
        """Handle data from individual sources."""
        if self.master_callback:
            # For now, just pass through individual source data
            # In a full implementation, this would synchronize multiple sources
            self.master_callback(data)

    def start_all(self) -> bool:
        """Start all data sources."""
        success = True
        for name, source in self.sources.items():
            if not source.start_acquisition():
                print(f"Failed to start source: {name}")
                success = False
        return success

    def stop_all(self) -> bool:
        """Stop all data sources."""
        success = True
        for name, source in self.sources.items():
            if not source.stop_acquisition():
                print(f"Failed to stop source: {name}")
                success = False
        return success

    def get_source_status(self) -> Dict[str, Dict]:
        """Get status of all sources."""
        status = {}
        for name, source in self.sources.items():
            status[name] = {
                'is_acquiring': source.is_acquiring,
                'buffer_status': source.get_buffer_status()
            }
        return status


def create_test_acquisition_system(
    n_channels: int = 64,
    sampling_rate: float = 30000.0,
    use_simulated: bool = True
) -> DataAcquisitionManager:
    """
    Create a test data acquisition system.

    Parameters
    ----------
    n_channels : int, default=64
        Number of channels
    sampling_rate : float, default=30000.0
        Sampling rate in Hz
    use_simulated : bool, default=True
        Whether to use simulated data

    Returns
    -------
    DataAcquisitionManager
        Configured acquisition manager
    """
    manager = DataAcquisitionManager()

    if use_simulated:
        sim_source = SimulatedDataAcquisition(
            sampling_rate=sampling_rate,
            n_channels=n_channels
        )
        manager.add_source("simulated", sim_source)

    return manager


def load_edf_emg_data(file_path: str) -> np.ndarray:
    """
    Load EMG data from an EDF file.

    Parameters
    ----------
    file_path : str
        Path to the EDF file

    Returns
    -------
    np.ndarray
        EMG data with shape (channels, samples)
    """
    if not HAS_PYEDFLIB:
        raise ImportError("pyedflib is not installed")

    # Open EDF file
    with pyedflib.EdfReader(file_path) as edf_reader:
        n_channels = edf_reader.signals_in_file
        fs = edf_reader.getSampleFrequency(0)  # Assuming all channels have the same frequency

        # Read all signals
        all_signals = np.array([edf_reader.readSignal(i) for i in range(n_channels)])

    # Transpose to (channels, samples)
    return all_signals


def load_bdf_emg_data(file_path: str) -> np.ndarray:
    """
    Load EMG data from a BDF file.

    Parameters
    ----------
    file_path : str
        Path to the BDF file

    Returns
    -------
    np.ndarray
        EMG data with shape (channels, samples)
    """
    if not HAS_MNE:
        raise ImportError("mne is not installed")

    # Load BDF file
    raw = mne.io.read_raw_bdf(file_path, preload=True)

    # Get data as numpy array
    data = raw.get_data()

    return data


def load_hdf5_emg_data(file_path: str, dataset_name: str = 'emg_data') -> np.ndarray:
    """
    Load EMG data from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file
    dataset_name : str, default='emg_data'
        Name of the dataset containing EMG data

    Returns
    -------
    np.ndarray
        EMG data with shape (channels, samples)
    """
    if not HAS_H5PY:
        raise ImportError("h5py is not installed")

    # Load HDF5 file
    with h5py.File(file_path, 'r') as hdf5_file:
        # Get the dataset
        dataset = hdf5_file[dataset_name]

        # Read data
        data = np.array(dataset)

    return data


def load_emg_data(file_path: str, file_format: str = 'auto') -> np.ndarray:
    """
    Load EMG data from a file (EDF, BDF, or HDF5).

    Parameters
    ----------
    file_path : str
        Path to the data file
    file_format : str, default='auto'
        Format of the data file ('edf', 'bdf', 'hdf5', or 'auto' to detect)

    Returns
    -------
    np.ndarray
        EMG data with shape (channels, samples)
    """
    file_path = Path(file_path)

    # Auto-detect file format
    if file_format == 'auto':
        if file_path.suffix == '.edf':
            file_format = 'edf'
        elif file_path.suffix == '.bdf':
            file_format = 'bdf'
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            file_format = 'hdf5'
        else:
            raise ValueError(f"Unsupported file format for auto-detection: {file_path.suffix}")

    # Load data using the appropriate function
    if file_format == 'edf':
        return load_edf_emg_data(str(file_path))
    elif file_format == 'bdf':
        return load_bdf_emg_data(str(file_path))
    elif file_format == 'hdf5':
        return load_hdf5_emg_data(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


# ---------------- Phase 4 Additions: NWB / MNE Adapters & Unified Scaling ----------------

try:  # Optional imports
    import pynwb  # type: ignore
    HAS_PYNWB = True
except Exception:  # pragma: no cover
    HAS_PYNWB = False


def _scale_to_float32(arr: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Convert integer / other dtypes to float32 applying scale if provided.

    Heuristics:
      - int16/int32 assumed to be microvolts if scale not provided; scale inferred so that max abs -> ~500 (arbitrary normalization) else use given scale.
      - Already float: cast to float32 only.
    """
    if arr.dtype in (np.int16, np.int32, np.int8):
        if scale is None:
            max_abs = np.max(np.abs(arr)) or 1
            scale = 500.0 / max_abs
        arr = arr.astype(np.float32) * float(scale)
    else:
        arr = arr.astype(np.float32, copy=False)
    return arr


def iter_mmap_chunks(file_path: str, dataset: str = 'data', chunk_samples: int = 30_000, axis_channels_first: bool = True):
    """Iterate over large array stored in .npy/.npz/.h5 using memory mapping.

    Parameters
    ----------
    file_path : str
        Path to file (.npy, .npz, .h5/.hdf5)
    dataset : str
        Dataset key if HDF5/NPZ
    chunk_samples : int
        Number of samples per yielded chunk
    axis_channels_first : bool
        If True, data assumed (channels, samples) else (samples, channels)
    """
    p = Path(file_path)
    if p.suffix == '.npy':
        mm = np.load(p, mmap_mode='r')  # type: ignore
        data = mm
    elif p.suffix == '.npz':
        with np.load(p) as z:
            data = z[dataset]
    elif p.suffix in ('.h5', '.hdf5'):
        if not HAS_H5PY:
            raise ImportError('h5py required for HDF5 memory mapping')
        import h5py  # type: ignore
        h = h5py.File(p, 'r')
        data = h[dataset]
    else:
        raise ValueError(f"Unsupported file extension for mmap iterator: {p.suffix}")

    if not axis_channels_first:
        # transpose view lazily
        data_view = np.swapaxes(data, 0, 1)
    else:
        data_view = data

    n_channels, n_samples = data_view.shape
    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        chunk = np.array(data_view[:, start:end])  # materialize slice to ndarray
        yield chunk

    # Close file handle if HDF5
    if p.suffix in ('.h5', '.hdf5'):
        try:
            h.close()  # type: ignore[name-defined]
        except Exception:  # pragma: no cover
            pass


class NWBAdapter(FileDataAcquisition):
    """Adapter for NWB (Neurodata Without Borders) files using PyNWB.

    Loads ElectricalSeries data and streams in chunks with unified scaling.
    """

    def __init__(self, file_path: str, electrical_series: Optional[str] = None, buffer_duration: float = 1.0, loop_data: bool = False, playback_speed: float = 1.0):
        if not HAS_PYNWB:
            raise ImportError('pynwb not installed; install with extra [nwb]')
        self._nwb_file_path = file_path
        self._electrical_series_name = electrical_series
        with pynwb.NWBHDF5IO(file_path, 'r') as io:  # type: ignore
            nwbfile = io.read()
            es = None
            if electrical_series and electrical_series in nwbfile.acquisition:
                es = nwbfile.acquisition[electrical_series]
            else:
                # pick first ElectricalSeries
                for obj in nwbfile.acquisition.values():
                    if obj.__class__.__name__ == 'ElectricalSeries':
                        es = obj
                        break
            if es is None:
                raise ValueError('No ElectricalSeries found in NWB file')
            data_shape = es.data.shape  # (samples, channels) or (channels, samples)
            # Heuristic orientation: assume (samples, channels)
            _samples = data_shape[0]
            channels = data_shape[1] if len(data_shape) > 1 else 1
            sampling_rate = getattr(es, 'starting_rate', None) or getattr(es, 'rate', None) or 30_000.0
        super().__init__(file_path=file_path, sampling_rate=sampling_rate, buffer_duration=buffer_duration, loop_data=loop_data, playback_speed=playback_speed)
        self.n_channels = channels
        self.chunk_size = int(sampling_rate * 0.01)
        self._es_name = es.name if es else electrical_series

    def _load_data_file(self) -> np.ndarray:  # Override parent file load with on-demand reading
        # For NWB we defer full load; return an empty placeholder sized (channels, 0)
        return np.zeros((self.n_channels, 0), dtype=np.float32)

    def _acquire_data_chunk(self) -> Optional[np.ndarray]:
        if not HAS_PYNWB:
            return None
        with pynwb.NWBHDF5IO(self._nwb_file_path, 'r') as io:  # reopen each chunk (could optimize persistent handle)
            nwbfile = io.read()
            es = nwbfile.acquisition[self._es_name]
            # Determine current position using internal file_index from parent if exists
            if not hasattr(self, 'file_index'):
                self.file_index = 0  # type: ignore[attr-defined]
            start = self.file_index  # type: ignore[attr-defined]
            end = min(start + self.chunk_size, es.data.shape[0])
            if start >= es.data.shape[0]:
                if self.loop_data:
                    self.file_index = 0  # type: ignore[attr-defined]
                    return self._acquire_data_chunk()
                return None
            data = es.data[start:end, :]  # (samples, channels)
            self.file_index = end  # type: ignore[attr-defined]
            data = np.array(data).T  # -> (channels, samples)
            data = _scale_to_float32(data, scale=None)
            return data


class MNEAdapter(FileDataAcquisition):
    """Adapter for loading EEG/MEG/iEEG files via MNE Raw interface."""

    SUPPORTED_EXT = {'.fif', '.edf', '.bdf', '.gdf'}

    def __init__(self, file_path: str, preload: bool = False, buffer_duration: float = 1.0, loop_data: bool = False, playback_speed: float = 1.0):
        if not HAS_MNE:
            raise ImportError('mne not installed; install with extra [nwb]')
        self._raw = mne.io.read_raw(file_path, preload=preload)  # type: ignore
        sr = self._raw.info['sfreq']
        channels = len(self._raw.ch_names)
        super().__init__(file_path=file_path, sampling_rate=sr, buffer_duration=buffer_duration, loop_data=loop_data, playback_speed=playback_speed)
        self.n_channels = channels
        self.chunk_size = int(sr * 0.01)
        if preload:
            data = self._raw.get_data()  # (channels, samples)
            self.file_data = _scale_to_float32(np.array(data), scale=None)  # type: ignore

    def _load_data_file(self) -> np.ndarray:
        if hasattr(self, '_raw') and self._raw.preload:  # type: ignore[attr-defined]
            return self.file_data  # type: ignore
        # placeholder; data accessed on demand
        return np.zeros((self.n_channels, 0), dtype=np.float32)

    def _acquire_data_chunk(self) -> Optional[np.ndarray]:
        if not hasattr(self, 'file_index'):
            self.file_index = 0  # type: ignore[attr-defined]
        start = self.file_index  # type: ignore[attr-defined]
        end = min(start + self.chunk_size, int(self._raw.n_times))  # type: ignore[attr-defined]
        if start >= self._raw.n_times:  # type: ignore[attr-defined]
            if self.loop_data:
                self.file_index = 0  # type: ignore[attr-defined]
                return self._acquire_data_chunk()
            return None
        data, _ = self._raw[:, start:end]  # type: ignore[attr-defined]
        self.file_index = end  # type: ignore[attr-defined]
        data = _scale_to_float32(data, scale=None)
        return data


__all__ = [
    # existing exports implicitly provided above
    'NWBAdapter',
    'MNEAdapter',
    'iter_mmap_chunks',
]
