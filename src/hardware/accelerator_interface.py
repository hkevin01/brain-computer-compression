"""Hardware acceleration interface for BCI compression."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np


class AcceleratorType(Enum):
    """Supported hardware accelerator types."""
    CUDA = "cuda"
    FPGA = "fpga"
    NEUROMORPHIC = "neuromorphic"
    ARM_CORTEX = "arm_cortex"
    CPU = "cpu"

class PowerMode(Enum):
    """Power optimization modes."""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    ULTRA_LOW_POWER = "ultra_low_power"

class HardwareAccelerator(ABC):
    """Abstract base class for hardware accelerators."""

    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize the accelerator with given configuration."""
        pass

    @abstractmethod
    def process_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process neural signal with hardware acceleration."""
        pass

    @abstractmethod
    def set_power_mode(self, mode: PowerMode) -> None:
        """Set power optimization mode."""
        pass

    @abstractmethod
    def get_device_stats(self) -> Dict:
        """Get current device statistics."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release hardware resources."""
        pass

class CircularBuffer:
    """Zero-copy circular buffer for real-time data streaming."""

    def __init__(self, buffer_size: int, dtype=np.float32):
        """Initialize circular buffer.

        Args:
            buffer_size: Size of buffer in samples
            dtype: Data type for buffer
        """
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        self.size = buffer_size
        self.write_ptr = 0
        self.read_ptr = 0

    def write(self, data: np.ndarray) -> int:
        """Write data to buffer.

        Args:
            data: Input data array

        Returns:
            Number of samples written
        """
        available = self.size - (self.write_ptr - self.read_ptr)
        n_write = min(len(data), available)

        if n_write == 0:
            return 0

        write_idx = self.write_ptr % self.size
        if write_idx + n_write <= self.size:
            self.buffer[write_idx:write_idx + n_write] = data[:n_write]
        else:
            # Handle wrap-around
            first_part = self.size - write_idx
            self.buffer[write_idx:] = data[:first_part]
            self.buffer[:n_write - first_part] = data[first_part:n_write]

        self.write_ptr += n_write
        return n_write

    def read(self, n_samples: int) -> np.ndarray:
        """Read data from buffer.

        Args:
            n_samples: Number of samples to read

        Returns:
            Array of read samples
        """
        available = self.write_ptr - self.read_ptr
        n_read = min(n_samples, available)

        if n_read == 0:
            return np.array([], dtype=self.buffer.dtype)

        read_idx = self.read_ptr % self.size
        if read_idx + n_read <= self.size:
            data = self.buffer[read_idx:read_idx + n_read].copy()
        else:
            # Handle wrap-around
            first_part = self.size - read_idx
            data = np.empty(n_read, dtype=self.buffer.dtype)
            data[:first_part] = self.buffer[read_idx:]
            data[first_part:] = self.buffer[:n_read - first_part]

        self.read_ptr += n_read
        return data
