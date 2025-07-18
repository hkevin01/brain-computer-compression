"""
Hardware-Specific Optimizations for BCI Compression Toolkit

Provides utilities for device selection and hardware-aware profiling.

References:
- NVIDIA CUDA Toolkit
- PyTorch/TensorFlow device management
"""

from typing import Dict, Any


class HardwareOptimizer:
    """
    Utilities for hardware-specific optimizations.
    """
    @staticmethod
    def select_device(prefer_gpu: bool = True) -> str:
        """Select best available device (GPU/CPU)."""
        try:
            import cupy
            if prefer_gpu and cupy.cuda.runtime.getDeviceCount() > 0:
                return f"cuda:{cupy.cuda.Device().id}"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def profile_device() -> Dict[str, Any]:
        """Return device profiling info."""
        info = {}
        try:
            import cupy
            info['gpu_count'] = cupy.cuda.runtime.getDeviceCount()
            info['gpu_memory'] = cupy.cuda.Device().mem_info
        except ImportError:
            info['gpu_count'] = 0
            info['gpu_memory'] = None
        return info
