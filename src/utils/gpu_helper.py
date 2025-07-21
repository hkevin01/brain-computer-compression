"""
GPU acceleration helper for BCI compression toolkit.
Handles CuPy integration and CPU fallback.

References:
- CuPy, NumPy
- GPU memory management
"""
from typing import Any
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # fallback
    GPU_AVAILABLE = False


def to_gpu(array: Any) -> Any:
    """
    Moves array to GPU if available.
    Args:
        array: NumPy array
    Returns:
        CuPy array or NumPy array
    """
    return cp.asarray(array)


def gpu_dot(a: Any, b: Any) -> Any:
    """
    Computes dot product on GPU if available.
    Args:
        a, b: Arrays
    Returns:
        Dot product result
    """
    return cp.dot(a, b)
