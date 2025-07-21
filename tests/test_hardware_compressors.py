import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import pytest

from bci_compression.algorithms.factory import create_compressor

# Check for CUDA availability
try:
    import cupy
    cupy.cuda.runtime.getDeviceCount()
    CUDA_AVAILABLE = True
except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
    CUDA_AVAILABLE = False


@pytest.mark.parametrize("algorithm", ["neon", "avx", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")), "fpga", "wasm"])
def test_hardware_compressor_roundtrip(algorithm):
    data = np.random.randn(100, 100).astype(np.float32)
    compressor = create_compressor(algorithm)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    decompressed = decompressed.reshape(data.shape)
    assert np.allclose(data, decompressed, atol=1e-6)
