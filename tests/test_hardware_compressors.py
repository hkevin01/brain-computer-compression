import numpy as np
import pytest

from bci_compression.algorithms.factory import create_compressor


@pytest.mark.parametrize("algorithm", ["neon", "avx", "cuda", "fpga", "wasm"])
def test_hardware_compressor_roundtrip(algorithm):
    data = np.random.randn(100, 100).astype(np.float32)
    compressor = create_compressor(algorithm)
    compressed = compressor.compress(data)
    decompressed = compressor.decompress(compressed)
    # Since stubs are identity, output should match input
    np.testing.assert_allclose(decompressed.flatten()[:data.size], data.flatten(), atol=1e-6)
