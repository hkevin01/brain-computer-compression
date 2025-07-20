import time

import numpy as np

from bci_compression.algorithms.factory import create_compressor

algorithms = ["neon", "avx", "cuda", "fpga", "wasm"]
data = np.random.randn(1000, 1000).astype(np.float32)

for algo in algorithms:
    compressor = create_compressor(algo)
    start = time.time()
    compressed = compressor.compress(data)
    compress_time = time.time() - start

    start = time.time()
    decompressed = compressor.decompress(compressed)
    decompress_time = time.time() - start

    print(f"{algo.upper()} | Compress: {compress_time:.4f}s | Decompress: {decompress_time:.4f}s | Ratio: {compressor.get_compression_ratio():.2f}")
