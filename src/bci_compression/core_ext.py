"""
Core extension: lossless compression algorithms for neural data streams.
Stub implementations for plugin registration and test compatibility.
"""

import numpy as np
import zlib
import lzma
import pickle
from typing import Any
from collections import Counter
import heapq
from bci_compression.plugins import CompressorPlugin, register_plugin


@register_plugin("adaptive_lz")
class AdaptiveLZCompressor(CompressorPlugin):
    """Adaptive LZ77 using zlib for lossless neural data compression."""
    name = "adaptive_lz"

    def compress(self, data: np.ndarray, **kwargs) -> bytes:
        arr_bytes = data.astype(np.float32).tobytes()
        return zlib.compress(arr_bytes, level=9)

    def decompress(self, compressed: bytes, **kwargs) -> np.ndarray:
        arr_bytes = zlib.decompress(compressed)
        return np.frombuffer(arr_bytes, dtype=np.float32)


@register_plugin("dictionary")
class DictionaryCompressor(CompressorPlugin):
    """Dictionary-based compression using lzma for neural data."""
    name = "dictionary"

    def compress(self, data: np.ndarray, **kwargs) -> bytes:
        arr_bytes = data.astype(np.float32).tobytes()
        return lzma.compress(arr_bytes)

    def decompress(self, compressed: bytes, **kwargs) -> np.ndarray:
        arr_bytes = lzma.decompress(compressed)
        return np.frombuffer(arr_bytes, dtype=np.float32)


@register_plugin("huffman")
class HuffmanCompressor(CompressorPlugin):
    """Huffman coding for neural data (using pickle for demonstration)."""
    name = "huffman"

    def compress(self, data: np.ndarray, **kwargs) -> bytes:
        arr_bytes = data.astype(np.float32).tobytes()
        # Simple frequency-based encoding (not optimal, for demonstration)
        freq = Counter(arr_bytes)
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huff_dict = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
        encoded = b''.join(huff_dict[b].encode() for b in arr_bytes)
        return pickle.dumps((encoded, huff_dict, data.shape))

    def decompress(self, compressed: bytes, **kwargs) -> np.ndarray:
        encoded, huff_dict, shape = pickle.loads(compressed)
        # Reverse dict
        rev_dict = {v: k for k, v in huff_dict.items()}
        # Decode (not optimal, for demonstration)
        decoded_bytes = bytearray()
        code = ""
        for bit in encoded.decode():
            code += bit
            if code in rev_dict:
                decoded_bytes.append(rev_dict[code])
                code = ""
        arr = np.frombuffer(decoded_bytes, dtype=np.float32)
        return arr.reshape(shape)


@register_plugin("lz77")
class LZ77Compressor(CompressorPlugin):
    """LZ77 using zlib for neural data compression."""
    name = "lz77"

    def compress(self, data: np.ndarray, **kwargs) -> bytes:
        arr_bytes = data.astype(np.float32).tobytes()
        return zlib.compress(arr_bytes, level=6)

    def decompress(self, compressed: bytes, **kwargs) -> np.ndarray:
        arr_bytes = zlib.decompress(compressed)
        return np.frombuffer(arr_bytes, dtype=np.float32)


def lz77_compress_core(data: np.ndarray, **kwargs: Any) -> bytes:
    """Core LZ77 compression using zlib."""
    arr_bytes = data.astype(np.float32).tobytes()
    return zlib.compress(arr_bytes, level=6)


def lz77_decompress_core(compressed: bytes, **kwargs: Any) -> np.ndarray:
    """Core LZ77 decompression using zlib."""
    arr_bytes = zlib.decompress(compressed)
    return np.frombuffer(arr_bytes, dtype=np.float32)

