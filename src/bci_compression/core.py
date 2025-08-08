"""Core compression functionality with robust error handling and streaming support."""

from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union, TYPE_CHECKING
import logging
from dataclasses import dataclass, field

try:  # Optional dependency
    from pydantic import BaseModel, Field  # type: ignore
except Exception:  # Fallback lightweight shim
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    def Field(default=None, **_):  # type: ignore
        return default

if TYPE_CHECKING:  # pragma: no cover
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = Any  # type: ignore


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

class Config(BaseModel):
    """Base configuration for compressors.

    Provides:
    - Schema validation (via Pydantic if available)
    - Reproducible seeds
    - Unified serialization
    """

    seed: int = Field(42, description="Random seed for reproducibility")
    quality: float = Field(0.8, ge=0.0, le=1.0, description="Quality-control knob (0-1)")
    device: str = Field("auto", description="'cpu', 'cuda', or 'auto'")
    streaming_chunk: int = Field(1024, gt=0, description="Preferred streaming chunk size (samples)")
    latency_budget_ms: float = Field(2.0, gt=0, description="Latency budget per window (ms)")

    def dict_compact(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dict__.keys() if not k.startswith('_')}

# --------------------------------------------------------------------------------------
# Device Abstraction
# --------------------------------------------------------------------------------------

class Device:
    """Simple device abstraction bridging NumPy/Torch/CuPy.

    Attributes
    ---------
    backend : str
        One of 'numpy', 'torch'
    torch_device : Optional[str]
        'cpu' or 'cuda' if torch backend
    """

    def __init__(self, preference: str = "auto"):
        self.backend = 'numpy'
        self.torch_device: Optional[str] = None
        if preference in ("torch", "auto"):
            try:
                import torch  # noqa
                self.backend = 'torch'
                self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.backend = 'numpy'
        logger.debug(f"Device initialized: backend={self.backend}, torch_device={self.torch_device}")

    def to_array(self, x: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:  # type: ignore[name-defined]
        if self.backend == 'torch':
            try:  # runtime guard
                import torch  # type: ignore
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
        return x  # type: ignore

    def from_array(self, x: np.ndarray):
        if self.backend == 'torch':
            try:
                import torch  # type: ignore
                return torch.from_numpy(x).to(self.torch_device)
            except Exception:
                return x
        return x

# --------------------------------------------------------------------------------------
# Streaming Context
# --------------------------------------------------------------------------------------

@dataclass
class StreamContext:
    """Holds state for streaming compression/decompression."""
    buffer: List[np.ndarray] = field(default_factory=list)
    overlap: int = 0
    total_samples: int = 0
    chunk_count: int = 0

    def append(self, chunk: np.ndarray):
        self.buffer.append(chunk)
        self.total_samples += chunk.shape[-1]
        self.chunk_count += 1

    def assemble(self) -> np.ndarray:
        if not self.buffer:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(self.buffer, axis=-1)

# --------------------------------------------------------------------------------------
# Metrics Recorder
# --------------------------------------------------------------------------------------

@dataclass
class MetricsRecorder:
    """Lightweight metrics recorder for latency and quality KPIs."""
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, **metrics):
        self.records.append(metrics)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        summary: Dict[str, Any] = {}
        numeric_keys = [k for k in self.records[0].keys() if isinstance(self.records[0][k], (int, float))]
        for k in numeric_keys:
            vals = [r[k] for r in self.records if isinstance(r.get(k), (int, float))]
            if vals:
                summary[f"mean_{k}"] = float(np.mean(vals))
                summary[f"p95_{k}"] = float(np.percentile(vals, 95))
        summary['count'] = len(self.records)
        return summary

# --------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------

class CompressionError(Exception):
    """Base exception for compression errors"""
    pass

# --------------------------------------------------------------------------------------
# Base Compressor
# --------------------------------------------------------------------------------------

class BaseCompressor:
    """Base class for all compressors with error handling & standardized metadata."""

    def __init__(self, name: str = "base", config: Optional[Config] = None):
        self.name = name
        self._is_initialized = False
        self.config = config or Config()
        self.device = Device(self.config.device)
        self.stream_context: Optional[StreamContext] = None
        self.metrics = MetricsRecorder()

    # ------------------------- Public API -------------------------

    def fit(self, data: np.ndarray) -> None:
        """Optional training phase."""
        self._fit_impl(data)

    def compress(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with error handling and metadata standardization."""
        start = time.perf_counter()
        if not isinstance(data, np.ndarray):
            raise CompressionError(f"Expected numpy array, got {type(data)}")
        if data.size == 0:
            raise CompressionError("Cannot compress empty array")
        compressed, meta = self._compress_impl(data)
        latency_ms = (time.perf_counter() - start) * 1000
        meta_std = self._standard_metadata(data, compressed, latency_ms, extra=meta)
        self.metrics.record(latency_ms=latency_ms, compression_ratio=meta_std['compression_ratio'])
        return compressed, meta_std

    def decompress(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        start = time.perf_counter()
        if not compressed:
            raise CompressionError("Cannot decompress empty data")
        arr = self._decompress_impl(compressed, metadata)
        latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.record(decompression_latency_ms=latency_ms)
        return arr

    def stream_chunk(self, chunk: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Stream a single chunk (default: stateless fallback)."""
        return self.compress(chunk)

    # ------------------------- Metadata ---------------------------

    def _standard_metadata(self, original: np.ndarray, compressed: bytes, latency_ms: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        original_size = original.nbytes
        compressed_size = len(compressed)
        ratio = (original_size / compressed_size) if compressed_size > 0 else float('inf')
        meta = {
            'algorithm': self.name,
            'compression_ratio': ratio,
            'original_bytes': original_size,
            'compressed_bytes': compressed_size,
            'latency_ms': latency_ms,
            'quality': self.config.quality,
            'seed': self.config.seed,
        }
        if extra:
            meta.update(extra)
        return meta

    # ------------------------- Internal Hooks ---------------------

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        raise NotImplementedError

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _fit_impl(self, data: np.ndarray) -> None:  # Optional
        return None

# --------------------------------------------------------------------------------------
# Integrity Validation
# --------------------------------------------------------------------------------------

def validate_compression_integrity(
    original: np.ndarray,
    decompressed: np.ndarray,
    check_shape: bool = True,
    check_dtype: bool = True,
    check_hash: bool = False
) -> None:
    """Validate compression/decompression integrity."""
    import hashlib

    if check_shape and original.shape != decompressed.shape:
        raise ValueError(f"Decompressed data shape {decompressed.shape} does not match original {original.shape}")
    if check_dtype and original.dtype != decompressed.dtype:
        raise ValueError(f"Decompressed data dtype {decompressed.dtype} does not match original {original.dtype}")
    if check_hash:
        orig_hash = hashlib.sha256(original.tobytes()).hexdigest()
        decomp_hash = hashlib.sha256(decompressed.tobytes()).hexdigest()
        if orig_hash != decomp_hash:
            raise ValueError("Decompressed data hash does not match original (possible corruption)")

# --------------------------------------------------------------------------------------
# Plugin-Based Generic Wrapper (Backwards Compatibility)
# --------------------------------------------------------------------------------------

class Compressor(BaseCompressor):
    """Generic compressor using plugin system (legacy wrapper)."""

    def __init__(self, algorithm: str, **kwargs):
        from .plugins import get_plugin  # Local import to avoid cycles
        config = kwargs.pop('config', None)
        super().__init__(name=algorithm, config=config)
        self.algorithm_name = algorithm
        self.compressor_plugin_cls = get_plugin(self.algorithm_name)
        self.compressor_instance = self.compressor_plugin_cls(**kwargs)

    def _compress_impl(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        if hasattr(self.compressor_instance, 'compress'):
            result = self.compressor_instance.compress(data)
            if isinstance(result, tuple) and len(result) == 2:
                return result  # (bytes, meta)
            # Assume raw array-like
            arr = np.asarray(result)
            return arr.tobytes(), {}
        raise CompressionError("Plugin missing compress method")

    def _decompress_impl(self, compressed: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        if hasattr(self.compressor_instance, 'decompress'):
            return self.compressor_instance.decompress(compressed, metadata) if 'metadata' in self.compressor_instance.decompress.__code__.co_varnames else self.compressor_instance.decompress(compressed)
        raise CompressionError("Plugin missing decompress method")

# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

def create_compressor(algorithm: str, **kwargs) -> Compressor:
    return Compressor(algorithm, **kwargs)
