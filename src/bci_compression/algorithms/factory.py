"""
Algorithm Factory Pattern for dynamic algorithm loading and management.

This module provides a registry-based system for creating and managing
compression algorithms dynamically, with support for lazy loading and
performance optimization.
"""

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from ..core import BaseCompressor

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """
    Registry for dynamic algorithm loading and management.

    This class provides a centralized registry for all compression algorithms,
    allowing dynamic loading and instantiation of algorithms by name.
    """

    def __init__(self):
        self._algorithms: Dict[str, Type[BaseCompressor]] = {}
        self._factories: Dict[str, Callable[..., BaseCompressor]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded_modules: Dict[str, bool] = {}

    def register(self, name: str, algorithm_class: Type[BaseCompressor],
                 factory_func: Optional[Callable[..., BaseCompressor]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an algorithm class and optional factory function.

        Parameters
        ----------
        name : str
            Unique name for the algorithm
        algorithm_class : Type[BaseCompressor]
            The algorithm class to register
        factory_func : Callable[..., BaseCompressor], optional
            Factory function for creating algorithm instances
        metadata : Dict[str, Any], optional
            Additional metadata about the algorithm
        """
        if name in self._algorithms:
            logger.warning(f"Algorithm '{name}' already registered, overwriting")

        self._algorithms[name] = algorithm_class
        if factory_func:
            self._factories[name] = factory_func
        else:
            # Create default factory function
            self._factories[name] = lambda **kwargs: algorithm_class(**kwargs)

        self._metadata[name] = metadata or {}
        logger.info(f"Registered algorithm: {name}")

    def create(self, name: str, **kwargs) -> BaseCompressor:
        """
        Create an algorithm instance by name.

        Parameters
        ----------
        name : str
            Name of the algorithm to create
        **kwargs
            Arguments to pass to the algorithm constructor

        Returns
        -------
        BaseCompressor
            Instance of the requested algorithm

        Raises
        ------
        KeyError
            If the algorithm is not registered
        """
        if name not in self._factories:
            raise KeyError(f"Algorithm '{name}' not registered. Available: {list(self._factories.keys())}")

        try:
            algorithm = self._factories[name](**kwargs)
            logger.debug(f"Created algorithm instance: {name}")
            return algorithm
        except Exception as e:
            logger.error(f"Failed to create algorithm '{name}': {e}")
            raise

    def list_available(self) -> List[str]:
        """
        List all available algorithms.

        Returns
        -------
        List[str]
            List of registered algorithm names
        """
        return list(self._algorithms.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific algorithm.

        Parameters
        ----------
        name : str
            Name of the algorithm

        Returns
        -------
        Dict[str, Any]
            Algorithm metadata
        """
        return self._metadata.get(name, {})

    def is_available(self, name: str) -> bool:
        """
        Check if an algorithm is available.

        Parameters
        ----------
        name : str
            Name of the algorithm to check

        Returns
        -------
        bool
            True if the algorithm is registered and available
        """
        return name in self._algorithms

    def lazy_load_module(self, module_name: str) -> bool:
        """
        Lazy load a module containing algorithms.

        Parameters
        ----------
        module_name : str
            Name of the module to load

        Returns
        -------
        bool
            True if module was loaded successfully
        """
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]

        try:
            importlib.import_module(module_name)
            self._loaded_modules[module_name] = True
            logger.info(f"Lazy loaded module: {module_name}")
            return True
        except ImportError as e:
            logger.warning(f"Failed to lazy load module '{module_name}': {e}")
            self._loaded_modules[module_name] = False
            return False


class UnifiedCompressor(BaseCompressor):
    """
    Unified interface for all compression algorithms.

    This class provides a consistent API for all compression algorithms,
    using the AlgorithmRegistry for dynamic algorithm creation.
    """

    def __init__(self, algorithm_name: str, registry: Optional[AlgorithmRegistry] = None, **kwargs):
        """
        Initialize unified compressor with specified algorithm.

        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm to use
        registry : AlgorithmRegistry, optional
            Algorithm registry to use (creates default if None)
        **kwargs
            Arguments to pass to the algorithm constructor
        """
        super().__init__()
        self.registry = registry or AlgorithmRegistry()
        self.algorithm_name = algorithm_name
        self.algorithm = self.registry.create(algorithm_name, **kwargs)
        self._last_compression_ratio = 0.0
        self._last_compression_time = 0.0

        # Copy algorithm attributes
        self.compression_ratio = self.algorithm.compression_ratio

    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress data using the selected algorithm.

        Parameters
        ----------
        data : np.ndarray
            Data to compress

        Returns
        -------
        bytes
            Compressed data
        """
        import time
        start_time = time.time()

        try:
            compressed = self.algorithm.compress(data)
            self._last_compression_time = time.time() - start_time
            self._last_compression_ratio = self.algorithm.get_compression_ratio()
            return compressed
        except Exception as e:
            logger.error(f"Compression failed with algorithm '{self.algorithm_name}': {e}")
            raise

    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress data using the selected algorithm.

        Parameters
        ----------
        compressed_data : bytes
            Compressed data to decompress

        Returns
        -------
        np.ndarray
            Decompressed data
        """
        try:
            return self.algorithm.decompress(compressed_data)
        except Exception as e:
            logger.error(f"Decompression failed with algorithm '{self.algorithm_name}': {e}")
            raise

    def get_compression_ratio(self) -> float:
        """Get the compression ratio of the last compression operation."""
        return self._last_compression_ratio

    def get_compression_time(self) -> float:
        """Get the compression time of the last compression operation."""
        return self._last_compression_time

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the current algorithm.

        Returns
        -------
        Dict[str, Any]
            Algorithm information and metadata
        """
        return {
            'name': self.algorithm_name,
            'metadata': self.registry.get_metadata(self.algorithm_name),
            'compression_ratio': self._last_compression_ratio,
            'compression_time': self._last_compression_time
        }


class PerformanceOptimizer:
    """
    Performance optimization with caching and memory pooling.

    This class provides caching mechanisms, lazy loading, and memory pooling
    for improved performance of compression operations.
    """

    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize performance optimizer.

        Parameters
        ----------
        max_cache_size : int
            Maximum number of cached items
        """
        self._cache: Dict[str, Any] = {}
        self._memory_pool: Dict[tuple, List[np.ndarray]] = {}
        self._max_cache_size = max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def cached_operation(self, key: str, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with caching.

        Parameters
        ----------
        key : str
            Cache key for the operation
        operation : Callable
            Operation to execute
        *args, **kwargs
            Arguments for the operation

        Returns
        -------
        Any
            Result of the operation (cached if available)
        """
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        result = operation(*args, **kwargs)

        # Simple LRU cache implementation
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest item (simple implementation)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result
        return result

    def get_memory_pool(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """
        Get array from memory pool or create new one.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : np.dtype
            Data type of the array

        Returns
        -------
        np.ndarray
            Array from pool or newly created
        """
        key = (shape, dtype)
        if key in self._memory_pool and self._memory_pool[key]:
            return self._memory_pool[key].pop()
        return np.empty(shape, dtype=dtype)

    def return_to_pool(self, array: np.ndarray) -> None:
        """
        Return array to memory pool.

        Parameters
        ----------
        array : np.ndarray
            Array to return to pool
        """
        key = (array.shape, array.dtype)
        if key not in self._memory_pool:
            self._memory_pool[key] = []
        self._memory_pool[key].append(array)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'memory_pool_size': sum(len(pool) for pool in self._memory_pool.values())
        }


# Global registry instance
_global_registry = AlgorithmRegistry()


def register_algorithm(name: str, algorithm_class: Type[BaseCompressor],
                      factory_func: Optional[Callable[..., BaseCompressor]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register an algorithm with the global registry.

    Parameters
    ----------
    name : str
        Unique name for the algorithm
    algorithm_class : Type[BaseCompressor]
        The algorithm class to register
    factory_func : Callable[..., BaseCompressor], optional
        Factory function for creating algorithm instances
    metadata : Dict[str, Any], optional
        Additional metadata about the algorithm
    """
    _global_registry.register(name, algorithm_class, factory_func, metadata)


def create_compressor(algorithm_name: str, **kwargs) -> BaseCompressor:
    """
    Create a compressor using the global registry.

    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm to create
    **kwargs
        Arguments to pass to the algorithm constructor

    Returns
    -------
    BaseCompressor
        Instance of the requested algorithm
    """
    return _global_registry.create(algorithm_name, **kwargs)


def list_available_algorithms() -> List[str]:
    """
    List all available algorithms in the global registry.

    Returns
    -------
    List[str]
        List of registered algorithm names
    """
    return _global_registry.list_available()


def get_algorithm_metadata(name: str) -> Dict[str, Any]:
    """
    Get metadata for a specific algorithm.

    Parameters
    ----------
    name : str
        Name of the algorithm

    Returns
    -------
    Dict[str, Any]
        Algorithm metadata
    """
    return _global_registry.get_metadata(name)


from .hardware.avx_kernels import avx_compress, avx_decompress
from .hardware.cuda_kernels import cuda_compress, cuda_decompress
from .hardware.fpga_pipeline import fpga_compress, fpga_decompress

# Hardware-optimized compressor stubs
from .hardware.neon_kernels import neon_compress, neon_decompress
from .hardware.wasm_interface import wasm_compress, wasm_decompress


class NeonCompressor(BaseCompressor):
    """Compressor using ARM NEON SIMD acceleration (stub)."""
    def compress(self, data: np.ndarray) -> bytes:
        arr = neon_compress(data)
        return arr.tobytes()
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        # Assume float32 for stub
        arr = np.frombuffer(compressed_data, dtype=np.float32)
        return neon_decompress(arr)

class AVXCompressor(BaseCompressor):
    """Compressor using Intel AVX/AVX2 acceleration (stub)."""
    def compress(self, data: np.ndarray) -> bytes:
        arr = avx_compress(data)
        return arr.tobytes()
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        arr = np.frombuffer(compressed_data, dtype=np.float32)
        return avx_decompress(arr)

class CudaCompressor(BaseCompressor):
    """Compressor using CUDA GPU acceleration (stub)."""
    def compress(self, data: np.ndarray) -> bytes:
        arr = cuda_compress(data)
        return arr.tobytes()
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        arr = np.frombuffer(compressed_data, dtype=np.float32)
        return cuda_decompress(arr)

class FpgaCompressor(BaseCompressor):
    """Compressor using FPGA acceleration (stub)."""
    def compress(self, data: np.ndarray) -> bytes:
        arr = fpga_compress(data)
        return arr.tobytes()
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        arr = np.frombuffer(compressed_data, dtype=np.float32)
        return fpga_decompress(arr)

class WasmCompressor(BaseCompressor):
    """Compressor using WebAssembly (WASM) acceleration (stub)."""
    def compress(self, data: np.ndarray) -> bytes:
        arr = wasm_compress(data)
        return arr.tobytes()
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        arr = np.frombuffer(compressed_data, dtype=np.float32)
        return wasm_decompress(arr)


def register_default_algorithms(registry: AlgorithmRegistry) -> None:
    """
    Register all default algorithms in the registry.

    Parameters
    ----------
    registry : AlgorithmRegistry
        Registry to register algorithms in
    """
    # Core algorithms
    try:
        from .lossless import AdaptiveLZCompressor, DictionaryCompressor
        registry.register("adaptive_lz", AdaptiveLZCompressor)
        registry.register("dictionary", DictionaryCompressor)
        logger.info("Registered lossless algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register lossless algorithms: {e}")

    # Lossy algorithms
    try:
        from .lossy import QuantizationCompressor, WaveletCompressor
        registry.register("quantization", QuantizationCompressor)
        registry.register("wavelet", WaveletCompressor)
        logger.info("Registered lossy algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register lossy algorithms: {e}")

    # Deep learning algorithms
    try:
        from .deep_learning import AutoencoderCompressor
        registry.register("autoencoder", AutoencoderCompressor)
        logger.info("Registered deep learning algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register deep learning algorithms: {e}")

    # Neural-optimized algorithms
    try:
        from .neural_lz import MultiChannelNeuralLZ, NeuralLZ77Compressor
        registry.register("neural_lz77", NeuralLZ77Compressor)
        registry.register("multi_channel_neural_lz", MultiChannelNeuralLZ)
        logger.info("Registered neural LZ algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register neural LZ algorithms: {e}")

    # Predictive algorithms
    try:
        from .predictive import MultiChannelPredictiveCompressor
        registry.register("predictive", MultiChannelPredictiveCompressor)
        logger.info("Registered predictive algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register predictive algorithms: {e}")

    # Context-aware algorithms
    try:
        from .context_aware import ContextAwareCompressor
        registry.register("context_aware", ContextAwareCompressor)
        logger.info("Registered context-aware algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register context-aware algorithms: {e}")

    # Phase 8: Advanced Neural Compression
    try:
        from .transformer_compression import (
            AdaptiveTransformerCompressor,
            TransformerCompressor,
        )
        registry.register("transformer", TransformerCompressor)
        registry.register("adaptive_transformer", AdaptiveTransformerCompressor)
        logger.info("Registered transformer algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register transformer algorithms: {e}")

    try:
        from .vae_compression import ConditionalVAECompressor, VAECompressor
        registry.register("vae", VAECompressor)
        registry.register("conditional_vae", ConditionalVAECompressor)
        logger.info("Registered VAE algorithms")
    except ImportError as e:
        logger.warning(f"Failed to register VAE algorithms: {e}")

    # Hardware-optimized algorithms
    registry.register("neon", NeonCompressor)
    registry.register("avx", AVXCompressor)
    registry.register("cuda", CudaCompressor)
    registry.register("fpga", FpgaCompressor)
    registry.register("wasm", WasmCompressor)
    logger.info("Registered hardware-optimized algorithms")


# Initialize global registry with default algorithms
register_default_algorithms(_global_registry)
