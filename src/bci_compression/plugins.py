"""
Plugin/Entry-Point System for BCI Compression Toolkit

This module enables modular, extensible loading of algorithms and data formats.
Third-party and experimental algorithms can be registered as plugins.
"""

from typing import Any, Callable, Dict, Type


# Base interface for plugins (algorithms, data formats, etc.)
class CompressorPlugin:
    """Base class for all compressor plugins."""
    name: str

    def compress(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError

    def decompress(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError

    def fit(self, data: Any, **kwargs) -> None:
        """Fit the compressor to the data. Optional."""
        pass


# Registry for plugins
PLUGIN_REGISTRY: Dict[str, Type[CompressorPlugin]] = {}


def register_plugin(name: str):
    """Decorator to register a compressor plugin by name."""
    def decorator(cls: Type[CompressorPlugin]):
        if name in PLUGIN_REGISTRY:
            raise ValueError(f"Plugin '{name}' already registered.")
        PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator


def get_plugin(name: str) -> Type[CompressorPlugin]:
    """Retrieve a registered plugin by name."""
    if name not in PLUGIN_REGISTRY:
        raise KeyError(f"Plugin '{name}' not found.")
    return PLUGIN_REGISTRY[name]


# Example: Register a dummy plugin (replace with real algorithm)
@register_plugin("dummy_lz")
class DummyLZCompressor(CompressorPlugin):
    name = "dummy_lz"

    def compress(self, data: Any, **kwargs) -> Any:
        return data  # No-op

    def decompress(self, data: Any, **kwargs) -> Any:
        return data

# TODO: Refactor real algorithms to register via this system
