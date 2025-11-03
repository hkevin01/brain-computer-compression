"""
Plugin/Entry-Point System for BCI Compression Toolkit

This module enables modular, extensible loading of algorithms and data formats.
Third-party and experimental algorithms can be registered as plugins.
Supports both explicit registration (decorator) and dynamic discovery via
Python entry points under group: 'bci_compression.compressors'.
"""

from __future__ import annotations

import importlib.metadata as md
import threading
from typing import Any, Callable, Dict, List, Type


# Base interface for plugins (algorithms, data formats, etc.)
class CompressorPlugin:  # pragma: no cover - interface
    """Base class for all compressor plugins."""
    name: str
    version: str = "0.0.0"

    def compress(self, data: Any, **kwargs) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def decompress(self, data: Any, **kwargs) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def fit(self, data: Any, **kwargs) -> None:  # pragma: no cover - optional
        pass

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:  # pragma: no cover - default
        return {}


# In-memory registry for explicitly registered plugins
PLUGIN_REGISTRY: Dict[str, Type[CompressorPlugin]] = {}
# Cache for entry-point discovered plugins (name -> object or load error placeholder)
_ENTRYPOINT_CACHE: Dict[str, Type[CompressorPlugin]] = {}
_DISCOVERY_DONE = False
_LOCK = threading.RLock()
_ENTRY_POINT_GROUP = 'bci_compression.compressors'


def register_plugin(name: str):
    """Decorator to register a compressor plugin by name.

    Explicit registration has precedence over entry-point discovery.
    """
    def decorator(cls: Type[CompressorPlugin]):
        with _LOCK:
            if name in PLUGIN_REGISTRY:
                import warnings
                warnings.warn(f"Plugin '{name}' is being re-registered.")
            PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator


def _discover_entrypoint_plugins(force: bool = False) -> None:
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE and not force:
        return
    with _LOCK:
        if _DISCOVERY_DONE and not force:
            return
        try:
            eps = md.entry_points()
            if hasattr(eps, 'select'):  # type: ignore[attr-defined]
                group_eps = list(eps.select(group=_ENTRY_POINT_GROUP))  # type: ignore
            else:  # pragma: no cover - legacy path
                group_eps = [e for e in eps.get(_ENTRY_POINT_GROUP, [])]  # type: ignore
            for ep in group_eps:
                name = ep.name
                if name in PLUGIN_REGISTRY or name in _ENTRYPOINT_CACHE:
                    continue
                try:
                    obj = ep.load()
                    if isinstance(obj, type):
                        if issubclass(obj, CompressorPlugin):
                            _ENTRYPOINT_CACHE[name] = obj
                        else:
                            Wrapped = type(
                                f"Wrapped{name}",
                                (CompressorPlugin,),
                                {
                                    'name': name,
                                    'compress': lambda self, data, **kw: obj().compress(data, **kw),  # type: ignore[attr-defined]
                                    'decompress': lambda self, data, **kw: obj().decompress(data, **kw),  # type: ignore[attr-defined]
                                },
                            )
                            _ENTRYPOINT_CACHE[name] = Wrapped
                    else:
                        factory = obj

                        class FactoryWrapper(CompressorPlugin):  # type: ignore[misc]
                            name = name
                            _factory: Callable = staticmethod(factory)

                            def __init__(self, **kw):
                                self._inst = self._factory(**kw)

                            def compress(self, data: Any, **kw) -> Any:  # noqa: D401
                                return self._inst.compress(data, **kw)

                            def decompress(self, data: Any, **kw) -> Any:  # noqa: D401
                                return self._inst.decompress(data, **kw)

                        _ENTRYPOINT_CACHE[name] = FactoryWrapper
                except Exception:  # pragma: no cover - defensive
                    continue
        finally:
            _DISCOVERY_DONE = True


def list_plugins(discover: bool = True) -> List[str]:
    if discover:
        _discover_entrypoint_plugins()
    with _LOCK:
        names = set(_ENTRYPOINT_CACHE.keys()) | set(PLUGIN_REGISTRY.keys())
    return sorted(names)


def get_plugin(name: str, discover: bool = True) -> Type[CompressorPlugin]:
    if discover:
        _discover_entrypoint_plugins()
    with _LOCK:
        if name in PLUGIN_REGISTRY:
            return PLUGIN_REGISTRY[name]
        if name in _ENTRYPOINT_CACHE:
            return _ENTRYPOINT_CACHE[name]
    raise KeyError(f"Plugin '{name}' not found. Available: {list_plugins()}")


def load_plugin(name: str, cls: Type[CompressorPlugin]) -> None:
    with _LOCK:
        if name in PLUGIN_REGISTRY or name in _ENTRYPOINT_CACHE:
            raise ValueError(f"Plugin '{name}' already registered.")
        PLUGIN_REGISTRY[name] = cls


def unload_plugin(name: str) -> None:
    with _LOCK:
        if name in PLUGIN_REGISTRY:
            del PLUGIN_REGISTRY[name]
        elif name in _ENTRYPOINT_CACHE:
            del _ENTRYPOINT_CACHE[name]
        else:
            raise KeyError(f"Plugin '{name}' not found.")


@register_plugin("dummy_lz")
class DummyLZCompressor(CompressorPlugin):  # pragma: no cover - trivial
    name = "dummy_lz"
    version = "1.0.0"

    def compress(self, data: Any, **kwargs) -> Any:
        return data  # No-op

    def decompress(self, data: Any, **kwargs) -> Any:
        return data


__all__ = [
    'CompressorPlugin', 'register_plugin', 'get_plugin', 'list_plugins',
    'load_plugin', 'unload_plugin'
]
