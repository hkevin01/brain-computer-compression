"""
PluginManager
============

Manages dynamic loading and unloading of compression plugins.

Usage:
    - Register plugins by class or module path.
    - Load/unload plugins at runtime for flexible algorithm selection.
    - Validate plugin interface compliance.
    - List available plugins and retrieve by name.

References:
    - CompressionPluginInterface
    - BCI compression toolkit plugin guidelines
"""
from typing import Dict, Type, Optional
import importlib
from .plugin_interface import CompressionPluginInterface

class PluginManager:
    """
    Manages dynamic loading/unloading of compression plugins.
    """
    def __init__(self) -> None:
        self._plugins: Dict[str, Type[CompressionPluginInterface]] = {}

    def register_plugin(self, plugin_cls: Type[CompressionPluginInterface]) -> None:
        name = plugin_cls().get_name()
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' already registered.")
        self._plugins[name] = plugin_cls

    def load_plugin(self, module_path: str, class_name: str) -> None:
        module = importlib.import_module(module_path)
        plugin_cls = getattr(module, class_name)
        if not issubclass(plugin_cls, CompressionPluginInterface):
            raise TypeError(f"{class_name} does not implement CompressionPluginInterface.")
        self.register_plugin(plugin_cls)

    def unload_plugin(self, name: str) -> None:
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not found.")
        del self._plugins[name]

    def get_plugin(self, name: str) -> Optional[Type[CompressionPluginInterface]]:
        return self._plugins.get(name)

    def list_plugins(self) -> Dict[str, Type[CompressionPluginInterface]]:
        return self._plugins.copy()
