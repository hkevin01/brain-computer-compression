"""BCI Compression Toolkit - Neural data compression for brain-computer interfaces."""

__version__ = "0.8.0"
__author__ = "Kevin"
__description__ = "Neural data compression toolkit for brain-computer interfaces"

# Package imports
try:
    from .accel import AccelBackend
except ImportError:
    pass

try:
    from .core import CompressionResult
except ImportError:
    pass

__all__ = ["__version__", "__author__", "__description__"]
