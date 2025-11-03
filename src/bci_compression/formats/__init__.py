"""
BCI System Format Adapters

Provides support for different BCI systems and standards:
- Different electrode layouts (10-20, 10-10, custom arrays)
- Various sampling rates (250Hz to 30kHz)
- Multiple recording systems (OpenBCI, Blackrock, Intan, etc.)
- Standard data formats (EDF, BDF, HDF5, NEV, NSx)
"""

from .system_profiles import (
    BCISystemProfile,
    get_system_profile,
    list_supported_systems,
    StandardSystems,
)
from .data_adapter import BCIDataAdapter, adapt_data

__all__ = [
    'BCISystemProfile',
    'get_system_profile',
    'list_supported_systems',
    'StandardSystems',
    'BCIDataAdapter',
    'adapt_data',
]
