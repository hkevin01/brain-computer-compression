"""
BCI System Profiles

Defines standard profiles for different BCI systems, including:
- Electrode layouts and channel counts
- Sampling rate specifications
- Data type requirements
- Compression recommendations
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ElectrodeStandard(str, Enum):
    """Standard electrode placement systems."""
    INTERNATIONAL_10_20 = "10-20"  # Standard 21-electrode system
    INTERNATIONAL_10_10 = "10-10"  # Extended 74-electrode system
    INTERNATIONAL_10_5 = "10-5"    # High-density 345-electrode system
    GSN_HYDRO_CEL = "GSN"          # EGI Geodesic Sensor Net
    BIOSEMI = "BioSemi"            # BioSemi electrode system
    CUSTOM = "custom"              # Custom electrode layout


@dataclass
class BCISystemProfile:
    """
    Profile for a specific BCI system configuration.
    
    Attributes
    ----------
    name : str
        System name
    num_channels : int
        Number of data channels
    sampling_rate : int
        Sampling rate in Hz
    electrode_standard : ElectrodeStandard
        Electrode placement standard
    bit_depth : int
        ADC bit depth (typically 12, 16, or 24)
    voltage_range : tuple
        Voltage range in microvolts (min, max)
    recommended_compression : str
        Recommended compression algorithm
    data_type : str
        Data type (EEG, EMG, ECoG, spikes, etc.)
    manufacturer : Optional[str]
        Device manufacturer
    description : str
        System description
    """
    name: str
    num_channels: int
    sampling_rate: int
    electrode_standard: ElectrodeStandard
    bit_depth: int
    voltage_range: tuple  # (min_uV, max_uV)
    recommended_compression: str
    data_type: str
    manufacturer: Optional[str] = None
    description: str = ""


class StandardSystems:
    """Pre-defined profiles for standard BCI systems."""
    
    # EEG Systems
    OPENBCI_8 = BCISystemProfile(
        name="OpenBCI Ganglion",
        num_channels=8,
        sampling_rate=200,
        electrode_standard=ElectrodeStandard.CUSTOM,
        bit_depth=12,
        voltage_range=(-200, 200),
        recommended_compression="emg_lz",
        data_type="EEG",
        manufacturer="OpenBCI",
        description="8-channel affordable EEG system"
    )
    
    OPENBCI_16 = BCISystemProfile(
        name="OpenBCI Cyton",
        num_channels=16,
        sampling_rate=250,
        electrode_standard=ElectrodeStandard.INTERNATIONAL_10_20,
        bit_depth=24,
        voltage_range=(-187500, 187500),
        recommended_compression="emg_lz",
        data_type="EEG",
        manufacturer="OpenBCI",
        description="16-channel research-grade EEG"
    )
    
    EMOTIV_14 = BCISystemProfile(
        name="Emotiv EPOC",
        num_channels=14,
        sampling_rate=128,
        electrode_standard=ElectrodeStandard.INTERNATIONAL_10_20,
        bit_depth=14,
        voltage_range=(-260, 260),
        recommended_compression="emg_lz",
        data_type="EEG",
        manufacturer="Emotiv",
        description="Consumer EEG headset"
    )
    
    BIOSEMI_64 = BCISystemProfile(
        name="BioSemi ActiveTwo 64",
        num_channels=64,
        sampling_rate=2048,
        electrode_standard=ElectrodeStandard.BIOSEMI,
        bit_depth=24,
        voltage_range=(-262144, 262144),
        recommended_compression="neural_lz",
        data_type="EEG",
        manufacturer="BioSemi",
        description="High-density research EEG"
    )
    
    GSN_128 = BCISystemProfile(
        name="EGI GSN HydroCel 128",
        num_channels=128,
        sampling_rate=1000,
        electrode_standard=ElectrodeStandard.GSN_HYDRO_CEL,
        bit_depth=16,
        voltage_range=(-32768, 32767),
        recommended_compression="neural_lz",
        data_type="EEG",
        manufacturer="Electrical Geodesics Inc",
        description="High-density Geodesic Sensor Net"
    )
    
    # EMG Systems
    DELSYS_TRIGNO = BCISystemProfile(
        name="Delsys Trigno",
        num_channels=16,
        sampling_rate=2000,
        electrode_standard=ElectrodeStandard.CUSTOM,
        bit_depth=16,
        voltage_range=(-11000, 11000),
        recommended_compression="emg_lz",
        data_type="EMG",
        manufacturer="Delsys",
        description="Wireless EMG system"
    )
    
    # Neural Recording Systems
    BLACKROCK_96 = BCISystemProfile(
        name="Blackrock Cerebus",
        num_channels=96,
        sampling_rate=30000,
        electrode_standard=ElectrodeStandard.CUSTOM,
        bit_depth=16,
        voltage_range=(-8192, 8191),
        recommended_compression="neural_lz",
        data_type="spikes",
        manufacturer="Blackrock Neurotech",
        description="Utah array neural recording"
    )
    
    INTAN_64 = BCISystemProfile(
        name="Intan RHD2000",
        num_channels=64,
        sampling_rate=20000,
        electrode_standard=ElectrodeStandard.CUSTOM,
        bit_depth=16,
        voltage_range=(-5000, 5000),
        recommended_compression="neural_lz",
        data_type="neural",
        manufacturer="Intan Technologies",
        description="Electrophysiology recording"
    )
    
    NEUROPIXELS = BCISystemProfile(
        name="Neuropixels 1.0",
        num_channels=384,
        sampling_rate=30000,
        electrode_standard=ElectrodeStandard.CUSTOM,
        bit_depth=10,
        voltage_range=(-512, 511),
        recommended_compression="neural_lz",
        data_type="spikes",
        manufacturer="IMEC",
        description="High-density neural probe"
    )


def get_system_profile(system_name: str) -> BCISystemProfile:
    """
    Get a pre-defined system profile by name.
    
    Parameters
    ----------
    system_name : str
        Name of the system (case-insensitive)
        
    Returns
    -------
    BCISystemProfile
        System profile
        
    Raises
    ------
    ValueError
        If system name is not found
    """
    system_map = {
        'openbci_8': StandardSystems.OPENBCI_8,
        'openbci_ganglion': StandardSystems.OPENBCI_8,
        'openbci_16': StandardSystems.OPENBCI_16,
        'openbci_cyton': StandardSystems.OPENBCI_16,
        'emotiv': StandardSystems.EMOTIV_14,
        'emotiv_epoc': StandardSystems.EMOTIV_14,
        'biosemi_64': StandardSystems.BIOSEMI_64,
        'gsn_128': StandardSystems.GSN_128,
        'egi_128': StandardSystems.GSN_128,
        'delsys': StandardSystems.DELSYS_TRIGNO,
        'blackrock': StandardSystems.BLACKROCK_96,
        'blackrock_96': StandardSystems.BLACKROCK_96,
        'intan': StandardSystems.INTAN_64,
        'intan_64': StandardSystems.INTAN_64,
        'neuropixels': StandardSystems.NEUROPIXELS,
    }
    
    key = system_name.lower().replace(' ', '_')
    if key not in system_map:
        available = ', '.join(sorted(set(system_map.keys())))
        raise ValueError(
            f"Unknown system '{system_name}'. "
            f"Available systems: {available}"
        )
    
    return system_map[key]


def list_supported_systems() -> List[Dict[str, Any]]:
    """
    List all supported BCI systems.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of system information dictionaries
    """
    systems = [
        StandardSystems.OPENBCI_8,
        StandardSystems.OPENBCI_16,
        StandardSystems.EMOTIV_14,
        StandardSystems.BIOSEMI_64,
        StandardSystems.GSN_128,
        StandardSystems.DELSYS_TRIGNO,
        StandardSystems.BLACKROCK_96,
        StandardSystems.INTAN_64,
        StandardSystems.NEUROPIXELS,
    ]
    
    return [
        {
            'name': sys.name,
            'channels': sys.num_channels,
            'sampling_rate': sys.sampling_rate,
            'data_type': sys.data_type,
            'manufacturer': sys.manufacturer,
            'description': sys.description,
        }
        for sys in systems
    ]
