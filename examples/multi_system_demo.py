#!/usr/bin/env python3
"""
Multi-BCI System Compression Demo

Demonstrates how to compress data from different BCI systems:
- OpenBCI (8 and 16 channels)
- Emotiv EPOC (14 channels)
- BioSemi ActiveTwo (64 channels)
- EGI GSN HydroCel (128 channels)
- Blackrock Cerebus (96 channels)
- Neuropixels (384 channels)

Each system has different:
- Channel counts
- Sampling rates
- Voltage ranges
- Recommended compression algorithms
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_compression.formats import (
    list_supported_systems,
    get_system_profile,
    adapt_data,
    StandardSystems
)
from bci_compression.algorithms.emg_compression import EMGLZCompressor


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_list_systems():
    """Demo: List all supported BCI systems."""
    print_header("Supported BCI Systems")
    
    systems = list_supported_systems()
    
    print(f"\nFound {len(systems)} supported systems:\n")
    
    for sys in systems:
        print(f"  {sys['name']}")
        print(f"    Manufacturer: {sys['manufacturer']}")
        print(f"    Channels: {sys['channels']}")
        print(f"    Sampling Rate: {sys['sampling_rate']} Hz")
        print(f"    Data Type: {sys['data_type']}")
        print(f"    Description: {sys['description']}")
        print()


def demo_system_profile(system_name):
    """Demo: Get detailed profile for a specific system."""
    print_header(f"System Profile: {system_name}")
    
    profile = get_system_profile(system_name)
    
    print(f"\n  Name: {profile.name}")
    print(f"  Channels: {profile.num_channels}")
    print(f"  Sampling Rate: {profile.sampling_rate} Hz")
    print(f"  Electrode Standard: {profile.electrode_standard}")
    print(f"  Bit Depth: {profile.bit_depth} bits")
    print(f"  Voltage Range: {profile.voltage_range[0]} to {profile.voltage_range[1]} µV")
    print(f"  Recommended Compression: {profile.recommended_compression}")
    print(f"  Data Type: {profile.data_type}")
    print(f"  Manufacturer: {profile.manufacturer}")


def demo_data_adaptation():
    """Demo: Adapt data between different systems."""
    print_header("Data Adaptation Example")
    
    # Simulate OpenBCI data (8 channels, 200 Hz)
    print("\n1. Source: OpenBCI Ganglion (8 channels, 200 Hz)")
    openbci_data = np.random.randn(8, 400).astype(np.float32)  # 2 seconds
    print(f"   Original shape: {openbci_data.shape}")
    print(f"   Duration: 2.0 seconds")
    
    # Adapt to 1000 Hz sampling rate
    print("\n2. Adapting to 1000 Hz...")
    adapted_data, settings = adapt_data(
        openbci_data,
        source_system='openbci_8',
        target_sampling_rate=1000
    )
    print(f"   Adapted shape: {adapted_data.shape}")
    print(f"   New sampling rate: {settings['sampling_rate']} Hz")
    print(f"   Recommended algorithm: {settings['algorithm']}")


def demo_compression_workflow():
    """Demo: Complete workflow with different systems."""
    print_header("Compression Workflow Example")
    
    systems = [
        ('openbci_8', 200),
        ('biosemi_64', 2048),
        ('gsn_128', 1000),
    ]
    
    compressor = EMGLZCompressor()
    
    for system_name, duration_samples in systems:
        profile = get_system_profile(system_name)
        
        print(f"\n{profile.name}:")
        print(f"  Channels: {profile.num_channels}")
        print(f"  Sampling Rate: {profile.sampling_rate} Hz")
        
        # Generate synthetic data (100ms worth)
        samples = int(profile.sampling_rate * 0.1)  # 100ms
        data = np.random.randn(profile.num_channels, samples).astype(np.float32)
        
        # Compress
        compressed = compressor.compress(data)
        stats = compressor.compression_stats
        
        original_size = data.nbytes
        compressed_size = len(compressed)
        ratio = stats['compression_ratio']
        
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {ratio:.2f}x")
        print(f"  Processing time: {stats['processing_time']*1000:.2f} ms")


def demo_high_density_systems():
    """Demo: High-density systems (128+ channels)."""
    print_header("High-Density BCI Systems")
    
    high_density = [
        StandardSystems.GSN_128,
        StandardSystems.NEUROPIXELS,
    ]
    
    for system in high_density:
        print(f"\n{system.name}:")
        print(f"  Channels: {system.num_channels}")
        print(f"  Sampling Rate: {system.sampling_rate:,} Hz")
        print(f"  Data Type: {system.data_type}")
        
        # Calculate data rate
        bytes_per_sample = system.bit_depth / 8
        data_rate = system.num_channels * system.sampling_rate * bytes_per_sample
        
        print(f"  Bit Depth: {system.bit_depth} bits")
        print(f"  Raw data rate: {data_rate / 1024 / 1024:.2f} MB/s")
        print(f"  Recommended: {system.recommended_compression}")


def demo_neural_recording_systems():
    """Demo: Neural recording systems for invasive BCI."""
    print_header("Neural Recording Systems")
    
    neural_systems = [
        StandardSystems.BLACKROCK_96,
        StandardSystems.INTAN_64,
        StandardSystems.NEUROPIXELS,
    ]
    
    print("\nInvasive BCI systems for neural spike recording:\n")
    
    for system in neural_systems:
        print(f"  {system.name} ({system.manufacturer}):")
        print(f"    {system.num_channels} channels @ {system.sampling_rate:,} Hz")
        print(f"    {system.description}")
        
        # Calculate bandwidth requirements
        bytes_per_sample = system.bit_depth / 8
        data_rate = system.num_channels * system.sampling_rate * bytes_per_sample
        daily_data = data_rate * 86400  # bytes per day
        
        print(f"    Uncompressed: {data_rate / 1024 / 1024:.1f} MB/s")
        print(f"    Daily data: {daily_data / 1024 / 1024 / 1024:.1f} GB/day")
        print()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  BCI MULTI-SYSTEM COMPRESSION DEMO")
    print("=" * 70)
    print("\nThis demo shows how to work with different BCI systems:")
    print("  - List supported systems")
    print("  - Get system profiles")
    print("  - Adapt data between systems")
    print("  - Compress data from various sources")
    
    try:
        # Run demos
        demo_list_systems()
        demo_system_profile('openbci_16')
        demo_data_adaptation()
        demo_compression_workflow()
        demo_high_density_systems()
        demo_neural_recording_systems()
        
        print_header("Demo Complete")
        print("\nAll demos completed successfully!")
        print("\nYou can now:")
        print("  1. Use get_system_profile() to load system configurations")
        print("  2. Use adapt_data() to convert between systems")
        print("  3. Apply appropriate compression algorithms")
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
