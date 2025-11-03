"""
OpenBCI Adapter Demo

Demonstrates how to use the BCI device adapters to convert OpenBCI data
to a standardized format and apply compression algorithms.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bci_compression.adapters.openbci import (
    OpenBCIAdapter,
    convert_openbci_to_standard,
)
from bci_compression.adapters import (
    apply_channel_groups,
    apply_calibration,
)
from bci_compression.algorithms.lossless import NeuralLZ77Compressor


def generate_synthetic_openbci_data(n_channels=8, n_samples=5000, sampling_rate=250):
    """Generate synthetic OpenBCI-like neural data."""
    print(f"Generating synthetic OpenBCI data: {n_channels} channels, {n_samples} samples @ {sampling_rate}Hz")

    # Generate multi-frequency neural-like signals
    t = np.arange(n_samples) / sampling_rate
    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Mix of different frequency components
        alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        noise = np.random.randn(n_samples) * 0.1

        data[ch] = alpha + 0.5 * beta + noise

    return data


def demo_basic_conversion():
    """Demo 1: Basic OpenBCI data conversion."""
    print("\n" + "="*70)
    print("DEMO 1: Basic OpenBCI Data Conversion")
    print("="*70)

    # Generate synthetic Cyton data
    raw_data = generate_synthetic_openbci_data(n_channels=8, n_samples=5000)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data size: {raw_data.nbytes / 1024:.2f} KB")

    # Convert using quick converter
    standard_data = convert_openbci_to_standard(raw_data, device='cyton_8ch')
    print(f"Standardized data shape: {standard_data.shape}")

    # Show channel mapping
    print("\nChannel Mapping (OpenBCI Cyton -> 10-20 system):")
    from bci_compression.adapters.openbci import OPENBCI_CYTON_8CH_MAPPING
    for ch_idx, electrode in OPENBCI_CYTON_8CH_MAPPING['mapping'].items():
        print(f"  {ch_idx} -> {electrode}")


def demo_resampling():
    """Demo 2: Data conversion with resampling."""
    print("\n" + "="*70)
    print("DEMO 2: Data Conversion with Resampling")
    print("="*70)

    # Generate data at 250 Hz
    raw_data = generate_synthetic_openbci_data(n_channels=8, n_samples=5000)
    print(f"Original sampling rate: 250 Hz")
    print(f"Original duration: {raw_data.shape[1] / 250:.2f} seconds")

    # Resample to 1000 Hz
    standard_data = convert_openbci_to_standard(
        raw_data,
        device='cyton_8ch',
        target_rate=1000
    )
    print(f"\nResampled to: 1000 Hz")
    print(f"New shape: {standard_data.shape}")
    print(f"Duration preserved: {standard_data.shape[1] / 1000:.2f} seconds")


def demo_channel_grouping():
    """Demo 3: Channel grouping for spatial filtering."""
    print("\n" + "="*70)
    print("DEMO 3: Channel Grouping for Spatial Filtering")
    print("="*70)

    # Generate data
    raw_data = generate_synthetic_openbci_data(n_channels=8, n_samples=5000)

    # Create adapter
    adapter = OpenBCIAdapter(device='cyton_8ch')
    standard_data = adapter.convert(raw_data)

    # Get channel groups
    groups = adapter.get_channel_groups()
    print(f"Channel groups available: {list(groups.keys())}")

    # Apply grouping
    grouped_data = apply_channel_groups(standard_data, groups, reducer='mean')

    print("\nGrouped data:")
    for group_name, group_data in grouped_data.items():
        print(f"  {group_name}: shape {group_data.shape}")
        print(f"    Channels: {groups[group_name]}")


def demo_calibration():
    """Demo 4: Applying calibration to data."""
    print("\n" + "="*70)
    print("DEMO 4: Applying Calibration")
    print("="*70)

    # Generate data
    raw_data = generate_synthetic_openbci_data(n_channels=8, n_samples=5000)

    # Define calibration parameters
    calibration = {
        'scale': [1.0, 1.2, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0],  # per-channel gain
        'offset': [0.0, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],  # per-channel DC offset
        'bad_channels': [5],  # Mark channel 5 as bad
    }

    print("Calibration parameters:")
    print(f"  Scale factors: {calibration['scale']}")
    print(f"  DC offsets: {calibration['offset']}")
    print(f"  Bad channels: {calibration['bad_channels']}")

    # Apply calibration
    calibrated_data = apply_calibration(raw_data, calibration)

    print(f"\nOriginal data mean (ch 5): {raw_data[5].mean():.4f}")
    print(f"Calibrated data mean (ch 5): {calibrated_data[5].mean():.4f} (should be ~0)")


def demo_compression_pipeline():
    """Demo 5: Full pipeline with compression."""
    print("\n" + "="*70)
    print("DEMO 5: Full Pipeline - Convert, Process, and Compress")
    print("="*70)

    # Generate data
    raw_data = generate_synthetic_openbci_data(n_channels=8, n_samples=10000)
    original_size = raw_data.nbytes
    print(f"Original size: {original_size / 1024:.2f} KB")

    # Step 1: Convert to standard format
    adapter = OpenBCIAdapter(device='cyton_8ch')
    standard_data = adapter.convert(raw_data)

    # Step 2: Resample if needed
    resampled_data = adapter.resample_to(standard_data, target_rate=1000)
    print(f"After resampling: {resampled_data.shape}")

    # Step 3: Apply calibration
    calibration = {
        'scale': [1.0] * 8,
        'offset': [0.0] * 8,
        'bad_channels': [],
    }
    calibrated_data = apply_calibration(resampled_data, calibration)

    # Step 4: Compress
    compressor = NeuralLZ77Compressor()
    compressed = compressor.compress(calibrated_data)
    compressed_size = len(compressed)

    print(f"Compressed size: {compressed_size / 1024:.2f} KB")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")

    # Step 5: Decompress and verify
    decompressed = compressor.decompress(compressed)
    decompressed = decompressed.reshape(calibrated_data.shape)

    # Check reconstruction error
    error = np.max(np.abs(calibrated_data - decompressed))
    print(f"Max reconstruction error: {error:.2e} (lossless)")


def demo_multi_device():
    """Demo 6: Working with multiple device types."""
    print("\n" + "="*70)
    print("DEMO 6: Multiple Device Support")
    print("="*70)

    # Cyton 8-channel
    adapter_8ch = OpenBCIAdapter(device='cyton_8ch')
    data_8ch = generate_synthetic_openbci_data(n_channels=8, n_samples=1000)
    converted_8ch = adapter_8ch.convert(data_8ch)
    print(f"Cyton 8-ch: {data_8ch.shape} -> {converted_8ch.shape}")

    # Daisy 16-channel
    adapter_16ch = OpenBCIAdapter(device='daisy_16ch')
    data_16ch = generate_synthetic_openbci_data(n_channels=16, n_samples=1000)
    converted_16ch = adapter_16ch.convert(data_16ch)
    print(f"Daisy 16-ch: {data_16ch.shape} -> {converted_16ch.shape}")

    # Show channel groups for each
    print("\nCyton groups:", list(adapter_8ch.get_channel_groups().keys()))
    print("Daisy groups:", list(adapter_16ch.get_channel_groups().keys()))


def main():
    """Run all demos."""
    print("="*70)
    print("OpenBCI Adapter Demonstration")
    print("="*70)
    print("\nThis demo shows how to use BCI device adapters to:")
    print("  1. Convert device-specific data to standardized formats")
    print("  2. Resample data to different sampling rates")
    print("  3. Group channels for spatial filtering")
    print("  4. Apply per-channel calibration")
    print("  5. Build complete compression pipelines")
    print("  6. Support multiple device types")

    try:
        demo_basic_conversion()
        demo_resampling()
        demo_channel_grouping()
        demo_calibration()
        demo_compression_pipeline()
        demo_multi_device()

        print("\n" + "="*70)
        print("All demos completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
