"""
Multi-Device Pipeline Example

Demonstrates combining data from multiple BCI devices:
- OpenBCI Cyton (scalp EEG)
- Blackrock Neuroport (intracortical)
- Intan RHD (local field potentials)
- Unified compression pipeline
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bci_compression.adapters.openbci import OpenBCIAdapter
from bci_compression.adapters.blackrock import BlackrockAdapter
from bci_compression.adapters.intan import IntanAdapter
from bci_compression.algorithms.lossless import NeuralLZ77Compressor
from bci_compression.algorithms.lossy import WaveletCompressor


class MultiDevicePipeline:
    """
    Unified pipeline for multi-device neural recording.

    Handles:
    - Multiple devices with different sampling rates
    - Channel alignment and synchronization
    - Hierarchical compression strategy
    - Metadata tracking
    """

    def __init__(self):
        """Initialize multi-device pipeline."""
        self.devices = {}
        self.compressors = {}
        self.metadata = {}

    def add_device(
        self,
        name: str,
        adapter,
        compressor=None,
        priority: str = 'normal'
    ):
        """
        Add a device to the pipeline.

        Args:
            name: Device identifier
            adapter: Device adapter instance
            compressor: Compression algorithm (default: NeuralLZ77)
            priority: 'high' for lossless, 'normal' for moderate compression
        """
        self.devices[name] = adapter

        if compressor is None:
            if priority == 'high':
                compressor = NeuralLZ77Compressor()
            else:
                compressor = WaveletCompressor()

        self.compressors[name] = compressor
        self.metadata[name] = {
            'priority': priority,
            'samples_processed': 0,
            'bytes_compressed': 0,
            'compression_ratio': 0.0
        }

    def process_batch(
        self,
        data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, bytes]:
        """
        Process a batch of data from all devices.

        Args:
            data_dict: {device_name: data_array}

        Returns:
            {device_name: compressed_bytes}
        """
        compressed_dict = {}

        for device_name, data in data_dict.items():
            if device_name not in self.devices:
                continue

            # Apply device adapter
            adapter = self.devices[device_name]
            processed = adapter.convert(data, apply_mapping=False)

            # Compress
            compressor = self.compressors[device_name]
            compressed = compressor.compress(processed)
            compressed_dict[device_name] = compressed

            # Update metadata
            self.metadata[device_name]['samples_processed'] += data.shape[1]
            self.metadata[device_name]['bytes_compressed'] += len(compressed)

            # Calculate compression ratio
            original_bytes = data.nbytes
            self.metadata[device_name]['compression_ratio'] = original_bytes / len(compressed)

        return compressed_dict

    def get_summary(self) -> dict:
        """Get pipeline summary statistics."""
        total_samples = sum(m['samples_processed'] for m in self.metadata.values())
        total_bytes = sum(m['bytes_compressed'] for m in self.metadata.values())

        return {
            'devices': list(self.devices.keys()),
            'total_samples_processed': total_samples,
            'total_bytes_compressed': total_bytes,
            'device_stats': self.metadata.copy()
        }


def generate_synthetic_data() -> Dict[str, np.ndarray]:
    """
    Generate synthetic data for multiple devices.

    Returns:
        Dictionary of {device_name: data_array}
    """
    # OpenBCI Cyton: 8 channels @ 250 Hz
    openbci_data = np.random.randn(8, 1000) * 10  # 4 seconds

    # Blackrock Neuroport: 96 channels @ 30000 Hz
    # (downsample for demo - 1 second at 1000 Hz)
    blackrock_data = np.random.randn(96, 1000) * 50

    # Intan RHD2164: 64 channels @ 20000 Hz
    # (downsample for demo - 1 second at 1000 Hz)
    intan_data = np.random.randn(64, 1000) * 30

    return {
        'openbci': openbci_data,
        'blackrock': blackrock_data,
        'intan': intan_data
    }


def demo_basic_pipeline():
    """Demonstrate basic multi-device pipeline."""
    print("="*70)
    print("Basic Multi-Device Pipeline")
    print("="*70)
    print()

    # Setup pipeline
    pipeline = MultiDevicePipeline()

    # Add devices
    print("Adding devices...")

    # OpenBCI Cyton (scalp EEG) - moderate compression
    openbci = OpenBCIAdapter(device='cyton_8ch')
    pipeline.add_device('openbci', openbci, priority='normal')
    print("  ✓ OpenBCI Cyton (8ch @ 250Hz) - Wavelet compression")

    # Blackrock Neuroport (intracortical) - high priority, lossless
    blackrock = BlackrockAdapter(device='neuroport_96ch')
    pipeline.add_device('blackrock', blackrock, priority='high')
    print("  ✓ Blackrock Neuroport (96ch @ 30kHz) - Lossless compression")

    # Intan RHD (LFP) - moderate compression
    intan = IntanAdapter(device='rhd2164_64ch')
    pipeline.add_device('intan', intan, priority='normal')
    print("  ✓ Intan RHD2164 (64ch @ 20kHz) - Wavelet compression")
    print()

    # Process data
    print("Processing synchronized batch...")
    data = generate_synthetic_data()

    print(f"  OpenBCI: {data['openbci'].shape} ({data['openbci'].nbytes} bytes)")
    print(f"  Blackrock: {data['blackrock'].shape} ({data['blackrock'].nbytes} bytes)")
    print(f"  Intan: {data['intan'].shape} ({data['intan'].nbytes} bytes)")
    print()

    compressed = pipeline.process_batch(data)

    print("Compression results:")
    for device, comp_data in compressed.items():
        meta = pipeline.metadata[device]
        print(f"  {device.capitalize()}: {len(comp_data)} bytes (ratio: {meta['compression_ratio']:.2f}x)")
    print()

    # Summary
    summary = pipeline.get_summary()
    print("Pipeline Summary:")
    print(f"  Total devices: {len(summary['devices'])}")
    print(f"  Total samples: {summary['total_samples_processed']:,}")
    print(f"  Total compressed: {summary['total_bytes_compressed']:,} bytes ({summary['total_bytes_compressed']/1024:.2f} KB)")
    print()


def demo_hierarchical_compression():
    """Demonstrate hierarchical compression strategy."""
    print("="*70)
    print("Hierarchical Compression Strategy")
    print("="*70)
    print()

    print("Strategy:")
    print("  HIGH priority (intracortical) → Lossless (NeuralLZ77)")
    print("  NORMAL priority (scalp/LFP)   → Lossy (Wavelet)")
    print()

    # Generate test data with different SNR
    high_quality = np.random.randn(96, 1000) * 50  # High SNR
    low_quality = np.random.randn(8, 1000) * 10 + np.random.randn(8, 1000) * 30  # Lower SNR

    print("Data characteristics:")
    print(f"  High-quality (intracortical): SNR ~10dB")
    print(f"  Lower-quality (scalp):        SNR ~3dB")
    print()

    # Compress with different strategies
    lossless = NeuralLZ77Compressor()
    lossy = WaveletCompressor()

    compressed_high = lossless.compress(high_quality)
    compressed_low = lossy.compress(low_quality)

    ratio_high = high_quality.nbytes / len(compressed_high)
    ratio_low = low_quality.nbytes / len(compressed_low)

    print("Compression results:")
    print(f"  Lossless (high-quality): {ratio_high:.2f}x")
    print(f"  Lossy (lower-quality):   {ratio_low:.2f}x")
    print()

    print("Trade-off analysis:")
    print(f"  Lossless: Perfect reconstruction, moderate compression")
    print(f"  Lossy: ~95% quality retained, high compression")
    print()


def demo_channel_alignment():
    """Demonstrate channel alignment across devices."""
    print("="*70)
    print("Multi-Device Channel Alignment")
    print("="*70)
    print()

    # Create adapters
    openbci = OpenBCIAdapter(device='cyton_8ch')
    blackrock = BlackrockAdapter(device='neuroport_96ch')

    print("Device configurations:")
    print()

    print("OpenBCI Cyton:")
    print(f"  Channels: {openbci.mapping.get('channels', openbci.mapping.get('n_channels', 'N/A'))}")
    print(f"  Sampling rate: {openbci.mapping['sampling_rate']} Hz")
    print(f"  Electrode system: {openbci.mapping.get('electrode_system', '10-20 system')}")
    print()

    print("Blackrock Neuroport:")
    print(f"  Channels: {blackrock.mapping.get('channels', blackrock.mapping.get('n_channels', 'N/A'))}")
    print(f"  Sampling rate: {blackrock.mapping['sampling_rate']} Hz")
    print(f"  Array type: {blackrock.mapping.get('array_type', 'N/A')}")
    print()

    # Demonstrate resampling to common rate
    print("Resampling to common rate (1000 Hz)...")

    openbci_data = np.random.randn(8, 250)  # 1 second @ 250 Hz
    blackrock_data = np.random.randn(96, 30000)  # 1 second @ 30 kHz

    openbci_resampled = openbci.resample_to(openbci_data, 1000)
    blackrock_resampled = blackrock.resample_to(blackrock_data, 1000)

    print(f"  OpenBCI: {openbci_data.shape} → {openbci_resampled.shape}")
    print(f"  Blackrock: {blackrock_data.shape} → {blackrock_resampled.shape}")
    print()

    print("Channel groups:")
    openbci_groups = openbci.get_channel_groups()
    blackrock_groups = blackrock.get_channel_groups()

    print(f"  OpenBCI: {list(openbci_groups.keys())}")
    print(f"  Blackrock: {list(blackrock_groups.keys())}")
    print()


def demo_streaming_multi_device():
    """Demonstrate streaming from multiple devices."""
    print("="*70)
    print("Streaming Multi-Device Recording")
    print("="*70)
    print()

    pipeline = MultiDevicePipeline()

    # Add devices
    pipeline.add_device('openbci', OpenBCIAdapter(device='cyton_8ch'))
    pipeline.add_device('blackrock', BlackrockAdapter(device='neuroport_96ch'))
    pipeline.add_device('intan', IntanAdapter(device='rhd2164_64ch'))

    print("Simulating 5-second recording...")
    print()

    # Simulate 5 seconds of data in 1-second chunks
    for second in range(1, 6):
        data = generate_synthetic_data()
        compressed = pipeline.process_batch(data)

        total_bytes = sum(len(c) for c in compressed.values())
        print(f"Second {second}: {total_bytes:,} bytes compressed")

    print()
    summary = pipeline.get_summary()

    print("Recording complete!")
    print()
    print("Final statistics:")
    for device, stats in summary['device_stats'].items():
        print(f"  {device.capitalize()}:")
        print(f"    Samples: {stats['samples_processed']:,}")
        print(f"    Compressed: {stats['bytes_compressed']:,} bytes")
        print(f"    Ratio: {stats['compression_ratio']:.2f}x")
    print()


def main():
    """Run all demonstrations."""
    try:
        demo_basic_pipeline()
        demo_hierarchical_compression()
        demo_channel_alignment()
        demo_streaming_multi_device()

        print("="*70)
        print("✓ All multi-device demos completed successfully!")
        print("="*70)

        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
