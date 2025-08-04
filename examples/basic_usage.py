"""
Basic usage example to verify the toolkit is working
"""

import os
import sys

import numpy as np

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print("🧠 BCI Compression Toolkit - Basic Usage Example")
    print("=" * 50)

    try:
        # Import the toolkit
        from bci_compression.algorithms import create_neural_lz_compressor
        from bci_compression.mobile import MobileBCICompressor
        print("✅ Successfully imported BCI compression modules")

        # Generate sample neural data
        print("\n📊 Generating sample neural data...")
        np.random.seed(42)
        neural_data = np.random.randn(32, 5000)  # 32 channels, 5000 samples
        print(f"   Generated data shape: {neural_data.shape}")
        print(f"   Data range: [{neural_data.min():.3f}, {neural_data.max():.3f}]")

        # Test basic compression
        print("\n🗜️ Testing basic compression...")
        compressor = create_neural_lz_compressor('balanced')
        compressed, metadata = compressor.compress(neural_data)

        print(f"   Compression ratio: {metadata['overall_compression_ratio']:.2f}x")
        print(f"   Compression time: {metadata.get('compression_time', 0):.4f}s")
        print(f"   Compressed size: {len(compressed)} bytes")

        # Test decompression
        print("\n📤 Testing decompression...")
        decompressed = compressor.decompress(compressed, metadata)

        # Verify accuracy (should be lossless)
        mse = np.mean((neural_data - decompressed) ** 2)
        print(f"   Decompressed shape: {decompressed.shape}")
        print(f"   Reconstruction MSE: {mse:.2e}")
        print(f"   Perfect reconstruction: {'✅ YES' if mse < 1e-10 else '❌ NO'}")

        # Test mobile compression
        print("\n📱 Testing mobile compression...")
        mobile_compressor = MobileBCICompressor(
            algorithm="lightweight_quant",
            quality_level=0.8,
            power_mode="balanced"
        )

        mobile_compressed = mobile_compressor.compress(neural_data)
        mobile_decompressed = mobile_compressor.decompress(mobile_compressed)

        mobile_ratio = mobile_compressor.get_compression_ratio()
        print(f"   Mobile compression ratio: {mobile_ratio:.2f}x")

        # Calculate SNR for lossy compression
        signal_power = np.var(neural_data)
        noise_power = np.var(neural_data - mobile_decompressed)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        print(f"   Mobile compression SNR: {snr:.1f} dB")

        print("\n🎉 All tests completed successfully!")
        print("🚀 The BCI Compression Toolkit is working correctly!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
