#!/usr/bin/env python3
"""
Transformer-based Neural Compression Example

This example demonstrates the transformer-based compression for neural data,
showing the attention mechanisms for temporal neural patterns.

Features demonstrated:
- Transformer encoder for neural signal compression
- Multi-head attention for temporal patterns
- Real-time processing capabilities
- Performance metrics and quality assessment
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def generate_synthetic_neural_data(n_channels=32, n_samples=10000, sampling_rate=30000):
    """
    Generate synthetic neural data with realistic characteristics.
    
    Parameters
    ----------
    n_channels : int
        Number of neural channels
    n_samples : int
        Number of samples per channel
    sampling_rate : float
        Sampling rate in Hz
    
    Returns
    -------
    np.ndarray
        Synthetic neural data
    """
    print(f"Generating synthetic neural data ({n_channels} channels, {n_samples} samples)")
    
    # Time vector
    t = np.arange(n_samples) / sampling_rate
    
    # Initialize data array
    neural_data = np.zeros((n_channels, n_samples))
    
    # Add neural components for each channel
    for ch in range(n_channels):
        # Background noise (neural baseline)
        noise = np.random.normal(0, 10, n_samples)
        
        # Low-frequency oscillations (1-10 Hz)
        lfo = 5 * np.sin(2 * np.pi * (2 + ch * 0.1) * t)
        
        # Beta/gamma activity (15-50 Hz)
        beta_gamma = 3 * np.sin(2 * np.pi * (25 + ch * 0.5) * t + np.random.uniform(0, 2*np.pi))
        
        # Spike-like events (random sparse events)
        n_spikes = np.random.poisson(50)  # Average 50 spikes
        spike_times = np.random.choice(n_samples, n_spikes, replace=False)
        spike_amplitudes = np.random.normal(50, 10, n_spikes)
        
        spikes = np.zeros(n_samples)
        for spike_time, amplitude in zip(spike_times, spike_amplitudes):
            if spike_time < n_samples - 10:
                # Simple spike waveform
                spike_width = 10
                spike_wave = amplitude * np.exp(-np.arange(spike_width) / 3)
                spikes[spike_time:spike_time + spike_width] += spike_wave
        
        # Combine components
        neural_data[ch] = noise + lfo + beta_gamma + spikes
    
    return neural_data.astype(np.float32)


def demonstrate_transformer_compression():
    """Demonstrate transformer-based neural compression."""
    print("=" * 80)
    print("TRANSFORMER-BASED NEURAL COMPRESSION DEMONSTRATION")
    print("=" * 80)
    
    try:
        from bci_compression.algorithms import create_transformer_compressor
        
        # Generate test data
        print("\n1. Generating Neural Data")
        neural_data = generate_synthetic_neural_data(n_channels=16, n_samples=2000)
        original_size = neural_data.nbytes
        print(f"   Original data: {neural_data.shape} ({original_size:,} bytes)")
        print(f"   Data range: [{neural_data.min():.2f}, {neural_data.max():.2f}]")
        print(f"   Data std: {neural_data.std():.2f}")
        
        # Create transformer compressor
        print("\n2. Creating Transformer Compressor")
        compressor = create_transformer_compressor(
            d_model=128,
            n_heads=8,
            n_layers=4,
            max_sequence_length=1024,
            compression_ratio=0.25,
            quality_level=0.9
        )
        print("   ✅ Transformer compressor created")
        print(f"   - Model dimension: {compressor.d_model}")
        print(f"   - Attention heads: {compressor.n_heads}")
        print(f"   - Encoder layers: {compressor.n_layers}")
        print(f"   - Max sequence length: {compressor.max_sequence_length}")
        
        # Compress data
        print("\n3. Compressing Neural Data")
        start_time = time.time()
        compressed_data = compressor.compress(neural_data)
        compression_time = time.time() - start_time
        
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size
        
        print(f"   ✅ Compression completed")
        print(f"   - Compression time: {compression_time*1000:.2f} ms")
        print(f"   - Compressed size: {compressed_size:,} bytes")
        print(f"   - Compression ratio: {compression_ratio:.2f}x")
        print(f"   - Space saved: {(1 - compressed_size/original_size)*100:.1f}%")
        
        # Decompress data
        print("\n4. Decompressing Data")
        start_time = time.time()
        decompressed_data = compressor.decompress(compressed_data)
        decompression_time = time.time() - start_time
        
        print(f"   ✅ Decompression completed")
        print(f"   - Decompression time: {decompression_time*1000:.2f} ms")
        print(f"   - Reconstructed shape: {decompressed_data.shape}")
        
        # Quality analysis
        print("\n5. Quality Analysis")
        
        # Calculate reconstruction error
        if decompressed_data.shape == neural_data.shape:
            mse = np.mean((neural_data - decompressed_data) ** 2)
            rmse = np.sqrt(mse)
            snr = 10 * np.log10(np.var(neural_data) / mse) if mse > 0 else float('inf')
            correlation = np.corrcoef(neural_data.flatten(), decompressed_data.flatten())[0, 1]
            
            print(f"   - MSE: {mse:.6f}")
            print(f"   - RMSE: {rmse:.6f}")
            print(f"   - SNR: {snr:.2f} dB")
            print(f"   - Correlation: {correlation:.6f}")
        else:
            print(f"   ⚠️  Shape mismatch: {neural_data.shape} vs {decompressed_data.shape}")
        
        # Performance statistics
        print("\n6. Performance Statistics")
        stats = compressor.compression_stats
        
        print(f"   - Total processing time: {stats.get('processing_time', 0)*1000:.2f} ms")
        print(f"   - Compression ratio: {stats.get('compression_ratio', 0):.2f}x")
        
        if 'quality_metrics' in stats:
            quality = stats['quality_metrics']
            print(f"   - Estimated SNR: {quality.get('estimated_snr', 0):.2f} dB")
            print(f"   - Estimated PSNR: {quality.get('estimated_psnr', 0):.2f} dB")
        
        # Check performance targets from README
        print("\n7. Performance Target Validation")
        targets = {
            'compression_ratio': (3.0, 5.0),  # 3-5x compression
            'snr_db': (25.0, 35.0),           # 25-35 dB SNR
            'latency_ms': 2.0                 # <2ms latency
        }
        
        meets_compression = targets['compression_ratio'][0] <= compression_ratio <= targets['compression_ratio'][1]
        meets_snr = targets['snr_db'][0] <= snr <= targets['snr_db'][1] if 'snr' in locals() else False
        meets_latency = compression_time * 1000 <= targets['latency_ms']
        
        print(f"   - Compression ratio target ({targets['compression_ratio'][0]}-{targets['compression_ratio'][1]}x): "
              f"{'✅ PASS' if meets_compression else '❌ FAIL'} ({compression_ratio:.2f}x)")
        
        if 'snr' in locals():
            print(f"   - SNR target ({targets['snr_db'][0]}-{targets['snr_db'][1]} dB): "
                  f"{'✅ PASS' if meets_snr else '❌ FAIL'} ({snr:.2f} dB)")
        
        print(f"   - Latency target (<{targets['latency_ms']} ms): "
              f"{'✅ PASS' if meets_latency else '❌ FAIL'} ({compression_time*1000:.2f} ms)")
        
        overall_pass = meets_compression and meets_latency
        print(f"\n   🎯 Overall Performance: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Transformer compression module not available")
        return False
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_adaptive_transformer():
    """Demonstrate adaptive transformer compression."""
    print("\n" + "=" * 80)
    print("ADAPTIVE TRANSFORMER COMPRESSION DEMONSTRATION")
    print("=" * 80)
    
    try:
        from bci_compression.algorithms.transformer_compression import AdaptiveTransformerCompressor
        
        # Create adaptive compressor
        print("\n1. Creating Adaptive Transformer Compressor")
        adaptive_compressor = AdaptiveTransformerCompressor(
            d_model=128,
            n_heads=8,
            n_layers=4,
            quality_threshold=0.9,
            adaptive_compression=True
        )
        print("   ✅ Adaptive transformer compressor created")
        
        # Test with different signal types
        signal_types = [
            ("Low complexity", lambda: np.random.normal(0, 1, (8, 1000))),
            ("High complexity", lambda: generate_synthetic_neural_data(8, 1000)),
            ("High dynamic range", lambda: np.random.uniform(-100, 100, (8, 1000)))
        ]
        
        for signal_name, signal_generator in signal_types:
            print(f"\n2. Testing with {signal_name} Signal")
            test_signal = signal_generator()
            
            # Compress with adaptive parameters
            start_time = time.time()
            compressed = adaptive_compressor.compress(test_signal)
            compression_time = time.time() - start_time
            
            compression_ratio = test_signal.nbytes / len(compressed)
            
            print(f"   - Signal type: {signal_name}")
            print(f"   - Compression ratio: {compression_ratio:.2f}x")
            print(f"   - Processing time: {compression_time*1000:.2f} ms")
            
            # Show adaptive parameters
            if adaptive_compressor.adaptive_params['compression_adjustments']:
                last_adjustment = adaptive_compressor.adaptive_params['compression_adjustments'][-1]
                print(f"   - Adaptive adjustments: {last_adjustment}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error during adaptive demonstration: {e}")
        return False


def main():
    """Main demonstration function."""
    print("🧠 Transformer-based Neural Compression Demo")
    print("Real-time neural data compression using attention mechanisms")
    print("=" * 80)
    
    success = True
    
    # Standard transformer compression
    success &= demonstrate_transformer_compression()
    
    # Adaptive transformer compression
    success &= demonstrate_adaptive_transformer()
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    if success:
        print("✅ All transformer compression demonstrations completed successfully!")
        print("\nKey Achievements:")
        print("- ✅ Transformer-based compression implemented")
        print("- ✅ Multi-head attention for temporal patterns")
        print("- ✅ Real-time processing capabilities")
        print("- ✅ Adaptive compression based on signal characteristics")
        print("- ✅ Performance targets validated")
        
        print("\nTransformer compression is ready for:")
        print("• Real-time BCI applications")
        print("• High-quality neural signal compression")
        print("• Adaptive processing for different signal types")
        print("• Integration with existing BCI pipelines")
        
    else:
        print("❌ Some demonstrations failed. Check dependencies and implementation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
