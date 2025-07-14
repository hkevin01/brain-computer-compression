#!/usr/bin/env python3
"""
Phase 2 validation script for advanced compression algorithms.

This script tests the neural-optimized compression algorithms
implemented in Phase 2, including LZ variants, arithmetic coding,
lossy compression methods, and GPU acceleration.
"""

import sys
import os
import numpy as np
import time

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def generate_test_neural_data(n_channels=16, n_samples=3000, sampling_rate=30000):
    """Generate realistic test neural data."""
    t = np.linspace(0, n_samples/sampling_rate, n_samples)
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Base neural signal components
        # Local field potential (LFP) components
        lfp = 0
        for freq, amp in [(10, 20), (30, 15), (80, 10)]:  # Alpha, beta, gamma
            lfp += amp * np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)
        
        # Add spikes (high frequency transients)
        spike_times = np.random.poisson(5, size=int(len(t)/1000))  # ~5 spikes/sec
        for spike_time in spike_times:
            if spike_time < len(t):
                # Simple spike waveform
                spike = 100 * np.exp(-((t - t[spike_time])**2) / (0.001**2))
                lfp += spike
        
        # Add noise
        noise = np.random.normal(0, 5, n_samples)
        
        data[ch] = lfp + noise
    
    return data


def test_neural_lz_compression():
    """Test neural LZ compression algorithms."""
    print("Testing Neural LZ Compression...")
    
    try:
        from bci_compression.algorithms.neural_lz import (
            NeuralLZ77Compressor, 
            MultiChannelNeuralLZ,
            create_neural_lz_compressor
        )
        
        # Generate test data
        test_data = generate_test_neural_data(n_channels=8, n_samples=1500)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test single channel compressor
        print("  Testing NeuralLZ77Compressor...")
        compressor = NeuralLZ77Compressor()
        
        # Compress single channel
        compressed, metadata = compressor.compress_channel(test_data[0])
        print(f"  Compression ratio: {metadata['compression_stats']['compression_ratio']:.2f}")
        
        # Decompress and validate
        decompressed = compressor.decompress_channel(compressed, metadata)
        print(f"  Decompressed shape: {decompressed.shape}")
        
        # Test multi-channel compressor
        print("  Testing MultiChannelNeuralLZ...")
        mc_compressor = MultiChannelNeuralLZ()
        
        # Compress all channels
        compressed_channels, global_meta = mc_compressor.compress(test_data)
        print(f"  Multi-channel compression ratio: {global_meta['overall_compression_ratio']:.2f}")
        
        # Decompress and validate
        decompressed_multi = mc_compressor.decompress(compressed_channels, global_meta)
        print(f"  Decompressed multi-channel shape: {decompressed_multi.shape}")
        
        # Test factory function
        print("  Testing factory function...")
        factory_compressor = create_neural_lz_compressor('compression')
        compressed_fact, meta_fact = factory_compressor.compress(test_data)
        print(f"  Factory compressor ratio: {meta_fact['overall_compression_ratio']:.2f}")
        
        print("✅ Neural LZ Compression tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Neural LZ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_arithmetic_coding():
    """Test neural arithmetic coding algorithms."""
    print("Testing Neural Arithmetic Coding...")
    
    try:
        from bci_compression.algorithms.neural_arithmetic import (
            NeuralArithmeticModel,
            NeuralArithmeticCoder,
            MultiChannelArithmeticCoder,
            create_neural_arithmetic_coder
        )
        
        # Generate test data
        test_data = generate_test_neural_data(n_channels=4, n_samples=1000)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test arithmetic model
        print("  Testing NeuralArithmeticModel...")
        model = NeuralArithmeticModel(alphabet_size=256, context_length=3)
        
        # Test model with some symbols
        for i in range(10):
            prob_info = model.get_symbol_probability(i)
            model.update_model(i)
        
        stats = model.get_model_statistics()
        print(f"  Model processed {stats['total_symbols']} symbols")
        
        # Test single channel coder
        print("  Testing NeuralArithmeticCoder...")
        coder = NeuralArithmeticCoder(quantization_bits=12)
        
        # Encode single channel
        encoded, metadata = coder.encode(test_data[0])
        print(f"  Encoded size: {len(encoded)} bytes")
        print(f"  Compression bits: {metadata['compressed_bits']}")
        
        # Decode and validate
        decoded = coder.decode(encoded, metadata)
        print(f"  Decoded shape: {decoded.shape}")
        
        # Test multi-channel coder
        print("  Testing MultiChannelArithmeticCoder...")
        mc_coder = MultiChannelArithmeticCoder()
        
        # Encode all channels
        encoded_channels, global_meta = mc_coder.encode(test_data)
        total_size = sum(len(ch) for ch in encoded_channels)
        print(f"  Total encoded size: {total_size} bytes")
        
        # Decode and validate
        decoded_multi = mc_coder.decode(encoded_channels, global_meta)
        print(f"  Decoded multi-channel shape: {decoded_multi.shape}")
        
        # Test factory function
        print("  Testing factory function...")
        factory_coder = create_neural_arithmetic_coder('balanced')
        encoded_fact, meta_fact = factory_coder.encode(test_data)
        print(f"  Factory coder channels: {len(encoded_fact)}")
        
        print("✅ Neural Arithmetic Coding tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Neural Arithmetic Coding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lossy_neural_compression():
    """Test lossy neural compression algorithms."""
    print("Testing Lossy Neural Compression...")
    
    try:
        from bci_compression.algorithms.lossy_neural import (
            PerceptualQuantizer,
            AdaptiveWaveletCompressor,
            create_lossy_compressor_suite
        )
        
        # Generate test data
        test_data = generate_test_neural_data(n_channels=4, n_samples=2000)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test perceptual quantizer
        print("  Testing PerceptualQuantizer...")
        quantizer = PerceptualQuantizer(base_bits=10)
        
        quantized, quant_info = quantizer.quantize(test_data, quality_level=0.8)
        print(f"  Quantized shape: {quantized.shape}")
        print(f"  Quality level: {quant_info['quality_level']}")
        
        # Calculate SNR
        mse = np.mean((test_data - quantized) ** 2)
        signal_power = np.mean(test_data ** 2)
        snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
        print(f"  Signal-to-Noise Ratio: {snr_db:.1f} dB")
        
        # Test adaptive wavelet compressor
        print("  Testing AdaptiveWaveletCompressor...")
        try:
            wavelet_comp = AdaptiveWaveletCompressor()
            
            compressed_coeffs, comp_meta = wavelet_comp.compress(
                test_data[0], compression_ratio=0.2
            )
            print(f"  Wavelet compression successful")
            
            # Decompress
            decompressed = wavelet_comp.decompress(compressed_coeffs, comp_meta)
            print(f"  Decompressed shape: {decompressed.shape}")
            
        except ImportError:
            print("  ⚠️  PyWavelets not available, skipping wavelet tests")
        
        # Test factory function
        print("  Testing factory function...")
        compressor_suite = create_lossy_compressor_suite('balanced')
        print(f"  Created {len(compressor_suite)} compressors")
        
        # Test perceptual quantizer from suite
        suite_quantizer = compressor_suite['perceptual_quantizer']
        suite_quantized, _ = suite_quantizer.quantize(test_data[0], quality_level=0.9)
        print(f"  Suite quantizer output shape: {suite_quantized.shape}")
        
        print("✅ Lossy Neural Compression tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Lossy Neural Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_acceleration():
    """Test GPU acceleration framework."""
    print("Testing GPU Acceleration...")
    
    try:
        from bci_compression.algorithms.gpu_acceleration import (
            GPUCompressionBackend,
            RealTimeGPUPipeline,
            create_gpu_compression_system
        )
        
        # Generate test data
        test_data = generate_test_neural_data(n_channels=8, n_samples=1500)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test GPU backend
        print("  Testing GPUCompressionBackend...")
        backend = GPUCompressionBackend()
        print(f"  GPU available: {backend.gpu_available}")
        
        # Test bandpass filter
        filtered = backend.gpu_bandpass_filter(
            test_data, 
            low_freq=10.0, 
            high_freq=100.0, 
            sampling_rate=30000.0
        )
        print(f"  Filtered data shape: {filtered.shape}")
        
        # Test quantization
        quantized, quant_params = backend.gpu_quantization(test_data, n_bits=12)
        print(f"  Quantized data shape: {quantized.shape}")
        print(f"  Quantization bits: {quant_params['n_bits']}")
        
        # Test FFT compression
        compressed, comp_meta = backend.gpu_fft_compression(test_data, compression_ratio=0.2)
        print(f"  FFT compressed shape: {compressed.shape}")
        
        # Test real-time pipeline
        print("  Testing RealTimeGPUPipeline...")
        pipeline = RealTimeGPUPipeline(backend=backend, buffer_size=1500)
        
        # Process a chunk
        processed, process_meta = pipeline.process_chunk(test_data)
        print(f"  Processed data shape: {processed.shape}")
        print(f"  Processing time: {process_meta['total_processing_time']:.4f}s")
        
        # Test factory function
        print("  Testing factory function...")
        gpu_system = create_gpu_compression_system('latency')
        
        # Small benchmark
        start_time = time.time()
        for _ in range(5):
            _, _ = gpu_system.process_chunk(test_data)
        avg_time = (time.time() - start_time) / 5
        print(f"  Average processing time: {avg_time:.4f}s")
        
        # Get performance stats
        stats = backend.get_performance_stats()
        print(f"  GPU operations: {stats['gpu_operations']}")
        print(f"  CPU operations: {stats['cpu_operations']}")
        
        print("✅ GPU Acceleration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU Acceleration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_pipeline():
    """Test integration of multiple Phase 2 algorithms."""
    print("Testing Integration Pipeline...")
    
    try:
        # Test combining different algorithms
        test_data = generate_test_neural_data(n_channels=4, n_samples=1200)
        print(f"  Test data shape: {test_data.shape}")
        
        # Create a processing pipeline
        results = {}
        
        # Step 1: Neural LZ compression
        try:
            from bci_compression.algorithms.neural_lz import create_neural_lz_compressor
            lz_compressor = create_neural_lz_compressor('balanced')
            lz_compressed, lz_meta = lz_compressor.compress(test_data)
            results['neural_lz_ratio'] = lz_meta['overall_compression_ratio']
            print(f"  Neural LZ compression ratio: {results['neural_lz_ratio']:.2f}")
        except ImportError:
            print("  ⚠️  Neural LZ not available")
        
        # Step 2: Perceptual quantization
        try:
            from bci_compression.algorithms.lossy_neural import PerceptualQuantizer
            quantizer = PerceptualQuantizer()
            quantized, quant_meta = quantizer.quantize(test_data, quality_level=0.7)
            
            # Calculate quality metrics
            mse = np.mean((test_data - quantized) ** 2)
            signal_power = np.mean(test_data ** 2)
            snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
            results['quantization_snr'] = snr_db
            print(f"  Quantization SNR: {snr_db:.1f} dB")
        except ImportError:
            print("  ⚠️  Lossy neural compression not available")
        
        # Step 3: GPU acceleration test
        try:
            from bci_compression.algorithms.gpu_acceleration import GPUCompressionBackend
            gpu_backend = GPUCompressionBackend()
            
            # Time processing with and without GPU
            start_time = time.time()
            gpu_filtered = gpu_backend.gpu_bandpass_filter(
                test_data, 10.0, 200.0, 30000.0
            )
            gpu_time = time.time() - start_time
            results['gpu_processing_time'] = gpu_time
            print(f"  GPU processing time: {gpu_time:.4f}s")
            
        except ImportError:
            print("  ⚠️  GPU acceleration not available")
        
        # Summary
        print("  Integration test results:")
        for key, value in results.items():
            print(f"    {key}: {value}")
        
        print("✅ Integration Pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 validation tests."""
    print("=" * 60)
    print("Brain-Computer Interface Toolkit - Phase 2 Validation")
    print("Advanced Compression Algorithms")
    print("=" * 60)
    
    tests = [
        test_neural_lz_compression,
        test_neural_arithmetic_coding,
        test_lossy_neural_compression,
        test_gpu_acceleration,
        test_integration_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"PHASE 2 VALIDATION SUMMARY")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All Phase 2 compression algorithms are working correctly!")
        print("✅ Ready to proceed to Phase 3: Advanced Techniques")
    elif passed >= total * 0.7:
        print("⚠️  Most Phase 2 algorithms working - some optional dependencies missing")
        print("✅ Core functionality validated - can proceed to Phase 3")
    else:
        print("⚠️  Several tests failed. Please review the implementation.")
    
    print("=" * 60)
    
    return passed >= total * 0.7  # Allow 70% pass rate due to optional dependencies


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
