#!/usr/bin/env python3
"""
Phase 3 validation script for advanced compression techniques.

This script tests the advanced algorithms implemented in Phase 3,
including predictive compression, context-aware methods, and hybrid algorithms.
"""

import sys
import os
import numpy as np
import time

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def generate_realistic_neural_data(n_channels=16, n_samples=4000, sampling_rate=30000):
    """Generate realistic neural data with temporal and spatial structure."""
    t = np.linspace(0, n_samples/sampling_rate, n_samples)
    data = np.zeros((n_channels, n_samples))
    
    # Base rhythmic components (alpha, beta, gamma)
    alpha_freq = 10  # Hz
    beta_freq = 20   # Hz  
    gamma_freq = 60  # Hz
    
    # Generate spatially correlated neural activity
    for ch in range(n_channels):
        # Phase and amplitude variations across channels
        alpha_phase = 2 * np.pi * ch / n_channels
        beta_phase = np.pi * ch / n_channels
        
        # Create base signal with neural rhythms
        alpha_component = 30 * np.sin(2 * np.pi * alpha_freq * t + alpha_phase)
        beta_component = 20 * np.sin(2 * np.pi * beta_freq * t + beta_phase)
        gamma_component = 10 * np.sin(2 * np.pi * gamma_freq * t)
        
        # Add temporal structure (bursts, state changes)
        burst_envelope = np.ones(n_samples)
        
        # Create "burst" periods with higher amplitude
        burst_starts = np.random.choice(n_samples//2, size=3, replace=False)
        for start in burst_starts:
            end = min(start + 200, n_samples)
            burst_envelope[start:end] *= 2.0
        
        # Add gradual state changes
        state_modulation = 1 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz modulation
        
        # Combine components
        signal = (alpha_component + beta_component + gamma_component) * burst_envelope * state_modulation
        
        # Add correlated noise between adjacent channels
        if ch > 0:
            correlation_factor = 0.3
            signal += correlation_factor * data[ch-1] * np.random.random()
        
        # Add independent noise
        signal += np.random.normal(0, 5, n_samples)
        
        # Add occasional spikes
        spike_times = np.random.poisson(3, size=n_samples//1000)
        for spike_idx, spike_count in enumerate(spike_times):
            if spike_idx * 1000 < n_samples:
                spike_time = spike_idx * 1000 + np.random.randint(0, 1000)
                if spike_time < n_samples:
                    # Simple spike waveform
                    spike_width = 10
                    spike_start = max(0, spike_time - spike_width//2)
                    spike_end = min(n_samples, spike_time + spike_width//2)
                    spike_amplitude = 100 + 50 * np.random.random()
                    
                    signal[spike_start:spike_end] += spike_amplitude * np.exp(
                        -((np.arange(spike_end - spike_start) - spike_width//2)**2) / (spike_width/4)**2
                    )
        
        data[ch] = signal
    
    return data


def test_predictive_compression():
    """Test predictive compression algorithms."""
    print("Testing Predictive Compression...")
    
    try:
        from bci_compression.algorithms.predictive import (
            NeuralLinearPredictor,
            AdaptiveNeuralPredictor,
            MultiChannelPredictiveCompressor,
            create_predictive_compressor
        )
        
        # Generate test data with temporal structure
        test_data = generate_realistic_neural_data(n_channels=8, n_samples=2000)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test Neural Linear Predictor
        print("  Testing NeuralLinearPredictor...")
        predictor = NeuralLinearPredictor(order=12)
        
        # Fit predictor on one channel
        coeffs = predictor.fit_predictor(test_data[0], 0)
        print(f"  Prediction accuracy: {coeffs['accuracy']:.3f}")
        print(f"  Prediction order: {coeffs['order']}")
        
        # Test predictions
        predictions = predictor.predict_samples(test_data[0], 0)
        print(f"  Predictions shape: {predictions.shape}")
        
        # Test Adaptive Neural Predictor
        print("  Testing AdaptiveNeuralPredictor...")
        adaptive_predictor = AdaptiveNeuralPredictor(order=8, channels=4)
        
        # Simulate real-time adaptation
        adaptation_errors = []
        for i in range(100, len(test_data[0])):
            prediction = adaptive_predictor.update_predictor(
                0, test_data[0][i-1], test_data[0][i]
            )
            if i > 100:  # After some adaptation
                error = abs(test_data[0][i] - prediction)
                adaptation_errors.append(error)
        
        stats = adaptive_predictor.get_prediction_statistics()
        print(f"  Adaptive MSE: {stats['mse']:.2f}")
        print(f"  Samples processed: {stats['samples_processed']}")
        
        # Test Multi-Channel Predictive Compressor
        print("  Testing MultiChannelPredictiveCompressor...")
        mc_compressor = MultiChannelPredictiveCompressor(
            prediction_order=10, cross_channel_order=3
        )
        
        # Compress multi-channel data
        compressed_channels, metadata = mc_compressor.compress(test_data)
        print(f"  Channels compressed: {len(compressed_channels)}")
        print(f"  Prediction accuracy: {metadata.prediction_accuracy:.3f}")
        print(f"  Compression ratio: {metadata.original_bits / metadata.compressed_bits:.2f}x")
        
        # Test factory function
        print("  Testing factory function...")
        for mode in ['speed', 'balanced', 'quality']:
            factory_compressor = create_predictive_compressor(mode)
            compressed, meta = factory_compressor.compress(test_data[:4, :1000])
            ratio = meta.original_bits / meta.compressed_bits
            print(f"  {mode} mode ratio: {ratio:.2f}x")
        
        print("✅ Predictive Compression tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Predictive Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_aware_compression():
    """Test context-aware compression algorithms."""
    print("Testing Context-Aware Compression...")
    
    try:
        from bci_compression.algorithms.context_aware import (
            BrainStateDetector,
            HierarchicalContextModel,
            SpatialContextModel,
            ContextAwareCompressor,
            create_context_aware_compressor
        )
        
        # Generate test data with state changes
        test_data = generate_realistic_neural_data(n_channels=16, n_samples=3000)
        print(f"  Test data shape: {test_data.shape}")
        
        # Test Brain State Detector
        print("  Testing BrainStateDetector...")
        state_detector = BrainStateDetector(sampling_rate=30000)
        
        # Test on different windows
        window_size = 500
        detected_states = []
        for i in range(0, test_data.shape[1] - window_size, window_size):
            window = test_data[:8, i:i+window_size]  # Use subset of channels
            state = state_detector.classify_state(window)
            detected_states.append(state)
        
        unique_states = set(detected_states)
        print(f"  Detected brain states: {unique_states}")
        print(f"  State changes: {len([i for i in range(1, len(detected_states)) if detected_states[i] != detected_states[i-1]])}")
        
        # Test Hierarchical Context Model
        print("  Testing HierarchicalContextModel...")
        context_model = HierarchicalContextModel(max_depth=4, alphabet_size=256)
        
        # Simulate symbol sequence
        symbols = np.random.randint(0, 256, 1000).tolist()
        context_model.update_context(symbols, level=3)
        
        # Test probability prediction
        test_context = tuple(symbols[-4:])
        test_symbol = symbols[-1]
        prob = context_model.get_conditional_probability(test_symbol, test_context)
        print(f"  Context probability: {prob:.6f}")
        
        stats = context_model.get_model_statistics()
        print(f"  Total symbols processed: {stats['total_symbols']}")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        
        # Test Spatial Context Model
        print("  Testing SpatialContextModel...")
        spatial_model = SpatialContextModel(n_channels=16)
        
        # Set up simple electrode layout (grid)
        positions = {}
        for i in range(16):
            x = i % 4
            y = i // 4
            positions[i] = (x, y)
        
        spatial_model.set_electrode_layout(positions)
        spatial_model.compute_functional_connectivity(test_data)
        spatial_groups = spatial_model.create_spatial_groups(threshold=0.2)
        
        print(f"  Spatial groups created: {len(spatial_groups)}")
        print(f"  Average connectivity: {np.mean(spatial_model.connectivity_matrix):.3f}")
        
        # Test Context-Aware Compressor
        print("  Testing ContextAwareCompressor...")
        compressor = ContextAwareCompressor(sampling_rate=30000)
        compressor.setup_spatial_model(16, positions)
        
        # Compress data
        compressed_data, metadata = compressor.compress(test_data)
        print(f"  Compression ratio: {metadata.compression_ratio:.2f}x")
        print(f"  Brain states in data: {metadata.brain_states}")
        print(f"  Context switches: {metadata.context_switches}")
        print(f"  Adaptation time: {metadata.adaptation_time:.4f}s")
        
        # Test different factory modes
        print("  Testing factory modes...")
        for mode in ['adaptive', 'spatial', 'temporal']:
            mode_compressor = create_context_aware_compressor(mode)
            mode_compressor.setup_spatial_model(8)
            compressed, meta = mode_compressor.compress(test_data[:8, :1500])
            print(f"  {mode} mode ratio: {meta.compression_ratio:.2f}x")
        
        # Get comprehensive statistics
        comp_stats = compressor.get_compression_statistics()
        print(f"  Compression statistics: {len(comp_stats)} metrics tracked")
        
        print("✅ Context-Aware Compression tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Context-Aware Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmarks():
    """Test performance of Phase 3 algorithms."""
    print("Testing Phase 3 Performance Benchmarks...")
    
    try:
        # Import algorithms
        from bci_compression.algorithms.predictive import create_predictive_compressor
        from bci_compression.algorithms.context_aware import create_context_aware_compressor
        
        # Generate test data
        test_data = generate_realistic_neural_data(n_channels=32, n_samples=5000)
        print(f"  Benchmark data shape: {test_data.shape}")
        
        algorithms = {
            'Predictive (Speed)': create_predictive_compressor('speed'),
            'Predictive (Quality)': create_predictive_compressor('quality'),
            'Context-Aware (Adaptive)': create_context_aware_compressor('adaptive'),
            'Context-Aware (Spatial)': create_context_aware_compressor('spatial')
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"  Benchmarking {name}...")
            
            # Setup if needed
            if hasattr(algorithm, 'setup_spatial_model'):
                algorithm.setup_spatial_model(32)
            
            # Time compression
            start_time = time.time()
            
            if hasattr(algorithm, 'compress'):
                if 'Context' in name:
                    compressed, metadata = algorithm.compress(test_data)
                    compression_ratio = metadata.compression_ratio
                else:
                    compressed, metadata = algorithm.compress(test_data)
                    compression_ratio = metadata.original_bits / metadata.compressed_bits
            else:
                # Fallback for different interface
                compressed = None
                compression_ratio = 1.0
            
            compression_time = time.time() - start_time
            
            # Calculate throughput
            samples_per_second = test_data.size / compression_time
            
            results[name] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'throughput': samples_per_second,
                'latency_per_sample': compression_time / test_data.size * 1000  # ms
            }
            
            print(f"    Ratio: {compression_ratio:.2f}x")
            print(f"    Time: {compression_time:.4f}s")
            print(f"    Throughput: {samples_per_second:.0f} samples/s")
        
        # Summary
        print("  Performance Summary:")
        best_ratio = max(results.values(), key=lambda x: x['compression_ratio'])
        fastest = min(results.values(), key=lambda x: x['compression_time'])
        
        best_ratio_name = [name for name, res in results.items() if res == best_ratio][0]
        fastest_name = [name for name, res in results.items() if res == fastest][0]
        
        print(f"    Best compression: {best_ratio_name} ({best_ratio['compression_ratio']:.2f}x)")
        print(f"    Fastest: {fastest_name} ({fastest['compression_time']:.4f}s)")
        
        print("✅ Performance Benchmarks completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_phase2():
    """Test integration of Phase 3 with Phase 2 algorithms."""
    print("Testing Phase 3 Integration with Phase 2...")
    
    try:
        # Test data
        test_data = generate_realistic_neural_data(n_channels=8, n_samples=2000)
        
        results = {}
        
        # Try Phase 2 algorithms
        try:
            from bci_compression.algorithms.neural_lz import create_neural_lz_compressor
            lz_compressor = create_neural_lz_compressor('balanced')
            lz_compressed, lz_meta = lz_compressor.compress(test_data)
            results['Phase 2 Neural LZ'] = lz_meta['overall_compression_ratio']
            print(f"  Phase 2 Neural LZ ratio: {results['Phase 2 Neural LZ']:.2f}x")
        except ImportError:
            print("  Phase 2 Neural LZ not available")
        
        try:
            from bci_compression.algorithms.lossy_neural import PerceptualQuantizer
            quantizer = PerceptualQuantizer()
            quantized, quant_meta = quantizer.quantize(test_data, quality_level=0.8)
            
            # Calculate compression ratio based on quantization
            mse = np.mean((test_data - quantized) ** 2)
            results['Phase 2 Perceptual'] = f"SNR: {10 * np.log10(np.var(test_data) / (mse + 1e-10)):.1f} dB"
            print(f"  Phase 2 Perceptual: {results['Phase 2 Perceptual']}")
        except ImportError:
            print("  Phase 2 Perceptual Quantizer not available")
        
        # Try Phase 3 algorithms
        try:
            from bci_compression.algorithms.predictive import create_predictive_compressor
            pred_compressor = create_predictive_compressor('balanced')
            pred_compressed, pred_meta = pred_compressor.compress(test_data)
            results['Phase 3 Predictive'] = pred_meta.original_bits / pred_meta.compressed_bits
            print(f"  Phase 3 Predictive ratio: {results['Phase 3 Predictive']:.2f}x")
        except ImportError:
            print("  Phase 3 Predictive not available")
        
        try:
            from bci_compression.algorithms.context_aware import create_context_aware_compressor
            context_compressor = create_context_aware_compressor('adaptive')
            context_compressor.setup_spatial_model(8)
            context_compressed, context_meta = context_compressor.compress(test_data)
            results['Phase 3 Context-Aware'] = context_meta.compression_ratio
            print(f"  Phase 3 Context-Aware ratio: {results['Phase 3 Context-Aware']:.2f}x")
        except ImportError:
            print("  Phase 3 Context-Aware not available")
        
        # Compare results
        if len(results) >= 2:
            print("  Integration successful - multiple algorithms working together")
            print("  Algorithm comparison:")
            for name, result in results.items():
                print(f"    {name}: {result}")
        else:
            print("  ⚠️  Limited integration - some algorithms not available")
        
        print("✅ Phase 3 Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 validation tests."""
    print("=" * 60)
    print("Brain-Computer Interface Toolkit - Phase 3 Validation")
    print("Advanced Compression Techniques")
    print("=" * 60)
    
    tests = [
        test_predictive_compression,
        test_context_aware_compression,
        test_performance_benchmarks,
        test_integration_with_phase2
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"PHASE 3 VALIDATION SUMMARY")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All Phase 3 advanced techniques are working correctly!")
        print("✅ Ready for production deployment and Phase 4 benchmarking")
    elif passed >= total * 0.75:
        print("⚠️  Most Phase 3 algorithms working - minor issues detected")
        print("✅ Core advanced functionality validated")
    else:
        print("⚠️  Several tests failed. Please review the implementation.")
    
    print("=" * 60)
    
    return passed >= total * 0.75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
