#!/usr/bin/env python3
"""
Performance Benchmark Suite

This benchmark suite validates the performance claims made in the README
and provides detailed performance metrics for all algorithms.
"""

import sys
import os
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking."""
    
    def __init__(self):
        self.results = {}
        
        # Performance targets from README
        self.performance_claims = {
            'neural': {
                'neural_lz': {'ratio': (1.5, 3.0), 'latency_ms': 1.0},
                'arithmetic': {'ratio': (2.0, 4.0), 'latency_ms': 1.0},
                'perceptual': {'ratio': (2.0, 10.0), 'latency_ms': 1.0, 'snr_db': (15, 25)},
                'predictive': {'ratio': (1.5, 2.0), 'latency_ms': 2.0},
            },
            'emg': {
                'emg_lz': {'ratio': (5.0, 12.0), 'latency_ms': 25.0, 'quality': (0.85, 0.95)},
                'emg_perceptual': {'ratio': (8.0, 20.0), 'latency_ms': 35.0, 'quality': (0.90, 0.98)},
                'emg_predictive': {'ratio': (10.0, 25.0), 'latency_ms': 50.0, 'quality': (0.88, 0.96)},
                'mobile_emg': {'ratio': (3.0, 8.0), 'latency_ms': 15.0, 'quality': (0.80, 0.90)},
            }
        }
    
    def benchmark_neural_algorithms(self) -> Dict[str, Any]:
        """Benchmark neural compression algorithms."""
        logger.info("Benchmarking Neural Algorithms...")
        
        results = {}
        
        # Generate test data (64 channels, 30k samples as claimed in README)
        neural_data = self._generate_neural_test_data()
        
        # Benchmark Neural LZ
        results['neural_lz'] = self._benchmark_neural_lz(neural_data)
        
        # Benchmark Perceptual Quantization
        results['perceptual'] = self._benchmark_perceptual(neural_data)
        
        return results
    
    def benchmark_emg_algorithms(self) -> Dict[str, Any]:
        """Benchmark EMG compression algorithms."""
        logger.info("Benchmarking EMG Algorithms...")
        
        results = {}
        
        # Generate test data (4 channels, 2k samples as claimed in README)
        emg_data = self._generate_emg_test_data()
        
        # Benchmark EMG algorithms
        results['emg_lz'] = self._benchmark_emg_lz(emg_data)
        results['emg_perceptual'] = self._benchmark_emg_perceptual(emg_data)
        results['emg_predictive'] = self._benchmark_emg_predictive(emg_data)
        results['mobile_emg'] = self._benchmark_mobile_emg(emg_data)
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Test scalability with different data sizes."""
        logger.info("Benchmarking Scalability...")
        
        results = {}
        
        # Test different neural data sizes
        neural_sizes = [
            (32, 10000),    # Small
            (64, 30000),    # Medium (README claim)
            (128, 60000),   # Large
            (256, 120000),  # Very large
        ]
        
        neural_scalability = {}
        for channels, samples in neural_sizes:
            size_name = f"{channels}ch_{samples // 1000}k"
            try:
                test_data = np.random.randn(channels, samples).astype(np.float32)
                data_size_mb = test_data.nbytes / (1024 * 1024)
                
                # Test with Neural LZ
                start_time = time.time()
                from bci_compression.algorithms import create_neural_lz_compressor
                compressor = create_neural_lz_compressor('balanced')
                compressed, metadata = compressor.compress(test_data)
                processing_time = time.time() - start_time
                
                compression_ratio = metadata.get('overall_compression_ratio', 1.0)
                throughput_mbps = data_size_mb / processing_time if processing_time > 0 else 0
                
                neural_scalability[size_name] = {
                    'channels': channels,
                    'samples': samples,
                    'data_size_mb': data_size_mb,
                    'processing_time_s': processing_time,
                    'compression_ratio': compression_ratio,
                    'throughput_mbps': throughput_mbps,
                    'meets_realtime': processing_time < 1.0  # 1 second of data in < 1 second
                }
                
                logger.info(f"Neural {size_name}: {data_size_mb:.1f}MB in {processing_time:.3f}s ({throughput_mbps:.1f} MB/s)")
                
            except Exception as e:
                neural_scalability[size_name] = {'error': str(e)}
        
        results['neural_scalability'] = neural_scalability
        
        # Test EMG scalability
        emg_sizes = [
            (2, 1000),   # Small
            (4, 2000),   # Medium (README claim)
            (8, 4000),   # Large
            (16, 8000),  # Very large
        ]
        
        emg_scalability = {}
        for channels, samples in emg_sizes:
            size_name = f"{channels}ch_{samples}"
            try:
                test_data = np.random.randn(channels, samples).astype(np.float32)
                data_size_mb = test_data.nbytes / (1024 * 1024)
                
                # Test with EMG LZ
                start_time = time.time()
                from bci_compression.algorithms.emg_compression import EMGLZCompressor
                compressor = EMGLZCompressor(sampling_rate=2000.0)
                compressed = compressor.compress(test_data)
                processing_time = time.time() - start_time
                
                compression_ratio = test_data.nbytes / len(compressed)
                throughput_mbps = data_size_mb / processing_time if processing_time > 0 else 0
                
                emg_scalability[size_name] = {
                    'channels': channels,
                    'samples': samples,
                    'data_size_mb': data_size_mb,
                    'processing_time_s': processing_time,
                    'compression_ratio': compression_ratio,
                    'throughput_mbps': throughput_mbps,
                    'meets_realtime': processing_time < 0.05  # Real-time EMG requirement
                }
                
                logger.info(f"EMG {size_name}: {data_size_mb:.1f}MB in {processing_time:.3f}s ({throughput_mbps:.1f} MB/s)")
                
            except Exception as e:
                emg_scalability[size_name] = {'error': str(e)}
        
        results['emg_scalability'] = emg_scalability
        
        return results
    
    def benchmark_realtime_performance(self) -> Dict[str, Any]:
        """Test real-time performance with streaming data."""
        logger.info("Benchmarking Real-time Performance...")
        
        results = {}
        
        # Test streaming neural data processing
        try:
            from bci_compression.algorithms import create_neural_lz_compressor
            compressor = create_neural_lz_compressor('balanced')
            
            # Simulate streaming: 1ms chunks at 30kHz (30 samples per chunk)
            chunk_size = 30
            n_chunks = 1000
            latencies = []
            
            for _ in range(n_chunks):
                chunk = np.random.randn(64, chunk_size).astype(np.float32)
                
                start_time = time.time()
                compressed, metadata = compressor.compress(chunk)
                latency = time.time() - start_time
                latencies.append(latency)
            
            results['neural_streaming'] = {
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'avg_latency_ms': np.mean(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'meets_realtime': np.mean(latencies) < 0.001,  # < 1ms target
                'consistent': np.std(latencies) < 0.0005  # Low variance
            }
            
        except Exception as e:
            results['neural_streaming'] = {'error': str(e)}
        
        # Test streaming EMG data processing
        try:
            from bci_compression.mobile.emg_mobile import MobileEMGCompressor
            compressor = MobileEMGCompressor(
                emg_sampling_rate=1000.0,
                target_latency_ms=25.0
            )
            
            # Simulate streaming: 25ms chunks at 1kHz (25 samples per chunk)
            chunk_size = 25
            n_chunks = 1000
            latencies = []
            
            for _ in range(n_chunks):
                chunk = np.random.randn(4, chunk_size).astype(np.float32)
                
                start_time = time.time()
                compressed = compressor.compress(chunk)
                latency = time.time() - start_time
                latencies.append(latency)
            
            results['emg_streaming'] = {
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'avg_latency_ms': np.mean(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'meets_realtime': np.mean(latencies) < 0.025,  # < 25ms target
                'consistent': np.std(latencies) < 0.005  # Low variance
            }
            
        except Exception as e:
            results['emg_streaming'] = {'error': str(e)}
        
        return results
    
    def _benchmark_neural_lz(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark Neural LZ algorithm."""
        try:
            from bci_compression.algorithms import create_neural_lz_compressor
            
            compressor = create_neural_lz_compressor('balanced')
            
            # Multiple runs for statistics
            runs = 10
            times = []
            ratios = []
            
            for _ in range(runs):
                start_time = time.time()
                compressed, metadata = compressor.compress(data)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                ratios.append(metadata.get('overall_compression_ratio', 1.0))
            
            claims = self.performance_claims['neural']['neural_lz']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            
            return {
                'algorithm': 'Neural LZ',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'ratio_range': (np.min(ratios), np.max(ratios)),
                'latency_range_ms': (np.min(times) * 1000, np.max(times) * 1000),
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio <= claims['ratio'][1] and 
                                   avg_time_ms <= claims['latency_ms']) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'Neural LZ', 'error': str(e), 'status': 'ERROR'}
    
    def _benchmark_perceptual(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark Perceptual Quantization."""
        try:
            from bci_compression.algorithms import PerceptualQuantizer
            
            quantizer = PerceptualQuantizer(base_bits=12)
            
            # Multiple runs for statistics
            runs = 10
            times = []
            ratios = []
            snrs = []
            
            for _ in range(runs):
                start_time = time.time()
                quantized, quant_info = quantizer.quantize(data, quality_level=0.8)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                
                compression_ratio = data.nbytes / quantized.nbytes
                ratios.append(compression_ratio)
                
                # Calculate SNR
                mse = np.mean((data - quantized) ** 2)
                snr = 10 * np.log10(np.var(data) / mse) if mse > 0 else 50.0
                snrs.append(snr)
            
            claims = self.performance_claims['neural']['perceptual']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            avg_snr = np.mean(snrs)
            
            return {
                'algorithm': 'Perceptual Quantization',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'avg_snr_db': avg_snr,
                'ratio_range': (np.min(ratios), np.max(ratios)),
                'latency_range_ms': (np.min(times) * 1000, np.max(times) * 1000),
                'snr_range_db': (np.min(snrs), np.max(snrs)),
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'meets_snr_claim': claims['snr_db'][0] <= avg_snr <= claims['snr_db'][1],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio <= claims['ratio'][1] and 
                                   avg_time_ms <= claims['latency_ms'] and
                                   claims['snr_db'][0] <= avg_snr) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'Perceptual', 'error': str(e), 'status': 'ERROR'}
    
    def _benchmark_emg_lz(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark EMG LZ algorithm."""
        try:
            from bci_compression.algorithms.emg_compression import EMGLZCompressor
            
            compressor = EMGLZCompressor(sampling_rate=2000.0)
            
            # Multiple runs for statistics
            runs = 10
            times = []
            ratios = []
            qualities = []
            
            for _ in range(runs):
                start_time = time.time()
                compressed = compressor.compress(data)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                
                compression_ratio = data.nbytes / len(compressed)
                ratios.append(compression_ratio)
                
                # Test decompression and quality
                decompressed = compressor.decompress(compressed)
                quality = self._calculate_emg_quality(data, decompressed)
                qualities.append(quality)
            
            claims = self.performance_claims['emg']['emg_lz']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            avg_quality = np.mean(qualities)
            
            return {
                'algorithm': 'EMG LZ',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'avg_quality_score': avg_quality,
                'ratio_range': (np.min(ratios), np.max(ratios)),
                'latency_range_ms': (np.min(times) * 1000, np.max(times) * 1000),
                'quality_range': (np.min(qualities), np.max(qualities)),
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'meets_quality_claim': claims['quality'][0] <= avg_quality <= claims['quality'][1],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio <= claims['ratio'][1] and 
                                   avg_time_ms <= claims['latency_ms'] and
                                   avg_quality >= claims['quality'][0]) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'EMG LZ', 'error': str(e), 'status': 'ERROR'}
    
    def _benchmark_emg_perceptual(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark EMG Perceptual algorithm."""
        try:
            from bci_compression.algorithms.emg_compression import EMGPerceptualQuantizer
            
            compressor = EMGPerceptualQuantizer(sampling_rate=2000.0, quality_level=0.8)
            
            runs = 5  # EMG perceptual is slower
            times = []
            ratios = []
            qualities = []
            
            for _ in range(runs):
                start_time = time.time()
                compressed = compressor.compress(data)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                
                compression_ratio = data.nbytes / len(compressed)
                ratios.append(compression_ratio)
                
                decompressed = compressor.decompress(compressed)
                quality = self._calculate_emg_quality(data, decompressed)
                qualities.append(quality)
            
            claims = self.performance_claims['emg']['emg_perceptual']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            avg_quality = np.mean(qualities)
            
            return {
                'algorithm': 'EMG Perceptual',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'avg_quality_score': avg_quality,
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'meets_quality_claim': claims['quality'][0] <= avg_quality <= claims['quality'][1],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio and 
                                   avg_time_ms <= claims['latency_ms'] and
                                   avg_quality >= claims['quality'][0]) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'EMG Perceptual', 'error': str(e), 'status': 'ERROR'}
    
    def _benchmark_emg_predictive(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark EMG Predictive algorithm."""
        try:
            from bci_compression.algorithms.emg_compression import EMGPredictiveCompressor
            
            compressor = EMGPredictiveCompressor()
            
            runs = 3  # EMG predictive is slower
            times = []
            ratios = []
            qualities = []
            
            for _ in range(runs):
                start_time = time.time()
                compressed = compressor.compress(data)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                
                compression_ratio = data.nbytes / len(compressed)
                ratios.append(compression_ratio)
                
                decompressed = compressor.decompress(compressed)
                quality = self._calculate_emg_quality(data, decompressed)
                qualities.append(quality)
            
            claims = self.performance_claims['emg']['emg_predictive']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            avg_quality = np.mean(qualities)
            
            return {
                'algorithm': 'EMG Predictive',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'avg_quality_score': avg_quality,
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'meets_quality_claim': claims['quality'][0] <= avg_quality <= claims['quality'][1],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio and 
                                   avg_time_ms <= claims['latency_ms'] and
                                   avg_quality >= claims['quality'][0]) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'EMG Predictive', 'error': str(e), 'status': 'ERROR'}
    
    def _benchmark_mobile_emg(self, data: np.ndarray) -> Dict[str, Any]:
        """Benchmark Mobile EMG algorithm."""
        try:
            from bci_compression.mobile.emg_mobile import MobileEMGCompressor
            
            compressor = MobileEMGCompressor(
                emg_sampling_rate=1000.0,
                target_latency_ms=15.0,
                battery_level=0.5
            )
            
            # Downsample for mobile (1kHz)
            mobile_data = data[:, ::2]
            
            runs = 10
            times = []
            ratios = []
            qualities = []
            
            for _ in range(runs):
                start_time = time.time()
                compressed = compressor.compress(mobile_data)
                processing_time = time.time() - start_time
                
                times.append(processing_time)
                
                compression_ratio = mobile_data.nbytes / len(compressed)
                ratios.append(compression_ratio)
                
                decompressed = compressor.decompress(compressed)
                quality = self._calculate_emg_quality(mobile_data, decompressed)
                qualities.append(quality)
            
            claims = self.performance_claims['emg']['mobile_emg']
            avg_ratio = np.mean(ratios)
            avg_time_ms = np.mean(times) * 1000
            avg_quality = np.mean(qualities)
            
            return {
                'algorithm': 'Mobile EMG',
                'avg_compression_ratio': avg_ratio,
                'avg_latency_ms': avg_time_ms,
                'avg_quality_score': avg_quality,
                'meets_ratio_claim': claims['ratio'][0] <= avg_ratio <= claims['ratio'][1],
                'meets_latency_claim': avg_time_ms <= claims['latency_ms'],
                'meets_quality_claim': claims['quality'][0] <= avg_quality <= claims['quality'][1],
                'status': 'PASS' if (claims['ratio'][0] <= avg_ratio <= claims['ratio'][1] and 
                                   avg_time_ms <= claims['latency_ms'] and
                                   avg_quality >= claims['quality'][0]) else 'FAIL'
            }
            
        except Exception as e:
            return {'algorithm': 'Mobile EMG', 'error': str(e), 'status': 'ERROR'}
    
    def _generate_neural_test_data(self) -> np.ndarray:
        """Generate realistic neural test data (64 channels, 30k samples)."""
        n_channels, n_samples = 64, 30000
        
        # Base neural noise
        data = np.random.normal(0, 10, (n_channels, n_samples))
        
        # Add spike events
        for ch in range(n_channels):
            n_spikes = np.random.randint(100, 300)
            spike_times = np.random.choice(n_samples, size=n_spikes, replace=False)
            
            for spike_time in spike_times:
                if spike_time < n_samples - 20:
                    # Realistic spike waveform
                    spike_duration = 20
                    spike_amplitude = np.random.uniform(50, 200) * np.random.choice([-1, 1])
                    spike_waveform = spike_amplitude * np.exp(-np.arange(spike_duration) / 5)
                    data[ch, spike_time:spike_time + spike_duration] += spike_waveform
        
        # Add oscillatory activity
        time = np.linspace(0, 1.0, n_samples)
        for ch in range(n_channels):
            # Alpha/beta oscillations
            freq = np.random.uniform(8, 30)  # 8-30 Hz
            amplitude = np.random.uniform(5, 15)
            data[ch, :] += amplitude * np.sin(2 * np.pi * freq * time)
        
        return data.astype(np.float32)
    
    def _generate_emg_test_data(self) -> np.ndarray:
        """Generate realistic EMG test data (4 channels, 2k samples)."""
        n_channels, n_samples = 4, 2000
        sampling_rate = 2000.0
        time = np.linspace(0, 1.0, n_samples)
        
        # Base EMG noise
        data = np.random.normal(0, 0.05, (n_channels, n_samples))
        
        # Add muscle activation patterns
        for ch in range(n_channels):
            # Activation bursts
            burst_start = 0.1 + ch * 0.2
            burst_end = burst_start + 0.4
            burst_mask = (time >= burst_start) & (time <= burst_end)
            
            if np.any(burst_mask):
                # EMG frequency content (primarily 50-200 Hz)
                emg_freqs = [50, 75, 100, 125, 150, 175, 200]
                activation_signal = np.zeros(n_samples)
                
                for freq in emg_freqs:
                    weight = np.exp(-(freq - 125)**2 / 2500)  # Peak around 125 Hz
                    activation_signal += weight * np.sin(2 * np.pi * freq * time + np.random.uniform(0, 2*np.pi))
                
                # Apply activation envelope
                envelope_center = (burst_start + burst_end) / 2
                envelope = np.exp(-((time - envelope_center) ** 2) / 0.1)
                
                # Scale by muscle strength
                muscle_strength = 0.3 + ch * 0.15
                data[ch, :] += muscle_strength * activation_signal * envelope
        
        return data.astype(np.float32)
    
    def _calculate_emg_quality(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate EMG quality score (correlation-based)."""
        try:
            # Overall correlation
            correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
            
            # RMS-based quality
            rms_original = np.sqrt(np.mean(original ** 2))
            rms_error = np.sqrt(np.mean((original - reconstructed) ** 2))
            
            if rms_original > 0:
                snr_linear = rms_original / rms_error if rms_error > 0 else 100
                snr_quality = min(1.0, snr_linear / 10)  # Normalize
            else:
                snr_quality = 0.5
            
            # Combined quality
            if not np.isnan(correlation):
                quality = 0.7 * correlation + 0.3 * snr_quality
            else:
                quality = snr_quality
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5


def run_performance_benchmark():
    """Run complete performance benchmark."""
    logger.info("Starting Performance Benchmark Suite...")
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    benchmark_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'neural_algorithms': benchmark.benchmark_neural_algorithms(),
        'emg_algorithms': benchmark.benchmark_emg_algorithms(),
        'scalability': benchmark.benchmark_scalability(),
        'realtime_performance': benchmark.benchmark_realtime_performance()
    }
    
    # Generate summary
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    error_tests = 0
    
    for category, results in benchmark_results.items():
        if category == 'timestamp':
            continue
            
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'status' in test_result:
                total_tests += 1
                if test_result['status'] == 'PASS':
                    passed_tests += 1
                elif test_result['status'] == 'FAIL':
                    failed_tests += 1
                elif test_result['status'] == 'ERROR':
                    error_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Errors: {error_tests}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    benchmark_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'error_tests': error_tests,
        'success_rate': success_rate,
        'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
    }
    
    if success_rate >= 80:
        logger.info("🎉 PERFORMANCE BENCHMARK PASSED!")
    else:
        logger.warning("⚠️  PERFORMANCE BENCHMARK FAILED")
    
    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'performance_benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    logger.info(f"Benchmark results saved to {results_dir / 'performance_benchmark_results.json'}")
    
    return benchmark_results


if __name__ == "__main__":
    results = run_performance_benchmark()
    
    # Exit with appropriate code
    success_rate = results['summary']['success_rate']
    exit_code = 0 if success_rate >= 80 else 1
    sys.exit(exit_code)
