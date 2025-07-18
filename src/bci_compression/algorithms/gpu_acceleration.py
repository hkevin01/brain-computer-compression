"""
GPU acceleration framework for real-time neural data compression.

This module provides GPU-accelerated implementations of compression
algorithms optimized for real-time brain-computer interface applications.
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    _has_cupy = True
except ImportError:
    cp = None
    _has_cupy = False


class GPUCompressionBackend:
    """
    GPU acceleration backend for neural compression algorithms.

    This class provides GPU-accelerated implementations of core
    compression operations with fallback to CPU implementations.
    """

    def __init__(self, device_id: int = 0, enable_memory_pool: bool = True):
        """
        Initialize GPU compression backend.

        Parameters
        ----------
        device_id : int, default=0
            GPU device ID to use
        enable_memory_pool : bool, default=True
            Whether to enable memory pooling for efficiency
        """
        self.device_id = device_id
        self.enable_memory_pool = enable_memory_pool
        self.gpu_available = False
        self.cupy_available = False

        # Try to initialize GPU support
        if _has_cupy:
            try:
                # Set device and initialize
                with cp.cuda.Device(device_id):
                    # Test basic operations
                    test_array = cp.array([1, 2, 3])
                    _ = cp.sum(test_array)
                    self.gpu_available = True

                    if enable_memory_pool:
                        # Enable memory pool for efficiency
                        mempool = cp.get_default_memory_pool()
                        mempool.set_limit(size=2**30)  # 1GB limit

                    warnings.warn(f"GPU acceleration enabled on device {device_id}")

            except Exception as e:
                warnings.warn(f"GPU initialization failed: {e}, using CPU fallback")
        else:
            warnings.warn("CuPy not available, using CPU fallback")

        # Performance monitoring
        self.performance_stats = {
            'gpu_operations': 0,
            'cpu_operations': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'memory_transfers': 0,
            'transfer_time': 0.0
        }

    def to_gpu(self, data: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Transfer data to GPU if available.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        array
            GPU array if available, otherwise CPU array
        """
        if self.gpu_available:
            start_time = time.time()
            gpu_data = self.cp.asarray(data)
            self.performance_stats['transfer_time'] += time.time() - start_time
            self.performance_stats['memory_transfers'] += 1
            return gpu_data
        return data

    def to_cpu(self, data: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """
        Transfer data to CPU.

        Parameters
        ----------
        data : array
            GPU or CPU array

        Returns
        -------
        np.ndarray
            CPU array
        """
        if self.gpu_available and hasattr(data, 'get'):
            start_time = time.time()
            cpu_data = data.get()
            self.performance_stats['transfer_time'] += time.time() - start_time
            self.performance_stats['memory_transfers'] += 1
            return cpu_data
        return np.asarray(data)

    def gpu_bandpass_filter(
        self,
        data: Union[np.ndarray, 'cp.ndarray'],
        low_freq: float,
        high_freq: float,
        sampling_rate: float
    ) -> Union[np.ndarray, 'cp.ndarray']:
        """
        GPU-accelerated bandpass filtering.

        Parameters
        ----------
        data : array
            Input data
        low_freq : float
            Low cutoff frequency
        high_freq : float
            High cutoff frequency
        sampling_rate : float
            Sampling rate

        Returns
        -------
        array
            Filtered data
        """
        if not self.gpu_available:
            # Fallback to CPU implementation
            return self._cpu_bandpass_filter(data, low_freq, high_freq, sampling_rate)

        start_time = time.time()

        with self.cp.cuda.Device(self.device_id):
            # Ensure data is on GPU
            gpu_data = self.to_gpu(data)

            # Design filter coefficients on CPU (small operation)
            from scipy import signal
            nyquist = sampling_rate / 2
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist

            b, a = signal.butter(4, [low_norm, high_norm], btype='band')

            # Transfer filter coefficients to GPU
            b_gpu = self.cp.asarray(b)
            a_gpu = self.cp.asarray(a)

            # Apply filter using GPU FFT
            if gpu_data.ndim == 1:
                # Single channel
                filtered_data = self._gpu_filtfilt(gpu_data, b_gpu, a_gpu)
            else:
                # Multi-channel
                filtered_data = self.cp.zeros_like(gpu_data)
                for ch in range(gpu_data.shape[0]):
                    filtered_data[ch] = self._gpu_filtfilt(gpu_data[ch], b_gpu, a_gpu)

        self.performance_stats['gpu_operations'] += 1
        self.performance_stats['gpu_time'] += time.time() - start_time

        return filtered_data

    def _gpu_filtfilt(self, data: 'cp.ndarray', b: 'cp.ndarray', a: 'cp.ndarray') -> 'cp.ndarray':
        """
        GPU implementation of zero-phase filtering.

        This is a simplified implementation using FFT-based filtering.
        For production use, consider more sophisticated implementations.
        """
        # Simple FFT-based filtering (approximation of filtfilt)
        # In practice, you'd want a more accurate implementation

        # FFT of signal
        n_fft = len(data)
        data_fft = self.cp.fft.fft(data, n_fft)

        # Create frequency response
        freqs = self.cp.fft.fftfreq(n_fft)

        # Simple frequency domain filter (approximation)
        # This is a basic implementation - for production, use proper filter design
        w = 2 * self.cp.pi * freqs

        # Evaluate filter frequency response (simplified)
        h_num = self.cp.polyval(b[::-1], self.cp.exp(1j * w))
        h_den = self.cp.polyval(a[::-1], self.cp.exp(1j * w))
        h = h_num / (h_den + 1e-12)

        # Apply filter
        filtered_fft = data_fft * h

        # IFFT to get filtered signal
        filtered_data = self.cp.real(self.cp.fft.ifft(filtered_fft))

        return filtered_data

    def _cpu_bandpass_filter(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float,
        sampling_rate: float
    ) -> np.ndarray:
        """CPU fallback for bandpass filtering."""
        start_time = time.time()

        from scipy import signal

        # Design filter
        nyquist = sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')

        # Apply filter
        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])

        self.performance_stats['cpu_operations'] += 1
        self.performance_stats['cpu_time'] += time.time() - start_time

        return filtered_data

    def gpu_quantization(
        self,
        data: Union[np.ndarray, 'cp.ndarray'],
        n_bits: int = 12
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Dict]:
        """
        GPU-accelerated uniform quantization.

        Parameters
        ----------
        data : array
            Input data
        n_bits : int, default=12
            Number of quantization bits

        Returns
        -------
        tuple
            (quantized_data, quantization_params)
        """
        if not self.gpu_available:
            return self._cpu_quantization(data, n_bits)

        start_time = time.time()

        with self.cp.cuda.Device(self.device_id):
            # Ensure data is on GPU
            gpu_data = self.to_gpu(data)

            # Calculate quantization parameters
            data_min = self.cp.min(gpu_data)
            data_max = self.cp.max(gpu_data)
            data_range = data_max - data_min

            n_levels = 2 ** n_bits

            if data_range > 0:
                # Normalize to [0, 1]
                normalized = (gpu_data - data_min) / data_range

                # Quantize
                quantized = self.cp.round(normalized * (n_levels - 1))

                # Scale back
                dequantized = (quantized / (n_levels - 1)) * data_range + data_min
            else:
                dequantized = gpu_data

            # Quantization parameters
            params = {
                'data_min': float(self.to_cpu(data_min)),
                'data_max': float(self.to_cpu(data_max)),
                'n_bits': n_bits,
                'n_levels': n_levels
            }

        self.performance_stats['gpu_operations'] += 1
        self.performance_stats['gpu_time'] += time.time() - start_time

        return dequantized, params

    def _cpu_quantization(self, data: np.ndarray, n_bits: int) -> Tuple[np.ndarray, Dict]:
        """CPU fallback for quantization."""
        start_time = time.time()

        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min
        n_levels = 2 ** n_bits

        if data_range > 0:
            normalized = (data - data_min) / data_range
            quantized = np.round(normalized * (n_levels - 1))
            dequantized = (quantized / (n_levels - 1)) * data_range + data_min
        else:
            dequantized = data

        params = {
            'data_min': data_min,
            'data_max': data_max,
            'n_bits': n_bits,
            'n_levels': n_levels
        }

        self.performance_stats['cpu_operations'] += 1
        self.performance_stats['cpu_time'] += time.time() - start_time

        return dequantized, params

    def gpu_fft_compression(
        self,
        data: Union[np.ndarray, 'cp.ndarray'],
        compression_ratio: float = 0.1
    ) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Dict]:
        """
        GPU-accelerated FFT-based compression.

        Parameters
        ----------
        data : array
            Input data
        compression_ratio : float, default=0.1
            Fraction of coefficients to keep

        Returns
        -------
        tuple
            (compressed_data, compression_metadata)
        """
        if not self.gpu_available:
            return self._cpu_fft_compression(data, compression_ratio)

        start_time = time.time()

        with self.cp.cuda.Device(self.device_id):
            # Ensure data is on GPU
            gpu_data = self.to_gpu(data)
            original_shape = gpu_data.shape

            if gpu_data.ndim == 1:
                gpu_data = gpu_data.reshape(1, -1)

            n_channels, n_samples = gpu_data.shape
            compressed_channels = []

            for ch in range(n_channels):
                channel_data = gpu_data[ch]

                # FFT
                fft_data = self.cp.fft.fft(channel_data)

                # Keep only significant coefficients
                magnitudes = self.cp.abs(fft_data)
                threshold = self.cp.percentile(magnitudes, (1 - compression_ratio) * 100)

                # Zero out small coefficients
                mask = magnitudes >= threshold
                compressed_fft = fft_data * mask

                # IFFT to get compressed signal
                compressed_signal = self.cp.real(self.cp.fft.ifft(compressed_fft))
                compressed_channels.append(compressed_signal)

            # Stack channels
            compressed_data = self.cp.stack(compressed_channels)
            compressed_data = compressed_data.reshape(original_shape)

            metadata = {
                'compression_ratio': compression_ratio,
                'original_shape': original_shape,
                'method': 'fft'
            }

        self.performance_stats['gpu_operations'] += 1
        self.performance_stats['gpu_time'] += time.time() - start_time

        return compressed_data, metadata

    def _cpu_fft_compression(
        self,
        data: np.ndarray,
        compression_ratio: float
    ) -> Tuple[np.ndarray, Dict]:
        """CPU fallback for FFT compression."""
        start_time = time.time()

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        compressed_channels = []

        for ch in range(n_channels):
            channel_data = data[ch]

            # FFT
            fft_data = np.fft.fft(channel_data)

            # Keep only significant coefficients
            magnitudes = np.abs(fft_data)
            threshold = np.percentile(magnitudes, (1 - compression_ratio) * 100)

            # Zero out small coefficients
            mask = magnitudes >= threshold
            compressed_fft = fft_data * mask

            # IFFT to get compressed signal
            compressed_signal = np.real(np.fft.ifft(compressed_fft))
            compressed_channels.append(compressed_signal)

        # Stack channels
        compressed_data = np.array(compressed_channels)
        compressed_data = compressed_data.reshape(original_shape)

        metadata = {
            'compression_ratio': compression_ratio,
            'original_shape': original_shape,
            'method': 'fft'
        }

        self.performance_stats['cpu_operations'] += 1
        self.performance_stats['cpu_time'] += time.time() - start_time

        return compressed_data, metadata

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        total_operations = (self.performance_stats['gpu_operations'] +
                            self.performance_stats['cpu_operations'])

        if total_operations > 0:
            gpu_percentage = (self.performance_stats['gpu_operations'] /
                              total_operations) * 100
        else:
            gpu_percentage = 0

        total_compute_time = (self.performance_stats['gpu_time'] +
                              self.performance_stats['cpu_time'])

        stats = self.performance_stats.copy()
        stats.update({
            'total_operations': total_operations,
            'gpu_percentage': gpu_percentage,
            'total_compute_time': total_compute_time,
            'average_gpu_time': (stats['gpu_time'] / max(1, stats['gpu_operations'])),
            'average_cpu_time': (stats['cpu_time'] / max(1, stats['cpu_operations'])),
            'gpu_available': self.gpu_available
        })

        return stats

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            with self.cp.cuda.Device(self.device_id):
                mempool = self.cp.get_default_memory_pool()
                mempool.free_all_blocks()

    def get_status(self) -> dict:
        """Return current backend status (CPU/GPU, device, memory pool, etc.)."""
        return {
            'gpu_available': self.gpu_available,
            'cupy_available': _has_cupy,
            'device_id': self.device_id,
            'enable_memory_pool': self.enable_memory_pool
        }


class RealTimeGPUPipeline:
    """
    Real-time GPU processing pipeline for neural compression.

    This class provides a complete GPU-accelerated pipeline for
    real-time neural data compression and decompression.
    """

    def __init__(
        self,
        backend: Optional[GPUCompressionBackend] = None,
        buffer_size: int = 3000,  # 100ms at 30kHz
        processing_pipeline: Optional[List[str]] = None
    ):
        """
        Initialize real-time GPU pipeline.

        Parameters
        ----------
        backend : GPUCompressionBackend, optional
            GPU backend to use
        buffer_size : int, default=3000
            Processing buffer size
        processing_pipeline : list, optional
            List of processing steps
        """
        if backend is None:
            self.backend = GPUCompressionBackend()
        else:
            self.backend = backend

        self.buffer_size = buffer_size

        if processing_pipeline is None:
            self.processing_pipeline = [
                'bandpass_filter',
                'quantization',
                'fft_compression'
            ]
        else:
            self.processing_pipeline = processing_pipeline

        # Pipeline configuration
        self.pipeline_config = {
            'bandpass_filter': {
                'low_freq': 1.0,
                'high_freq': 300.0,
                'sampling_rate': 30000.0
            },
            'quantization': {
                'n_bits': 12
            },
            'fft_compression': {
                'compression_ratio': 0.2
            }
        }

        # Performance monitoring
        self.pipeline_stats = {
            'processed_chunks': 0,
            'total_processing_time': 0.0,
            'average_latency': 0.0
        }

    def process_chunk(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single data chunk through the GPU pipeline.

        Parameters
        ----------
        data : np.ndarray
            Input data chunk

        Returns
        -------
        tuple
            (processed_data, processing_metadata)
        """
        start_time = time.time()

        # Transfer to GPU
        processed_data = self.backend.to_gpu(data)
        metadata = {'pipeline_steps': []}

        # Execute pipeline steps
        for step in self.processing_pipeline:
            step_start = time.time()

            if step == 'bandpass_filter':
                config = self.pipeline_config['bandpass_filter']
                processed_data = self.backend.gpu_bandpass_filter(
                    processed_data,
                    config['low_freq'],
                    config['high_freq'],
                    config['sampling_rate']
                )

            elif step == 'quantization':
                config = self.pipeline_config['quantization']
                processed_data, quant_params = self.backend.gpu_quantization(
                    processed_data,
                    config['n_bits']
                )
                metadata['quantization_params'] = quant_params

            elif step == 'fft_compression':
                config = self.pipeline_config['fft_compression']
                processed_data, comp_metadata = self.backend.gpu_fft_compression(
                    processed_data,
                    config['compression_ratio']
                )
                metadata['compression_metadata'] = comp_metadata

            step_time = time.time() - step_start
            metadata['pipeline_steps'].append({
                'step': step,
                'processing_time': step_time
            })

        # Transfer back to CPU
        final_data = self.backend.to_cpu(processed_data)

        # Update statistics
        total_time = time.time() - start_time
        self.pipeline_stats['processed_chunks'] += 1
        self.pipeline_stats['total_processing_time'] += total_time
        self.pipeline_stats['average_latency'] = (
            self.pipeline_stats['total_processing_time'] /
            self.pipeline_stats['processed_chunks']
        )

        metadata['total_processing_time'] = total_time

        return final_data, metadata

    def update_config(self, step: str, config: Dict):
        """Update configuration for a pipeline step."""
        if step in self.pipeline_config:
            self.pipeline_config[step].update(config)

    def get_pipeline_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        stats = self.pipeline_stats.copy()
        stats.update(self.backend.get_performance_stats())
        return stats

    def benchmark_pipeline(
        self,
        test_data: np.ndarray,
        n_iterations: int = 100
    ) -> Dict:
        """
        Benchmark the GPU pipeline performance.

        Parameters
        ----------
        test_data : np.ndarray
            Test data for benchmarking
        n_iterations : int, default=100
            Number of benchmark iterations

        Returns
        -------
        dict
            Benchmark results
        """
        print(f"Benchmarking GPU pipeline with {n_iterations} iterations...")

        # Warm up
        for _ in range(10):
            _, _ = self.process_chunk(test_data)

        # Clear statistics
        self.pipeline_stats = {
            'processed_chunks': 0,
            'total_processing_time': 0.0,
            'average_latency': 0.0
        }

        # Benchmark
        latencies = []

        for i in range(n_iterations):
            start_time = time.time()
            _, metadata = self.process_chunk(test_data)
            latency = time.time() - start_time
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{n_iterations} iterations")

        # Calculate statistics
        latencies = np.array(latencies)

        benchmark_results = {
            'n_iterations': n_iterations,
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'throughput_chunks_per_sec': n_iterations / np.sum(latencies),
            'data_shape': test_data.shape,
            'gpu_available': self.backend.gpu_available,
            'pipeline_config': self.pipeline_config
        }

        # Add backend statistics
        benchmark_results.update(self.get_pipeline_stats())

        return benchmark_results


def create_gpu_compression_system(
    optimization_target: str = 'latency',
    device_id: int = 0
) -> RealTimeGPUPipeline:
    """
    Factory function to create optimized GPU compression system.

    Parameters
    ----------
    optimization_target : str, default='latency'
        Optimization target ('latency', 'throughput', 'quality')
    device_id : int, default=0
        GPU device ID

    Returns
    -------
    RealTimeGPUPipeline
        Configured GPU pipeline
    """
    backend = GPUCompressionBackend(device_id=device_id)

    if optimization_target == 'latency':
        # Minimize latency
        pipeline = RealTimeGPUPipeline(
            backend=backend,
            buffer_size=1500,  # 50ms at 30kHz
            processing_pipeline=['bandpass_filter', 'quantization']
        )
        pipeline.update_config('quantization', {'n_bits': 10})

    elif optimization_target == 'throughput':
        # Maximize throughput
        pipeline = RealTimeGPUPipeline(
            backend=backend,
            buffer_size=6000,  # 200ms at 30kHz
            processing_pipeline=['quantization', 'fft_compression']
        )
        pipeline.update_config('fft_compression', {'compression_ratio': 0.3})

    else:  # quality
        # Optimize for quality
        pipeline = RealTimeGPUPipeline(
            backend=backend,
            buffer_size=3000,
            processing_pipeline=['bandpass_filter', 'quantization', 'fft_compression']
        )
        pipeline.update_config('quantization', {'n_bits': 14})
        pipeline.update_config('fft_compression', {'compression_ratio': 0.1})

    return pipeline
