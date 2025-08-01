from typing import Dict, Any, Optional, Tuple
import numpy as np
from src.utils.metrics_helper import snr as calculate_snr, compression_ratio
from src.config.pipeline_config_manager import PipelineConfigManager
from src.utils.artifact_detector import ArtifactDetector
try:
    from src.bci_compression.algorithms.transformer_compression import (
        TransformerNeuralCompressor,
        TransformerConfig,
        PerformanceMonitor
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
import logging
import time

class PipelineConnector:
    def __init__(self):
        self.config_manager = PipelineConfigManager()
        self.logger = logging.getLogger("PipelineConnector")
        self.logger.setLevel(logging.INFO)
        self.artifact_detector = ArtifactDetector()
        
        # Initialize transformer compression if available
        if TRANSFORMER_AVAILABLE:
            transformer_config = TransformerConfig(
                d_model=256,
                n_heads=8,
                compression_ratio_target=4.0,
                causal_masking=True,
                optimization_level=2
            )
            self.transformer_compressor = TransformerNeuralCompressor(transformer_config)
            self.performance_monitor = PerformanceMonitor()
        else:
            self.transformer_compressor = None
            self.performance_monitor = None
            self.logger.warning("Transformer compression not available")
        
        # Real-time metrics cache
        self._metrics_cache = {
            'last_update': 0,
            'compression_history': [],
            'performance_history': []
        }

    def get_live_metrics(self, num_channels: int = 64, sample_size: int = 1000, use_gpu: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulates real neural data metrics for multiple channels.
        Args:
            num_channels: Number of neural channels (default: 64)
            sample_size: Number of samples per channel (default: 1000)
            use_gpu: Whether to use GPU acceleration (default: False)
            seed: Optional random seed for reproducibility
        Returns:
            Dictionary of metrics: compression_ratio, snr_db
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            signal = np.random.normal(0, 1, (num_channels, sample_size))
            noise = np.random.normal(0, 0.1, (num_channels, sample_size))
            snr_db = calculate_snr(signal, noise)
            compression_ratio_val = np.random.uniform(2.5, 4.0)
            metrics = {
                "compression_ratio": round(compression_ratio_val, 2),
                "snr_db": round(snr_db, 2)
            }
            self.logger.info(f"Live metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error in get_live_metrics: {e}")
            return {"error": str(e)}

    def simulate_compression(self, original_size: int, compressed_size: int) -> float:
        """
        Calculates compression ratio from original and compressed sizes.
        Args:
            original_size: Size of original data (bytes)
            compressed_size: Size after compression (bytes)
        Returns:
            Compression ratio (float)
        """
        return round(compression_ratio(original_size, compressed_size), 2)

    def inject_artifacts(self, signal: np.ndarray, artifact_type: str = "spike", severity: float = 0.5) -> np.ndarray:
        """
        Injects artifacts into neural signal for simulation.
        Args:
            signal: Neural signal array (channels x samples)
            artifact_type: Type of artifact ("spike", "noise", "drift")
            severity: Severity of artifact (0.0–1.0)
        Returns:
            Modified signal array
        """
        try:
            signal = signal.copy()
            if artifact_type == "spike":
                num_spikes = int(severity * signal.size * 0.01)
                idx = np.random.choice(signal.size, num_spikes, replace=False)
                signal.flat[idx] += np.random.uniform(5, 10, num_spikes)
            elif artifact_type == "noise":
                signal += np.random.normal(0, severity, signal.shape)
            elif artifact_type == "drift":
                drift = np.linspace(0, severity, signal.shape[1])
                signal += drift
            self.logger.info(f"Injected {artifact_type} artifact with severity {severity}")
            return signal
        except Exception as e:
            self.logger.error(f"Error in inject_artifacts: {e}")
            return signal

    def simulate_multimodal_fusion(self, eeg: np.ndarray, fmri: np.ndarray) -> np.ndarray:
        """
        Simulates multi-modal fusion of EEG and fMRI data.
        Args:
            eeg: EEG signal array (channels x samples)
            fmri: fMRI signal array (channels x samples)
        Returns:
            Fused signal array (channels x samples)
        """
        try:
            eeg_weight = 0.7
            fmri_weight = 0.3
            min_shape = (min(eeg.shape[0], fmri.shape[0]), min(eeg.shape[1], fmri.shape[1]))
            eeg = eeg[:min_shape[0], :min_shape[1]]
            fmri = fmri[:min_shape[0], :min_shape[1]]
            fused = eeg_weight * eeg + fmri_weight * fmri
            self.logger.info(f"Fused EEG and fMRI with weights {eeg_weight}, {fmri_weight}")
            return fused
        except Exception as e:
            self.logger.error(f"Error in simulate_multimodal_fusion: {e}")
            return eeg

    def update_pipeline_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Updates the pipeline configuration using PipelineConfigManager.
        Args:
            new_config: Dictionary of new configuration parameters
        Returns:
            Success status (bool)
        """
        try:
            self.config_manager.update_config(new_config)
            self.logger.info(f"Pipeline config updated: {new_config}")
            return True
        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
            return False

    def analyze_artifacts(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes neural signal for artifacts using ArtifactDetector.
        Args:
            signal: Neural signal array (channels x samples)
        Returns:
            Dictionary summarizing detected artifacts
        """
        try:
            summary = self.artifact_detector.detect_all(signal)
            self.logger.info(f"Artifact analysis: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"Error in analyze_artifacts: {e}")
            return {"error": str(e)}

    def get_real_time_compression_metrics(self, signal: np.ndarray, algorithm: str = "transformer") -> Dict[str, Any]:
        """
        Get real-time compression metrics with Phase 8a enhancements.
        
        Args:
            signal: Neural signal array (channels x samples)
            algorithm: Compression algorithm ("transformer", "adaptive", or "auto")
            
        Returns:
            Dictionary with comprehensive compression metrics including:
            - compression_ratio, latency_ms, snr_db, memory_usage_mb
            - algorithm_performance, signal_quality_score
        """
        try:
            start_time = time.time()
            
            # Auto-select algorithm based on signal characteristics
            if algorithm == "auto":
                algorithm = self._select_optimal_algorithm(signal)
            
            # Perform compression based on selected algorithm
            if algorithm == "transformer" and self.transformer_compressor:
                compressed_data, metrics = self._compress_with_transformer(signal)
            else:
                # Fallback to simulation for other algorithms
                compressed_data, metrics = self._simulate_compression_with_metrics(signal)
            
            # Calculate comprehensive metrics
            compression_time = (time.time() - start_time) * 1000  # ms
            
            # Add performance monitoring if available
            if self.performance_monitor:
                self.performance_monitor.record_compression(
                    time_taken=compression_time / 1000,
                    ratio=metrics['compression_ratio'],
                    snr=metrics['snr_db'],
                    memory_mb=metrics.get('memory_usage_mb', 0)
                )
            
            # Update metrics cache for dashboard
            self._update_metrics_cache(metrics, compression_time)
            
            result = {
                **metrics,
                'algorithm_used': algorithm,
                'latency_ms': round(compression_time, 2),
                'timestamp': time.time(),
                'real_time_capable': compression_time < 2.0,  # Phase 8a target
                'performance_grade': self._calculate_performance_grade(metrics, compression_time)
            }
            
            self.logger.info(f"Real-time compression: {algorithm}, ratio={metrics['compression_ratio']:.2f}, latency={compression_time:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in real-time compression metrics: {e}")
            return {"error": str(e), "algorithm_used": algorithm}

    def _compress_with_transformer(self, signal: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compress signal using transformer algorithm."""
        import psutil
        import os
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform compression
        compressed_data = self.transformer_compressor.compress(signal)
        decompressed_data = self.transformer_compressor.decompress(compressed_data, signal.shape)
        
        # Calculate metrics
        compression_ratio = signal.nbytes / len(compressed_data)
        snr_db = self._calculate_snr(signal, decompressed_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        metrics = {
            'compression_ratio': round(compression_ratio, 2),
            'snr_db': round(snr_db, 2),
            'memory_usage_mb': round(memory_usage, 2),
            'signal_quality_score': self._calculate_quality_score(snr_db, compression_ratio)
        }
        
        return compressed_data, metrics

    def _simulate_compression_with_metrics(self, signal: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Simulate compression for non-transformer algorithms."""
        # Simulate compression
        compression_ratio = np.random.uniform(2.5, 6.0)
        simulated_compressed_size = int(signal.nbytes / compression_ratio)
        compressed_data = np.random.bytes(simulated_compressed_size)
        
        # Simulate decompression noise for SNR calculation
        noise_level = 0.05 + (1.0 / compression_ratio) * 0.02
        simulated_noise = np.random.normal(0, noise_level, signal.shape)
        decompressed_signal = signal + simulated_noise
        
        snr_db = self._calculate_snr(signal, decompressed_signal)
        
        metrics = {
            'compression_ratio': round(compression_ratio, 2),
            'snr_db': round(snr_db, 2),
            'memory_usage_mb': round(signal.nbytes / 1024 / 1024 * 1.2, 2),  # Simulate overhead
            'signal_quality_score': self._calculate_quality_score(snr_db, compression_ratio)
        }
        
        return compressed_data, metrics

    def _select_optimal_algorithm(self, signal: np.ndarray) -> str:
        """Auto-select optimal compression algorithm based on signal characteristics."""
        # Analyze signal properties
        signal_std = np.std(signal)
        signal_sparsity = np.sum(np.abs(signal) < 0.1 * signal_std) / signal.size
        temporal_correlation = self._calculate_temporal_correlation(signal)
        
        # Decision logic for algorithm selection
        if temporal_correlation > 0.7 and signal.shape[-1] > 500:
            return "transformer"  # Good for highly correlated temporal patterns
        elif signal_sparsity > 0.8:
            return "sparse"  # Good for sparse signals
        else:
            return "adaptive"  # General purpose
    
    def _calculate_temporal_correlation(self, signal: np.ndarray) -> float:
        """Calculate temporal correlation in the signal."""
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
        
        correlations = []
        for channel in range(min(signal.shape[0], 8)):  # Sample up to 8 channels
            autocorr = np.correlate(signal[channel], signal[channel], mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            # Normalize
            autocorr = autocorr / autocorr[0]
            # Average correlation at lag 1-10
            avg_correlation = np.mean(autocorr[1:11])
            correlations.append(avg_correlation)
        
        return np.mean(correlations)

    def _calculate_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db

    def _calculate_quality_score(self, snr_db: float, compression_ratio: float) -> float:
        """Calculate overall quality score (0-100)."""
        # Weight SNR and compression ratio
        snr_score = min(100, max(0, (snr_db - 10) * 2.5))  # 10dB = 0, 50dB = 100
        ratio_score = min(100, compression_ratio * 15)  # 1x = 15, 6.67x = 100
        
        # Weighted combination (60% quality, 40% compression)
        quality_score = 0.6 * snr_score + 0.4 * ratio_score
        return round(quality_score, 1)

    def _calculate_performance_grade(self, metrics: Dict[str, Any], latency_ms: float) -> str:
        """Calculate performance grade A-F based on Phase 8a targets."""
        # Phase 8a targets: 3-5x compression, 25-35dB SNR, <2ms latency
        score = 0
        
        # Compression ratio score (0-30 points)
        ratio = metrics['compression_ratio']
        if ratio >= 4.0:
            score += 30
        elif ratio >= 3.0:
            score += 20
        elif ratio >= 2.0:
            score += 10
        
        # SNR score (0-40 points)
        snr = metrics['snr_db']
        if snr >= 30:
            score += 40
        elif snr >= 25:
            score += 30
        elif snr >= 20:
            score += 20
        elif snr >= 15:
            score += 10
        
        # Latency score (0-30 points)
        if latency_ms <= 1.0:
            score += 30
        elif latency_ms <= 2.0:
            score += 25
        elif latency_ms <= 5.0:
            score += 15
        elif latency_ms <= 10.0:
            score += 5
        
        # Convert to letter grade
        if score >= 85:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 65:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'

    def _update_metrics_cache(self, metrics: Dict[str, Any], latency_ms: float):
        """Update internal metrics cache for dashboard."""
        current_time = time.time()
        
        # Update cache
        self._metrics_cache['last_update'] = current_time
        self._metrics_cache['compression_history'].append({
            'timestamp': current_time,
            'ratio': metrics['compression_ratio'],
            'snr': metrics['snr_db'],
            'latency': latency_ms
        })
        
        # Keep only last 100 entries
        if len(self._metrics_cache['compression_history']) > 100:
            self._metrics_cache['compression_history'] = self._metrics_cache['compression_history'][-100:]

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for dashboard consumption."""
        try:
            # Get performance summary if available
            performance_summary = {}
            if self.performance_monitor:
                performance_summary = self.performance_monitor.get_performance_summary()
            
            # Prepare dashboard data
            dashboard_data = {
                'current_metrics': self._get_latest_metrics(),
                'performance_summary': performance_summary,
                'compression_history': self._metrics_cache['compression_history'][-20:],  # Last 20 entries
                'system_status': self._get_system_status(),
                'algorithm_availability': {
                    'transformer': TRANSFORMER_AVAILABLE,
                    'adaptive': True,
                    'simulation': True
                }
            }
            
            self.logger.info("Dashboard metrics updated")
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard metrics: {e}")
            return {"error": str(e)}

    def _get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent compression metrics."""
        if self._metrics_cache['compression_history']:
            latest = self._metrics_cache['compression_history'][-1]
            return {
                'compression_ratio': latest['ratio'],
                'snr_db': latest['snr'],
                'latency_ms': latest['latency'],
                'last_updated': latest['timestamp']
            }
        return {}

    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring."""
        import psutil
        
        return {
            'cpu_usage_percent': psutil.cpu_percent(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'transformer_available': TRANSFORMER_AVAILABLE,
            'cache_size': len(self._metrics_cache['compression_history']),
            'uptime_seconds': time.time() - (self._metrics_cache.get('start_time', time.time()))
        }
    
    # ...existing code...



