"""
Real-time metrics collection for BCI compression performance monitoring.

This module collects and tracks compression performance metrics including
compression ratio, latency, SNR, power consumption, and system health.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import GPUtil
except ImportError:
    GPUtil = None
import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Container for compression performance metrics."""
    timestamp: float
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float
    total_latency_ms: float
    snr_db: float
    psnr_db: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    power_consumption_mw: Optional[float] = None
    error_count: int = 0
    algorithm_name: str = ""
    data_size_bytes: int = 0


class MetricsCollector:
    """
    Real-time metrics collector for BCI compression performance.

    Collects and maintains historical metrics for compression algorithms,
    system performance, and quality assessment.
    """

    def __init__(
        self,
        max_history_size: int = 1000,
        collection_interval: float = 1.0,
        enable_system_monitoring: bool = True
    ):
        """
        Initialize metrics collector.

        Parameters
        ----------
        max_history_size : int, default=1000
            Maximum number of metrics to keep in history
        collection_interval : float, default=1.0
            System metrics collection interval in seconds
        enable_system_monitoring : bool, default=True
            Enable automatic system metrics collection
        """
        self.max_history_size = max_history_size
        self.collection_interval = collection_interval
        self.enable_system_monitoring = enable_system_monitoring

        # Metrics storage
        self.compression_metrics: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=max_history_size)

        # Current state
        self.current_metrics = CompressionMetrics(
            timestamp=time.time(),
            compression_ratio=1.0,
            compression_time_ms=0.0,
            decompression_time_ms=0.0,
            total_latency_ms=0.0,
            snr_db=0.0,
            psnr_db=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0
        )

        # System monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        if enable_system_monitoring:
            self._start_system_monitoring()

    def record_compression_metrics(
        self,
        compression_ratio: float,
        compression_time_ms: float,
        decompression_time_ms: float,
        snr_db: float,
        psnr_db: float,
        algorithm_name: str = "",
        data_size_bytes: int = 0,
        error_count: int = 0
    ) -> None:
        """
        Record compression performance metrics.

        Parameters
        ----------
        compression_ratio : float
            Compression ratio achieved
        compression_time_ms : float
            Compression time in milliseconds
        decompression_time_ms : float
            Decompression time in milliseconds
        snr_db : float
            Signal-to-noise ratio in dB
        psnr_db : float
            Peak signal-to-noise ratio in dB
        algorithm_name : str, default=""
            Name of the compression algorithm used
        data_size_bytes : int, default=0
            Size of the original data in bytes
        error_count : int, default=0
            Number of errors encountered
        """
        # Update current metrics
        self.current_metrics.timestamp = time.time()
        self.current_metrics.compression_ratio = compression_ratio
        self.current_metrics.compression_time_ms = compression_time_ms
        self.current_metrics.decompression_time_ms = decompression_time_ms
        self.current_metrics.total_latency_ms = compression_time_ms + decompression_time_ms
        self.current_metrics.snr_db = snr_db
        self.current_metrics.psnr_db = psnr_db
        self.current_metrics.algorithm_name = algorithm_name
        self.current_metrics.data_size_bytes = data_size_bytes
        self.current_metrics.error_count = error_count

        # Add to history
        self.compression_metrics.append(CompressionMetrics(
            timestamp=self.current_metrics.timestamp,
            compression_ratio=self.current_metrics.compression_ratio,
            compression_time_ms=self.current_metrics.compression_time_ms,
            decompression_time_ms=self.current_metrics.decompression_time_ms,
            total_latency_ms=self.current_metrics.total_latency_ms,
            snr_db=self.current_metrics.snr_db,
            psnr_db=self.current_metrics.psnr_db,
            memory_usage_mb=self.current_metrics.memory_usage_mb,
            cpu_usage_percent=self.current_metrics.cpu_usage_percent,
            gpu_usage_percent=self.current_metrics.gpu_usage_percent,
            power_consumption_mw=self.current_metrics.power_consumption_mw,
            error_count=self.current_metrics.error_count,
            algorithm_name=self.current_metrics.algorithm_name,
            data_size_bytes=self.current_metrics.data_size_bytes
        ))

        logger.debug(f"Recorded compression metrics: ratio={compression_ratio:.2f}, "
                    f"latency={self.current_metrics.total_latency_ms:.2f}ms, "
                    f"SNR={snr_db:.1f}dB")

    def get_recent_metrics(self, count: int = 100) -> List[CompressionMetrics]:
        """
        Get recent compression metrics.

        Parameters
        ----------
        count : int, default=100
            Number of recent metrics to return

        Returns
        -------
        List[CompressionMetrics]
            Recent compression metrics
        """
        return list(self.compression_metrics)[-count:]

    def get_system_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent system metrics.

        Parameters
        ----------
        count : int, default=100
            Number of recent system metrics to return

        Returns
        -------
        List[Dict[str, Any]]
            Recent system metrics
        """
        return list(self.system_metrics)[-count:]

    def get_current_metrics(self) -> CompressionMetrics:
        """
        Get current metrics.

        Returns
        -------
        CompressionMetrics
            Current compression metrics
        """
        return self.current_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.

        Returns
        -------
        Dict[str, Any]
            Performance summary including averages, min/max values
        """
        if not self.compression_metrics:
            return {}

        metrics_list = list(self.compression_metrics)

        # Calculate statistics
        compression_ratios = [m.compression_ratio for m in metrics_list]
        latencies = [m.total_latency_ms for m in metrics_list]
        snr_values = [m.snr_db for m in metrics_list]
        psnr_values = [m.psnr_db for m in metrics_list]

        return {
            'compression_ratio': {
                'mean': np.mean(compression_ratios),
                'min': np.min(compression_ratios),
                'max': np.max(compression_ratios),
                'std': np.std(compression_ratios)
            },
            'latency_ms': {
                'mean': np.mean(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'std': np.std(latencies)
            },
            'snr_db': {
                'mean': np.mean(snr_values),
                'min': np.min(snr_values),
                'max': np.max(snr_values),
                'std': np.std(snr_values)
            },
            'psnr_db': {
                'mean': np.mean(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values),
                'std': np.std(psnr_values)
            },
            'total_operations': len(metrics_list),
            'error_rate': sum(m.error_count for m in metrics_list) / len(metrics_list)
        }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)

            # GPU usage (if available)
            gpu_percent = None
            if GPUtil is not None:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                except Exception:
                    pass

            # Power consumption (estimated)
            power_mw = None
            try:
                # Estimate power based on CPU usage
                power_mw = cpu_percent * 10  # Rough estimate: 10mW per 1% CPU
            except Exception:
                pass

            metrics = {
                'timestamp': time.time(),
                'cpu_usage_percent': cpu_percent,
                'memory_usage_mb': memory_mb,
                'memory_percent': memory.percent,
                'gpu_usage_percent': gpu_percent,
                'power_consumption_mw': power_mw
            }

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'timestamp': time.time(),
                'cpu_usage_percent': 0.0,
                'memory_usage_mb': 0.0,
                'memory_percent': 0.0,
                'gpu_usage_percent': None,
                'power_consumption_mw': None
            }

    def _system_monitoring_loop(self) -> None:
        """System monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)

                # Update current metrics with system info
                self.current_metrics.memory_usage_mb = metrics['memory_usage_mb']
                self.current_metrics.cpu_usage_percent = metrics['cpu_usage_percent']
                self.current_metrics.gpu_usage_percent = metrics['gpu_usage_percent']
                self.current_metrics.power_consumption_mw = metrics['power_consumption_mw']

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(self.collection_interval)

    def _start_system_monitoring(self) -> None:
        """Start system monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")

    def clear_history(self) -> None:
        """Clear all historical metrics."""
        self.compression_metrics.clear()
        self.system_metrics.clear()
        logger.info("Metrics history cleared")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring()
