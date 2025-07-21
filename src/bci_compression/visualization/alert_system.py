"""
Alert system for BCI compression quality monitoring.

This module provides real-time alerting for quality degradation,
artifact detection, and system performance issues.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message container."""
    timestamp: float
    level: AlertLevel
    message: str
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class AlertSystem:
    """
    Real-time alert system for BCI compression monitoring.

    Monitors compression quality, system performance, and detects
    artifacts and quality degradation issues.
    """

    def __init__(
        self,
        max_alerts: int = 1000,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize alert system.

        Parameters
        ----------
        max_alerts : int, default=1000
            Maximum number of alerts to keep in history
        alert_callbacks : List[Callable], optional
            List of callback functions to call when alerts are triggered
        """
        self.max_alerts = max_alerts
        self.alert_callbacks = alert_callbacks or []

        # Alert storage
        self.alerts: List[Alert] = []

        # Alert thresholds
        self.thresholds = {
            'compression_ratio_min': 1.0,
            'compression_ratio_max': 100.0,
            'latency_max_ms': 10.0,
            'snr_min_db': 20.0,
            'psnr_min_db': 30.0,
            'cpu_usage_max_percent': 90.0,
            'memory_usage_max_percent': 90.0,
            'error_rate_max': 0.01,  # 1%
            'gpu_usage_max_percent': 95.0
        }

        # Alert counters
        self.alert_counts = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.ERROR: 0,
            AlertLevel.CRITICAL: 0
        }

        # Thread safety
        self._lock = threading.Lock()

    def set_threshold(self, key: str, value: float) -> None:
        """
        Set alert threshold.

        Parameters
        ----------
        key : str
            Threshold key (e.g., 'latency_max_ms')
        value : float
            Threshold value
        """
        if key in self.thresholds:
            self.thresholds[key] = value
            logger.info(f"Set threshold {key} = {value}")
        else:
            logger.warning(f"Unknown threshold key: {key}")

    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add alert callback function.

        Parameters
        ----------
        callback : Callable
            Function to call when alerts are triggered
        """
        self.alert_callbacks.append(callback)

    def check_compression_metrics(
        self,
        compression_ratio: float,
        latency_ms: float,
        snr_db: float,
        psnr_db: float,
        algorithm_name: str = ""
    ) -> List[Alert]:
        """
        Check compression metrics and generate alerts.

        Parameters
        ----------
        compression_ratio : float
            Achieved compression ratio
        latency_ms : float
            Processing latency in milliseconds
        snr_db : float
            Signal-to-noise ratio in dB
        psnr_db : float
            Peak signal-to-noise ratio in dB
        algorithm_name : str, default=""
            Name of the compression algorithm

        Returns
        -------
        List[Alert]
            List of generated alerts
        """
        alerts = []

        # Check compression ratio
        if compression_ratio < self.thresholds['compression_ratio_min']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"Low compression ratio: {compression_ratio:.2f}x",
                source=f"compression_metrics_{algorithm_name}",
                details={'compression_ratio': compression_ratio}
            ))

        if compression_ratio > self.thresholds['compression_ratio_max']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.INFO,
                message=f"High compression ratio: {compression_ratio:.2f}x",
                source=f"compression_metrics_{algorithm_name}",
                details={'compression_ratio': compression_ratio}
            ))

        # Check latency
        if latency_ms > self.thresholds['latency_max_ms']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.ERROR,
                message=f"High latency: {latency_ms:.2f}ms",
                source=f"compression_metrics_{algorithm_name}",
                details={'latency_ms': latency_ms}
            ))

        # Check SNR
        if snr_db < self.thresholds['snr_min_db']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"Low SNR: {snr_db:.1f}dB",
                source=f"compression_metrics_{algorithm_name}",
                details={'snr_db': snr_db}
            ))

        # Check PSNR
        if psnr_db < self.thresholds['psnr_min_db']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"Low PSNR: {psnr_db:.1f}dB",
                source=f"compression_metrics_{algorithm_name}",
                details={'psnr_db': psnr_db}
            ))

        # Add alerts to system
        for alert in alerts:
            self._add_alert(alert)

        return alerts

    def check_system_metrics(
        self,
        cpu_usage_percent: float,
        memory_usage_percent: float,
        gpu_usage_percent: Optional[float] = None,
        error_count: int = 0,
        total_operations: int = 1
    ) -> List[Alert]:
        """
        Check system metrics and generate alerts.

        Parameters
        ----------
        cpu_usage_percent : float
            CPU usage percentage
        memory_usage_percent : float
            Memory usage percentage
        gpu_usage_percent : float, optional
            GPU usage percentage
        error_count : int, default=0
            Number of errors
        total_operations : int, default=1
            Total number of operations

        Returns
        -------
        List[Alert]
            List of generated alerts
        """
        alerts = []

        # Check CPU usage
        if cpu_usage_percent > self.thresholds['cpu_usage_max_percent']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {cpu_usage_percent:.1f}%",
                source="system_metrics",
                details={'cpu_usage_percent': cpu_usage_percent}
            ))

        # Check memory usage
        if memory_usage_percent > self.thresholds['memory_usage_max_percent']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"High memory usage: {memory_usage_percent:.1f}%",
                source="system_metrics",
                details={'memory_usage_percent': memory_usage_percent}
            ))

        # Check GPU usage
        if gpu_usage_percent is not None and gpu_usage_percent > self.thresholds['gpu_usage_max_percent']:
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                message=f"High GPU usage: {gpu_usage_percent:.1f}%",
                source="system_metrics",
                details={'gpu_usage_percent': gpu_usage_percent}
            ))

        # Check error rate
        if total_operations > 0:
            error_rate = error_count / total_operations
            if error_rate > self.thresholds['error_rate_max']:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level=AlertLevel.ERROR,
                    message=f"High error rate: {error_rate:.3f}",
                    source="system_metrics",
                    details={'error_rate': error_rate, 'error_count': error_count}
                ))

        # Add alerts to system
        for alert in alerts:
            self._add_alert(alert)

        return alerts

    def detect_artifacts(
        self,
        signal_data: 'np.ndarray',
        sampling_rate: float = 30000.0
    ) -> List[Alert]:
        """
        Detect artifacts in neural signals.

        Parameters
        ----------
        signal_data : np.ndarray
            Neural signal data
        sampling_rate : float, default=30000.0
            Signal sampling rate in Hz

        Returns
        -------
        List[Alert]
            List of artifact detection alerts
        """
        alerts = []

        try:
            # Simple artifact detection based on signal statistics
            signal_std = signal_data.std()
            signal_mean = signal_data.mean()

            # Check for clipping (saturation)
            max_val = signal_data.max()
            min_val = signal_data.min()
            if max_val > 0.95 or min_val < -0.95:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level=AlertLevel.WARNING,
                    message="Signal clipping detected",
                    source="artifact_detection",
                    details={'max_val': max_val, 'min_val': min_val}
                ))

            # Check for excessive noise
            if signal_std > 0.5:  # Threshold for excessive noise
                alerts.append(Alert(
                    timestamp=time.time(),
                    level=AlertLevel.WARNING,
                    message="Excessive noise detected",
                    source="artifact_detection",
                    details={'signal_std': signal_std}
                ))

            # Check for DC offset
            if abs(signal_mean) > 0.1:  # Threshold for DC offset
                alerts.append(Alert(
                    timestamp=time.time(),
                    level=AlertLevel.INFO,
                    message="DC offset detected",
                    source="artifact_detection",
                    details={'signal_mean': signal_mean}
                ))

        except Exception as e:
            logger.error(f"Error in artifact detection: {e}")
            alerts.append(Alert(
                timestamp=time.time(),
                level=AlertLevel.ERROR,
                message=f"Artifact detection error: {str(e)}",
                source="artifact_detection",
                details={'error': str(e)}
            ))

        # Add alerts to system
        for alert in alerts:
            self._add_alert(alert)

        return alerts

    def _add_alert(self, alert: Alert) -> None:
        """Add alert to system and trigger callbacks."""
        with self._lock:
            self.alerts.append(alert)
            self.alert_counts[alert.level] += 1

            # Remove old alerts if exceeding max
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.info(f"Alert [{alert.level.value}]: {alert.message}")

    def get_recent_alerts(
        self,
        count: int = 100,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Get recent alerts.

        Parameters
        ----------
        count : int, default=100
            Number of recent alerts to return
        level : AlertLevel, optional
            Filter by alert level

        Returns
        -------
        List[Alert]
            Recent alerts
        """
        with self._lock:
            if level is None:
                return self.alerts[-count:]
            else:
                return [alert for alert in self.alerts[-count:] if alert.level == level]

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary statistics.

        Returns
        -------
        Dict[str, Any]
            Alert summary including counts by level
        """
        with self._lock:
            return {
                'total_alerts': len(self.alerts),
                'alert_counts': self.alert_counts.copy(),
                'unacknowledged_alerts': len([a for a in self.alerts if not a.acknowledged]),
                'recent_alerts': len([a for a in self.alerts if time.time() - a.timestamp < 3600])  # Last hour
            }

    def acknowledge_alert(self, alert_index: int) -> bool:
        """
        Acknowledge an alert.

        Parameters
        ----------
        alert_index : int
            Index of alert to acknowledge

        Returns
        -------
        bool
            True if alert was acknowledged, False if not found
        """
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                return True
            return False

    def clear_alerts(self, level: Optional[AlertLevel] = None) -> None:
        """
        Clear alerts.

        Parameters
        ----------
        level : AlertLevel, optional
            Clear only alerts of this level
        """
        with self._lock:
            if level is None:
                self.alerts.clear()
                self.alert_counts = {level: 0 for level in AlertLevel}
            else:
                self.alerts = [alert for alert in self.alerts if alert.level != level]
                self.alert_counts[level] = 0

        logger.info(f"Cleared alerts (level={level.value if level else 'all'})")
