"""Telemetry and monitoring infrastructure."""
import time
from typing import Any, Dict, Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class MetricsCollector:
    """Collector for system and compression metrics."""

    def __init__(self, service_name: str):
        """Initialize metrics collection.
        
        Args:
            service_name: Name of service for metrics
        """
        # Set up OpenTelemetry
        self.meter = metrics.get_meter(service_name)
        self.tracer = trace.get_tracer(service_name)
        
        # Set up Prometheus metrics
        self.registry = CollectorRegistry()
        
        # Compression metrics
        self.compression_ratio = Histogram(
            "compression_ratio",
            "Compression ratio achieved",
            buckets=[1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        self.compression_latency = Histogram(
            "compression_latency_ms",
            "Compression latency in milliseconds",
            buckets=[1, 2, 5, 10, 20, 50, 100, 200],
            registry=self.registry
        )
        
        self.snr = Histogram(
            "signal_to_noise_ratio_db",
            "Signal-to-noise ratio in dB",
            buckets=[10, 15, 20, 25, 30, 35, 40],
            registry=self.registry
        )
        
        # Hardware metrics
        self.gpu_memory_used = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory usage in bytes",
            ["device"],
            registry=self.registry
        )
        
        self.fpga_temperature = Gauge(
            "fpga_temperature_celsius",
            "FPGA temperature in Celsius",
            ["device"],
            registry=self.registry
        )
        
        # Error metrics
        self.compression_errors = Counter(
            "compression_errors_total",
            "Total number of compression errors",
            ["error_type"],
            registry=self.registry
        )
        
        # Initialize span processors
        trace.set_tracer_provider(TracerProvider())
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter())
        )
        
        # Initialize metric exporters
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=10000
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

    def record_compression(
        self,
        ratio: float,
        latency: float,
        snr: float,
        error: Optional[str] = None
    ) -> None:
        """Record compression metrics.
        
        Args:
            ratio: Achieved compression ratio
            latency: Processing latency in ms
            snr: Signal-to-noise ratio in dB
            error: Optional error message
        """
        # Record metrics
        self.compression_ratio.observe(ratio)
        self.compression_latency.observe(latency)
        self.snr.observe(snr)
        
        if error:
            self.compression_errors.labels(error_type=error).inc()

    def record_hardware_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Record hardware-specific metrics.
        
        Args:
            metrics: Dictionary of hardware metrics
        """
        # GPU metrics
        if 'gpu_memory' in metrics:
            for device, memory in metrics['gpu_memory'].items():
                self.gpu_memory_used.labels(device=device).set(memory)
                
        # FPGA metrics
        if 'fpga_temp' in metrics:
            for device, temp in metrics['fpga_temp'].items():
                self.fpga_temperature.labels(device=device).set(temp)

    def current_time_ms(self) -> float:
        """Get current time in milliseconds."""
        return time.time() * 1000

    def increment(self, metric: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        if hasattr(self, metric):
            counter = getattr(self, metric)
            if isinstance(counter, Counter):
                counter.inc(value)

    def record_value(self, metric: str, value: float) -> None:
        """Record a value for a metric."""
        if hasattr(self, metric):
            metric_obj = getattr(self, metric)
            if isinstance(metric_obj, (Gauge, Histogram)):
                metric_obj.observe(value)
