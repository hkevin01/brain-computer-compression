"""
MetricsStreamer for real-time metrics streaming.
Supports future WebSocket integration for dashboard frontend.

References:
- Real-time streaming requirements (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Callable, Dict, Any
import threading
import time
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector

pipeline_connector = PipelineConnector()

class MetricsStreamer:
    """
    Streams real-time metrics to subscribers (e.g., WebSocket clients).
    Now uses PipelineConnector for real metrics.
    """
    def __init__(self):
        self.subscribers: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self.running = False

    def subscribe(self, client_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Registers a subscriber callback for metrics updates.
        """
        self.subscribers[client_id] = callback

    def unsubscribe(self, client_id: str) -> None:
        """
        Removes a subscriber.
        """
        if client_id in self.subscribers:
            del self.subscribers[client_id]

    def start_stream(self, interval: float = 1.0) -> None:
        """
        Starts streaming metrics to all subscribers at the given interval (seconds).
        """
        self.running = True
        def stream_loop():
            while self.running:
                metrics = self.get_metrics()
                for callback in self.subscribers.values():
                    callback(metrics)
                time.sleep(interval)
        threading.Thread(target=stream_loop, daemon=True).start()

    def stop_stream(self) -> None:
        """
        Stops the metrics streaming loop.
        """
        self.running = False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns current metrics from PipelineConnector.
        """
        return pipeline_connector.get_live_metrics()
