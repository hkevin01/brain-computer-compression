"""Secure BCI Compression SDK Client."""
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
from cryptography.fernet import Fernet
from opentelemetry import trace
from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor

from src.config.security import SecurityConfig
from src.monitoring.telemetry import MetricsCollector
from src.utils.encryption import EncryptionManager


class BCIPlatform(Enum):
    """Supported BCI hardware platforms."""
    OPENBCI = "openbci"
    NEURALINK = "neuralink"
    KERNEL = "kernel"
    CUSTOM = "custom"


@dataclass
class CompressionRequest:
    """Compression request parameters."""
    signal: np.ndarray
    metadata: Dict[str, Any]
    compression_config: Dict[str, Any]
    platform: BCIPlatform
    encryption_enabled: bool = True
    trace_enabled: bool = True


class BCICompressionClient:
    """Secure client for BCI compression platform."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.bcicompression.io/v1",
        platform: BCIPlatform = BCIPlatform.CUSTOM,
        encryption_key: Optional[str] = None,
        secure_enclave: bool = False
    ):
        """Initialize SDK client.

        Args:
            api_key: API authentication key
            endpoint: API endpoint URL
            platform: BCI hardware platform type
            encryption_key: Optional encryption key for data
            secure_enclave: Whether to use secure enclaves
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.platform = platform

        # Set up security
        self.security_config = SecurityConfig()
        self.encryption_manager = EncryptionManager(
            encryption_key or Fernet.generate_key(),
            use_enclave=secure_enclave
        )

        # Set up monitoring
        self.logger = logging.getLogger("BCICompressionSDK")
        self.metrics = MetricsCollector("bci_compression_sdk")
        self.tracer = trace.get_tracer(__name__)

        # Set up async HTTP client
        self.session = None
        AioHttpClientInstrumentor().instrument()

    async def __aenter__(self):
        """Set up async context."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Platform": self.platform.value
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        if self.session:
            await self.session.close()

    async def compress_signal(
        self,
        request: CompressionRequest
    ) -> Dict[str, Any]:
        """Compress neural signal data.

        Args:
            request: Compression request parameters

        Returns:
            Dictionary with compressed data and metrics
        """
        async with self.tracer.start_as_current_span("compress_signal") as span:
            try:
                # Track metrics
                self.metrics.increment("compression_requests")
                start_time = self.metrics.current_time_ms()

                # Encrypt sensitive data if enabled
                if request.encryption_enabled:
                    signal_bytes = request.signal.tobytes()
                    encrypted_data = self.encryption_manager.encrypt(signal_bytes)
                    metadata = self.encryption_manager.encrypt(
                        json.dumps(request.metadata).encode()
                    )
                else:
                    encrypted_data = request.signal.tobytes()
                    metadata = json.dumps(request.metadata).encode()

                # Prepare request
                payload = {
                    "data": encrypted_data,
                    "metadata": metadata,
                    "config": request.compression_config,
                    "encryption_enabled": request.encryption_enabled
                }

                # Make API request
                async with self.session.post(
                    f"{self.endpoint}/compress",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                # Track latency
                latency = self.metrics.current_time_ms() - start_time
                self.metrics.record_value("compression_latency", latency)

                # Add trace info
                if request.trace_enabled:
                    span.set_attribute("compression.size", len(encrypted_data))
                    span.set_attribute("compression.latency", latency)

                return result

            except Exception as e:
                self.logger.error(f"Compression failed: {e}")
                self.metrics.increment("compression_errors")
                raise

    async def get_supported_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of available compression algorithms."""
        try:
            async with self.session.get(
                f"{self.endpoint}/algorithms"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.logger.error(f"Failed to get algorithms: {e}")
            return []

    async def submit_custom_algorithm(
        self,
        algorithm_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit custom compression algorithm to marketplace."""
        try:
            async with self.session.post(
                f"{self.endpoint}/marketplace/submit",
                json=algorithm_def
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.logger.error(f"Algorithm submission failed: {e}")
            raise

    def configure_platform(
        self,
        platform_config: Dict[str, Any]
    ) -> None:
        """Configure BCI platform-specific settings."""
        if self.platform == BCIPlatform.OPENBCI:
            self._configure_openbci(platform_config)
        elif self.platform == BCIPlatform.NEURALINK:
            self._configure_neuralink(platform_config)
        elif self.platform == BCIPlatform.KERNEL:
            self._configure_kernel(platform_config)

    def _configure_openbci(self, config: Dict[str, Any]) -> None:
        """Configure OpenBCI-specific settings."""
        # Set up board type, sample rate, etc.
        self.logger.info(f"Configuring OpenBCI: {config}")

    def _configure_neuralink(self, config: Dict[str, Any]) -> None:
        """Configure Neuralink-specific settings."""
        # Set up device ID, streaming mode, etc.
        self.logger.info(f"Configuring Neuralink: {config}")

    def _configure_kernel(self, config: Dict[str, Any]) -> None:
        """Configure Kernel-specific settings."""
        # Set up device parameters, data format, etc.
        self.logger.info(f"Configuring Kernel: {config}")
