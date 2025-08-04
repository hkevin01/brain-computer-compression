"""FPGA-based neural signal processing with ultra-low latency."""
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pynq
    from pynq import Overlay
    FPGA_AVAILABLE = True
except ImportError:
    FPGA_AVAILABLE = False

from src.hardware.accelerator_interface import HardwareAccelerator, PowerMode


class FPGAAccelerator(HardwareAccelerator):
    """FPGA-based hardware acceleration for neural signal processing."""

    def __init__(self):
        """Initialize FPGA accelerator."""
        if not FPGA_AVAILABLE:
            raise RuntimeError("PYNQ FPGA libraries not available")

        self.overlay = None
        self.dma = None
        self.power_mode = PowerMode.BALANCED
        self._circular_buffers = {}

    def initialize(self, config: Dict) -> bool:
        """Initialize FPGA device and bitstream.

        Args:
            config: Configuration dictionary with:
                - bitstream_path: Path to FPGA bitstream
                - clock_freq: Operating frequency in MHz
                - buffer_size: Size of DMA buffers

        Returns:
            Success status
        """
        try:
            # Load FPGA bitstream
            self.overlay = Overlay(config['bitstream_path'])

            # Configure DMA engine
            self.dma = self.overlay.axi_dma_0

            # Set up circular buffers for zero-copy DMA
            buffer_size = config.get('buffer_size', 8192)
            self._setup_dma_buffers(buffer_size)

            # Configure clock frequency
            if 'clock_freq' in config:
                self._set_clock_frequency(config['clock_freq'])

            return True

        except Exception as e:
            print(f"FPGA initialization failed: {e}")
            return False

    def process_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process neural signal using FPGA acceleration.

        Args:
            signal: Input neural signal array

        Returns:
            Tuple of (processed_signal, metrics)
        """
        try:
            # Get DMA buffer
            in_buffer = self._get_input_buffer(signal.shape)
            out_buffer = self._get_output_buffer(signal.shape)

            # Copy input data
            np.copyto(in_buffer, signal)

            # Start DMA transfer
            start_time = self.overlay.ip_dict['timer'].read()

            self.dma.sendchannel.transfer(in_buffer)
            self.dma.recvchannel.transfer(out_buffer)

            # Wait for completion
            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()

            end_time = self.overlay.ip_dict['timer'].read()

            # Calculate metrics
            processing_time_us = (end_time - start_time) / 100.0  # Convert to microseconds

            metrics = {
                'latency_us': processing_time_us,
                'throughput_mbps': signal.nbytes * 8 / (processing_time_us * 1000),
                'buffer_utilization': len(self._circular_buffers)
            }

            return np.copy(out_buffer), metrics

        except Exception as e:
            print(f"FPGA processing failed: {e}")
            return signal, {'error': str(e)}

    def set_power_mode(self, mode: PowerMode) -> None:
        """Set FPGA power optimization mode."""
        self.power_mode = mode

        # Adjust clock frequency based on power mode
        if mode == PowerMode.PERFORMANCE:
            self._set_clock_frequency(200)  # 200MHz
        elif mode == PowerMode.BALANCED:
            self._set_clock_frequency(100)  # 100MHz
        elif mode == PowerMode.POWER_SAVE:
            self._set_clock_frequency(50)   # 50MHz

    def get_device_stats(self) -> Dict:
        """Get current device statistics."""
        stats = {
            'power_mode': self.power_mode.value,
            'clock_frequency': self._get_clock_frequency(),
            'buffer_count': len(self._circular_buffers),
            'temperature': self._read_temperature()
        }

        return stats

    def cleanup(self) -> None:
        """Release FPGA resources."""
        if self.overlay:
            self.overlay.free()

    def _setup_dma_buffers(self, size: int) -> None:
        """Initialize DMA buffer pool."""
        # Pre-allocate some buffers
        for _ in range(4):  # Start with 4 buffers
            buffer = pynq.allocate(shape=(size,), dtype=np.float32)
            self._circular_buffers[size] = buffer

    def _get_input_buffer(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Get or create input DMA buffer."""
        size = np.prod(shape)
        if size not in self._circular_buffers:
            self._circular_buffers[size] = pynq.allocate(shape=shape, dtype=np.float32)
        return self._circular_buffers[size]

    def _get_output_buffer(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Get or create output DMA buffer."""
        return self._get_input_buffer(shape)  # Reuse same-sized buffers

    def _set_clock_frequency(self, freq_mhz: float) -> None:
        """Set FPGA clock frequency."""
        try:
            self.overlay.clock.fclk0_mhz = freq_mhz
        except Exception as e:
            print(f"Failed to set clock frequency: {e}")

    def _get_clock_frequency(self) -> float:
        """Get current clock frequency in MHz."""
        return self.overlay.clock.fclk0_mhz

    def _read_temperature(self) -> float:
        """Read FPGA die temperature."""
        try:
            return self.overlay.uio.read(0x0)  # Assumes temperature sensor at offset 0
        except:
            return 0.0
