"""
Main application entry point for BCI compression pipeline orchestration.
Initializes pipeline, loads configuration, and runs main loop.

References:
- PipelineConnector
- signal_processing, gpu_helper
"""

# To run: python -m src.main from the project root
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.visualization.web_dashboard.pipeline_connector import PipelineConnector
from src.utils import signal_processing, gpu_helper, stream_processor
from src.utils import artifact_simulator, data_format_handler, performance_monitor


def main():
    connector = PipelineConnector()
    # Simulate neural data
    data = np.random.normal(0, 1, (64, 1000))
    # Artifact simulation examples
    spike_signal = artifact_simulator.inject_spike(data, severity=0.5)
    noise_signal = artifact_simulator.inject_noise(data, severity=0.5)
    drift_signal = artifact_simulator.inject_drift(data, severity=0.5)
    print("Spike, Noise, Drift signals computed.")
    # Performance monitoring example
    latency = performance_monitor.measure_latency(connector.get_live_metrics, 64, 1000)
    throughput = performance_monitor.get_throughput(1000, latency)
    mem_usage = performance_monitor.get_memory_usage()
    print(f"Latency: {latency:.2f} ms, Throughput: {throughput:.2f} samples/s, Memory: {mem_usage:.2f} MB")
    # Data format handler stubs
    nev_data = data_format_handler.load_nev("/path/to/file.nev")
    nsx_data = data_format_handler.load_nsx("/path/to/file.nsx")
    hdf5_data = data_format_handler.load_hdf5("/path/to/file.h5")
    print("Loaded NEV, NSx, HDF5 data (stub arrays).")
    print("Streaming windows:")
    for window, idx in stream_processor.stream_processor(data, buffer_size=256, overlap=64):
        metrics = connector.get_live_metrics(num_channels=window.shape[0], sample_size=window.shape[1])
        print(f"Window {idx}: Metrics {metrics}")
    print("FFT shape:", signal_processing.apply_fft(data).shape)
    print("Filtered shape:", signal_processing.apply_iir_filter(data, b=np.array([0.2, 0.8]), a=np.array([1.0, -0.5])).shape)
    coeffs, reconstructed = signal_processing.apply_wavelet_transform(data)
    print("Wavelet coeffs length:", len(coeffs))
    print("Wavelet reconstructed shape:", reconstructed.shape)
    # GPU acceleration
    gpu_data = gpu_helper.to_gpu(data)
    print("Dot product shape:", gpu_helper.gpu_dot(gpu_data, gpu_data.T).shape)
    # Pipeline metrics
    metrics = connector.get_live_metrics(num_channels=64, sample_size=1000)
    print("Pipeline Metrics:", metrics)
    # Compression simulation
    ratio = connector.simulate_compression(100000, 25000)
    print("Compression Ratio:", ratio)
    print("Noisy signal shape:", connector.inject_artifacts(data, artifact_type="spike", severity=0.5).shape)
    # Multi-modal fusion (example)
    fmri = np.random.normal(0, 1, (64, 1000))
    fused = connector.simulate_multimodal_fusion(data, fmri)
    print("Fused shape:", fused.shape)


if __name__ == "__main__":
    main()
