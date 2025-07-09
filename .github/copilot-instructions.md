<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Brain-Computer Interface Data Compression Toolkit - Copilot Instructions

## Project Context

This is a Brain-Computer Interface (BCI) data compression toolkit focused on developing and benchmarking compression algorithms for neural data streams. The project emphasizes real-time processing, GPU acceleration, and signal processing techniques.

## Technical Focus Areas

### Signal Processing

- Use NumPy and SciPy for mathematical operations
- Implement FFT and IIR filters for frequency domain processing
- Apply wavelet transforms for time-frequency analysis
- Ensure all signal processing maintains data integrity for neural signals

### Compression Algorithms

- Prioritize lossless compression for critical neural data
- Implement lossy compression with configurable quality levels
- Consider temporal correlations in neural data streams
- Optimize for real-time processing with minimal latency

### GPU Acceleration

- Use CuPy for GPU-accelerated NumPy operations
- Implement CUDA kernels for custom operations when needed
- Ensure fallback to CPU implementations for compatibility
- Profile GPU memory usage and optimize for streaming data

### Performance Requirements

- Target < 1ms latency for real-time applications
- Optimize memory usage for continuous data streams
- Implement efficient buffering for streaming scenarios
- Use appropriate data types to minimize memory footprint

## Code Style Guidelines

### Python Standards

- Follow PEP 8 for code formatting
- Use type hints for all function signatures
- Implement comprehensive error handling for signal processing
- Add detailed docstrings for all public methods

### Documentation

- Include mathematical formulas in docstrings where relevant
- Provide usage examples for compression algorithms
- Document performance characteristics and trade-offs
- Reference relevant research papers in comments

### Testing

- Write unit tests for all compression algorithms
- Include integration tests for end-to-end pipelines
- Test with synthetic and real neural data
- Benchmark performance against baseline implementations

## Domain-Specific Considerations

### Neural Data Characteristics

- Account for multi-channel recordings (32-256+ channels)
- Handle varying sampling rates (1kHz - 30kHz typical)
- Consider noise characteristics in neural recordings
- Preserve spatial and temporal relationships between channels

### BCI Applications

- Prioritize low-latency processing for real-time control
- Maintain signal quality for feature extraction
- Support common BCI data formats (NEV, NSx, HDF5)
- Enable configurable compression parameters for different use cases

### Research Reproducibility

- Use fixed random seeds for reproducible results
- Log compression parameters and performance metrics
- Save intermediate processing steps for analysis
- Provide clear benchmarking methodologies

## Common Patterns

### Data Processing Pipeline

```python
# Standard pipeline structure
def process_neural_data(data, config):
    # Preprocessing (filtering, normalization)
    # Compression algorithm application
    # Performance metric calculation
    # Results logging
```

### GPU Memory Management

```python
# Memory-efficient GPU operations
with cupy.cuda.Device(device_id):
    # Allocate memory pools
    # Process data in chunks
    # Clean up memory explicitly
```

### Real-time Streaming

```python
# Streaming data processing
def stream_processor(buffer_size, overlap):
    # Implement sliding window processing
    # Handle buffer management
    # Maintain state between chunks
```

This project contributes to the advancement of brain-computer interfaces through efficient data compression techniques.
