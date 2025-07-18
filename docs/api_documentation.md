# API Reference

## Core Algorithms

### Neural LZ Compression

#### `NeuralLZ77Compressor`

LZ77 variant optimized for neural signal characteristics.

```python
class NeuralLZ77Compressor:
    def __init__(self, window_size=4096, lookahead_size=256, quantization_bits=12):
        """
        Initialize Neural LZ77 compressor.

        Args:
            window_size: History window size
            lookahead_size: Lookahead buffer size
            quantization_bits: Quantization precision (8-16 bits)
        """

    def compress_channel(self, data):
        """
        Compress single channel data.

        Args:
            data: 1D numpy array of neural signal

        Returns:
            compressed: Compressed byte data
            metadata: Compression metadata
        """

    def decompress_channel(self, compressed, metadata):
        """Decompress single channel data."""
```

#### Factory Functions

```python
def create_neural_lz_compressor(mode='balanced'):
    """
    Create neural LZ compressor with preset configuration.

    Args:
        mode: 'speed', 'balanced', or 'compression'

    Returns:
        Configured compressor instance
    """
```

### Arithmetic Coding

#### `NeuralArithmeticCoder`

Context-aware arithmetic coding for neural signals.

```python
class NeuralArithmeticCoder:
    def __init__(self, quantization_bits=12, context_length=4):
        """
        Initialize neural arithmetic coder.

        Args:
            quantization_bits: Signal quantization precision
            context_length: Context model depth
        """

    def encode(self, data):
        """
        Encode neural signal data.

        Args:
            data: Input signal array

        Returns:
            encoded: Compressed byte data
            metadata: Encoding metadata
        """
```

### Lossy Compression

#### `PerceptualQuantizer`

Frequency-domain perceptual quantization.

```python
class PerceptualQuantizer:
    def __init__(self, base_bits=10, frequency_bands=None):
        """
        Initialize perceptual quantizer.

        Args:
            base_bits: Base quantization bits
            frequency_bands: Custom frequency band definitions
        """

    def quantize(self, data, quality_level=0.8):
        """
        Apply perceptual quantization.

        Args:
            data: Multi-channel neural data
            quality_level: Quality level (0.1-1.0)

        Returns:
            quantized: Quantized data
            metadata: Quantization information
        """
```

### Predictive Compression

#### `NeuralLinearPredictor`

Linear predictive coding for neural signals.

```python
class NeuralLinearPredictor:
    def __init__(self, order=10, channels=None):
        """
        Initialize neural linear predictor.

        Args:
            order: Prediction filter order
            channels: Channel indices to process
        """

    def fit_predictor(self, data, channel_id):
        """
        Fit LPC coefficients for specific channel.

        Args:
            data: Neural signal data
            channel_id: Channel identifier

        Returns:
            Prediction coefficients and metrics
        """

    def predict_samples(self, history, channel_id):
        """Predict samples based on history."""
```

### Context-Aware Compression

#### `BrainStateDetector`

Real-time brain state classification.

```python
class BrainStateDetector:
    def __init__(self, sampling_rate=30000.0):
        """
        Initialize brain state detector.

        Args:
            sampling_rate: Signal sampling rate in Hz
        """

    def classify_state(self, data):
        """
        Classify current brain state.

        Args:
            data: Current data window

        Returns:
            Detected brain state ('rest', 'active', 'motor', 'cognitive')
        """
```

#### `ContextAwareCompressor`

Adaptive compression based on neural context.

```python
class ContextAwareCompressor:
    def __init__(self, sampling_rate=30000.0):
        """Initialize context-aware compressor."""

    def compress(self, data, window_size=None):
        """
        Compress data using context-aware methods.

        Args:
            data: Multi-channel neural data
            window_size: Processing window size

        Returns:
            compressed: List of compressed chunks
            metadata: Context metadata
        """
```

### GPU Acceleration

#### `GPUCompressionBackend`

CUDA-accelerated compression operations.

```python
class GPUCompressionBackend:
    def __init__(self, device_id=0):
        """
        Initialize GPU compression backend.

        Args:
            device_id: CUDA device ID
        """

    def gpu_bandpass_filter(self, data, low_freq, high_freq, sampling_rate):
        """Apply GPU-accelerated bandpass filter."""

    def gpu_quantization(self, data, n_bits):
        """GPU-accelerated quantization."""

    def gpu_fft_compression(self, data, compression_ratio):
        """FFT-based compression on GPU."""
```

## Data Structures

### Metadata Classes

```python
@dataclass
class CompressionMetadata:
    """Metadata for compression operations."""
    algorithm: str
    compression_ratio: float
    processing_time: float
    quality_metrics: Dict

@dataclass
class PredictionMetadata:
    """Metadata for predictive compression."""
    predictor_type: str
    prediction_accuracy: float
    original_bits: int
    compressed_bits: int

@dataclass
class ContextMetadata:
    """Metadata for context-aware compression."""
    brain_states: List[str]
    context_switches: int
    adaptation_time: float
```

## Utility Functions

### Data Generation

```python
def generate_synthetic_neural_data(n_channels, n_samples, sampling_rate):
    """Generate realistic synthetic neural data for testing."""

def load_neural_data(file_path, format='auto'):
    """Load neural data from various file formats."""
```

### Performance Evaluation

```python
def calculate_compression_metrics(original, compressed, decompressed):
    """Calculate comprehensive compression metrics."""

def benchmark_algorithm_performance(algorithm, test_data):
    """Benchmark algorithm performance."""
```

## Configuration

### Factory Presets

- **'speed'**: Optimized for lowest latency
- **'balanced'**: Balance of quality and speed
- **'compression'**: Maximum compression ratio
- **'quality'**: Highest signal fidelity

### Parameter Guidelines

- **Quantization bits**: 8-16 (higher = better quality, larger size)
- **Prediction order**: 6-16 (higher = better prediction, more computation)
- **Context depth**: 2-8 (higher = better context, more memory)
- **Window size**: 100-2000 samples (larger = better compression, higher latency)

## Error Handling

All algorithms include comprehensive error handling:

- **Input validation**: Data type and shape checking
- **Graceful degradation**: CPU fallback for GPU operations
- **Memory management**: Automatic cleanup and bounded usage
- **Exception handling**: Informative error messages

# API Documentation: BCI Compression Toolkit

This document provides a comprehensive reference for all public classes, functions, and modules in the toolkit.

## Compression Algorithms
- **NeuralLZ77Compressor**: Lossless LZ77 variant for neural data
- **NeuralArithmeticCoder**: Context-aware arithmetic coding
- **PerceptualQuantizer**: Quantization-based lossy compression
- **AdaptiveWaveletCompressor**: Transform-based compression
- **NeuralAutoencoder**: Neural network-based compression
- **NeuralLinearPredictor**: Linear predictive coding for neural signals
- **AdaptiveNeuralPredictor**: Nonlinear prediction models
- **MultiChannelPredictiveCompressor**: Multi-channel predictive compression
- **HierarchicalContextModel**: Hierarchical context modeling
- **BrainStateDetector**: Neural state-aware compression
- **SpatialContextModel**: Spatial context modeling
- **ContextAwareCompressor**: Unified context-aware compression system

## Benchmarking Tools
- **BenchmarkMetrics**: Standardized evaluation metrics
- **PerformanceProfiler**: CPU/GPU/memory profiling
- **CompressionComparison**: Comparison framework
- **RealTimeEvaluator**: Real-time streaming evaluation
- **HardwareOptimizer**: Hardware-specific optimizations

## Usage Examples
```python
from src.compression import NeuralLZ77Compressor
compressor = NeuralLZ77Compressor()
compressed = compressor.compress(data)
decompressed = compressor.decompress(compressed)
```

## Mathematical Formulas
- **Compression Ratio**: $CR = \frac{Original\ Size}{Compressed\ Size}$
- **SNR**: $SNR = 10 \log_{10} \frac{Signal\ Power}{Noise\ Power}$

## References
- See code docstrings for detailed parameter descriptions and error handling.
- For benchmarking, see `docs/benchmarking_guide.md`.

---

## Maintenance & Updates

- Monitor issues and pull requests on GitHub
- Update dependencies and documentation regularly
- Plan for feature enhancements and bug fixes
- Engage with the community for feedback and contributions

---

## Release Notes

- All phases implemented and validated
- Comprehensive documentation and benchmarking published
- Ready for community use and further research
