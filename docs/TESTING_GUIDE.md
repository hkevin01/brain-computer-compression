# Comprehensive Testing and Validation Guide

This guide provides detailed information about testing and validating the Brain-Computer Compression Toolkit to ensure all components work as intended.

## Overview

The toolkit includes multiple testing levels to accommodate different needs:

- **Quick Tests** (2 minutes): Essential functionality verification
- **Standard Tests** (10 minutes): Development and CI/CD validation
- **Comprehensive Tests** (30 minutes): Full feature and performance validation

## Test Suite Components

### 1. Simple Unit Tests (`test_simple_validation.py`)

**Purpose**: Basic functionality verification
**Duration**: ~2 minutes
**Coverage**: Core algorithms, basic compression/decompression, plugin system

```bash
python tests/test_simple_validation.py
```

**Tests Included**:
- Neural LZ compression functionality
- Perceptual quantization basic operation
- EMG compression algorithms (LZ, Perceptual, Predictive, Mobile)
- Quality metrics calculation
- Plugin system discovery and creation
- Mobile optimization features

**Expected Results**:
```
Test Results:
- TestNeuralAlgorithms: 2/2 tests passed
- TestEMGAlgorithms: 3/3 tests passed
- TestQualityMetrics: 2/2 tests passed
- TestPluginSystem: 2/2 tests passed
- TestMobileOptimization: 1/1 tests passed

Overall: ✅ ALL TESTS PASSED
```

### 2. Performance Benchmark (`test_performance_benchmark.py`)

**Purpose**: Validate performance claims against actual measurements
**Duration**: ~5-10 minutes
**Coverage**: Compression ratios, latency, quality scores, scalability

```bash
python tests/test_performance_benchmark.py
```

**Performance Targets Validated**:

#### Neural Algorithms
- **Neural LZ**: 1.5-3x compression, <1ms latency
- **Perceptual**: 2-10x compression, 15-25dB SNR, <1ms latency

#### EMG Algorithms
- **EMG LZ**: 5-12x compression, 0.85-0.95 quality, <25ms latency
- **EMG Perceptual**: 8-20x compression, 0.90-0.98 quality, <35ms latency
- **EMG Predictive**: 10-25x compression, 0.88-0.96 quality, <50ms latency
- **Mobile EMG**: 3-8x compression, 0.80-0.90 quality, <15ms latency

**Expected Results**:
```
Neural LZ: 2.1x ratio, 0.8ms - PASS
Perceptual: 4.5x ratio, 18.2dB SNR, 0.7ms - PASS
EMG LZ: 8.3x ratio, Q=0.91, 18.4ms - PASS
EMG Perceptual: 12.1x ratio, Q=0.94, 28.2ms - PASS
...
Success Rate: 95.2%
🎉 PERFORMANCE BENCHMARK PASSED!
```

### 3. Comprehensive Validation (`test_comprehensive_validation_clean.py`)

**Purpose**: Exhaustive testing including edge cases and stress tests
**Duration**: ~15-30 minutes
**Coverage**: Performance validation, memory stress, latency stress, robustness

```bash
python tests/test_comprehensive_validation_clean.py
```

**Components**:
- **Performance Validation**: Detailed algorithm performance measurement
- **Quality Metrics Validation**: EMG quality metrics functionality
- **Plugin System Validation**: Complete plugin discovery and usage
- **Mobile Optimization Validation**: Power optimization scenarios
- **Latency Stress Test**: Real-time performance under load
- **Robustness Test**: Edge cases (zeros, infinities, NaN values)

## Test Runner (`run_tests.py`)

The test runner provides a unified interface for all testing:

```bash
# Quick tests (2 minutes)
python tests/run_tests.py quick

# Standard tests (10 minutes)
python tests/run_tests.py standard

# Comprehensive tests (30 minutes)
python tests/run_tests.py comprehensive

# Individual test suites
python tests/run_tests.py --test simple
python tests/run_tests.py --test performance

# Dependency check only
python tests/run_tests.py --dependencies-only
```

## Using Make Commands

For convenience, use the provided Makefile:

```bash
# Quick tests
make test-quick

# Standard tests (recommended)
make test

# Comprehensive tests
make test-comprehensive

# Individual components
make test-simple
make test-performance
make benchmark
make validate

# Development workflow
make dev-check        # Dependencies + lint + quick tests
make ci              # CI/CD simulation
```

## Performance Validation Details

### Neural Algorithm Validation

The toolkit validates the following neural performance claims:

1. **Neural LZ Compression**
   - Target: 1.5-3x compression ratio
   - Target: <1ms latency
   - Test: 64-channel, 30k-sample neural data

2. **Perceptual Quantization**
   - Target: 2-10x compression ratio
   - Target: 15-25dB SNR
   - Target: <1ms latency
   - Test: Quality-controlled quantization

### EMG Algorithm Validation

1. **EMG LZ Compressor**
   - Target: 5-12x compression ratio
   - Target: 0.85-0.95 quality score
   - Target: <25ms latency
   - Test: 4-channel, 2kHz EMG data

2. **EMG Perceptual Quantizer**
   - Target: 8-20x compression ratio
   - Target: 0.90-0.98 quality score
   - Target: <35ms latency
   - Test: Clinical-grade compression

3. **EMG Predictive Compressor**
   - Target: 10-25x compression ratio
   - Target: 0.88-0.96 quality score
   - Target: <50ms latency
   - Test: Biomechanical model-based

4. **Mobile EMG Compressor**
   - Target: 3-8x compression ratio
   - Target: 0.80-0.90 quality score
   - Target: <15ms latency
   - Test: Power-optimized compression

## Quality Metrics Validation

### EMG Quality Metrics

The toolkit validates EMG-specific quality metrics:

1. **Muscle Activation Detection**
   - Precision/Recall for activation events
   - Target: >90% accuracy

2. **Envelope Correlation**
   - Critical for prosthetic control
   - Target: >0.95 correlation

3. **Spectral Fidelity**
   - Frequency domain preservation
   - Target: High correlation in EMG bands (20-500Hz)

4. **Timing Precision**
   - Temporal accuracy for real-time applications
   - Target: <5ms timing errors

## Stress Testing

### Memory Stress Tests

Tests algorithm performance with large datasets:

- 32-1024 channels
- 10k-120k samples
- Up to several GB of data
- Memory efficiency validation

### Latency Stress Tests

Tests real-time performance:

- Small data chunks (25-500 samples)
- 100 iterations per chunk size
- Latency distribution analysis
- Consistency validation

### Robustness Tests

Tests algorithm behavior with edge cases:

- All-zero data
- All-ones data
- Very large values (1000x normal)
- Very small values (0.001x normal)
- Infinite values (should handle gracefully)
- NaN values (should handle gracefully)

## Troubleshooting Test Failures

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Ensure toolkit is installed
   pip install -e .

   # Check dependencies
   python tests/run_tests.py --dependencies-only
   ```

2. **Performance Test Failures**
   - **Slow System**: Performance targets may need adjustment for slower hardware
   - **Memory Issues**: Reduce test data sizes in benchmark files
   - **CPU Load**: Close other applications during testing

3. **Algorithm Not Found**
   ```bash
   # Check if modules are properly installed
   python -c "import bci_compression.algorithms; print('OK')"
   python -c "import bci_compression.algorithms.emg_compression; print('OK')"
   ```

4. **Numeric Issues**
   - **NaN/Inf Results**: Check input data ranges and algorithm parameters
   - **Low Quality Scores**: May indicate algorithm implementation issues

### Debug Mode

For detailed debugging, modify test files to add verbose output:

```python
# In test files, add:
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in algorithm tests
print(f"Input data shape: {data.shape}")
print(f"Input data range: {data.min():.3f} to {data.max():.3f}")
```

## Continuous Integration

### CI/CD Pipeline

For automated testing in CI/CD systems:

```yaml
# Example GitHub Actions workflow
- name: Setup and Test
  run: |
    python tests/run_tests.py --dependencies-only
    python tests/run_tests.py standard

# Or using Make
- name: CI Pipeline
  run: make ci
```

### Test Reports

All tests generate detailed JSON reports:

- `test_results/` - Test runner reports
- `validation_results/` - Comprehensive validation reports
- `benchmark_results/` - Performance benchmark reports

## Custom Test Configuration

### Modifying Performance Targets

Edit performance targets in test files:

```python
# In test_performance_benchmark.py
self.performance_claims = {
    'neural': {
        'neural_lz': {'ratio': (1.5, 3.0), 'latency_ms': 1.0},
        # Modify targets as needed
    }
}
```

### Adding Custom Tests

Create new test files following the pattern:

```python
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestCustomFeature(unittest.TestCase):
    def test_custom_functionality(self):
        # Your test code here
        pass

if __name__ == "__main__":
    unittest.main()
```

## Interpreting Test Results

### Success Criteria

- **Unit Tests**: All individual tests pass
- **Performance Tests**: 80%+ of performance targets met
- **Comprehensive Tests**: 85%+ overall success rate
- **Quality Metrics**: All metric calculations complete without errors

### Performance Interpretation

- **Compression Ratios**: Higher is better, but quality matters
- **Latency**: Lower is better, must meet real-time requirements
- **Quality Scores**: Higher is better (0.0-1.0 scale)
- **SNR**: Higher is better (dB scale)

## Conclusion

This comprehensive testing framework ensures that:

1. **All algorithms work correctly** - Unit tests verify basic functionality
2. **Performance claims are accurate** - Benchmarks validate README claims
3. **System is robust** - Stress tests ensure reliability
4. **Quality is maintained** - Metrics validate signal preservation
5. **Real-time requirements are met** - Latency tests ensure applicability

The testing framework is designed to catch issues early and provide confidence that the toolkit works as advertised across different systems and use cases.
