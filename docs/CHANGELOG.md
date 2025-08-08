# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-01-XX

### Added
- **Mobile Module**: Complete mobile-optimized compression library for BCI devices
  - `MobileBCICompressor`: Lightweight, power-efficient compression algorithms
  - `MobileStreamingPipeline`: Real-time streaming with bounded memory usage
  - `PowerOptimizer`: Dynamic power/quality trade-off management
  - `MobileMetrics`: Mobile-specific performance and quality metrics
  - `AdaptiveQualityController`: Real-time quality adjustment
- Enhanced compression algorithms with improved pattern detection and quantization
- Comprehensive test suite for mobile module (6/6 tests passing)
- Mobile module documentation and integration examples

### Improved
- Enhanced LZ compression with better pattern recognition
- Improved lightweight quantization with adaptive bit allocation (8-12 bits)
- Fast prediction compression with autocorrelation-based coefficients
- Multi-channel support for all mobile algorithms
- Reduced dithering intensity for better SNR in quantization

### Performance
- Mobile compression ratios: 1.5-4x (LZ), 2-8x (quantization), 1.5-3x (prediction)
- Processing latency: < 1ms for mobile algorithms
- Memory usage: < 50MB for typical configurations
- SNR range: -10dB to +15dB depending on algorithm and settings

## [0.9.0] - 2024-01-XX

### Added
- PSNR metric implementation and testing
- Enhanced error handling in benchmark runner
- Cross-platform CI workflow for Linux, macOS, and Windows
- Comprehensive test suite with all tests passing
