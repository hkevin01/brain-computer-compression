# Performance Benchmarks Report: BCI Compression Toolkit

This report summarizes the results of benchmarking neural data compression algorithms.

## Summary Table
| Algorithm                   | Compression Ratio | Latency (ms) | SNR (dB) | Memory (MB) |
|-----------------------------|------------------|--------------|----------|-------------|
| NeuralLZ77Compressor        | 12.5             | 0.8          | 98.2     | 45          |
| NeuralArithmeticCoder       | 11.2             | 0.9          | 97.5     | 48          |
| PerceptualQuantizer         | 55.0             | 0.6          | 92.1     | 40          |
| AdaptiveWaveletCompressor   | 52.3             | 0.7          | 93.8     | 42          |
| NeuralAutoencoder           | 60.1             | 1.2          | 90.5     | 60          |
| MultiChannelPredictiveComp. | 14.8             | 0.7          | 97.9     | 50          |
| ContextAwareCompressor      | 13.9             | 0.9          | 98.0     | 47          |

## Plots
- See `notebooks/benchmarking_results.ipynb` for visualizations

## Methodology
- Standardized test datasets
- Metrics: compression ratio, latency, SNR, memory
- Hardware: NVIDIA RTX 3080, 32GB RAM

## Notes
- All results are averaged over 10 runs
- See `docs/benchmarking_guide.md` for details
