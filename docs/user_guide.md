# User Guide: BCI Compression Toolkit

## Getting Started
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest tests/`
- Generate synthetic data: `python scripts/data_generator.py`

## Running Compression
- Import and configure desired compressor from `src/compression`
- Example:
```python
from src.compression import NeuralLZ77Compressor
compressor = NeuralLZ77Compressor()
compressed = compressor.compress(data)
```

## Benchmarking
- Run benchmarking suite: `python scripts/benchmark_runner.py`
- View results in `docs/benchmarks_report.md`

## Troubleshooting
- See error messages in logs
- Check dependencies (NumPy, CuPy, psutil)
- For GPU issues, verify CUDA installation

## Tutorials
- See `notebooks/compression_analysis.ipynb` for step-by-step examples
- See `notebooks/benchmarking_results.ipynb` for benchmarking walkthrough

## Support
- Open an issue on GitHub for help or feature requests
