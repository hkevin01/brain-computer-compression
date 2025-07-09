# Contributing to BCI Data Compression Toolkit

We welcome contributions to the Brain-Computer Interface Data Compression Challenge Toolkit! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/brain-computer-compression.git
   cd brain-computer-compression
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Code Standards

### Python Style
- Follow PEP 8 for code formatting
- Use type hints for all function signatures
- Maximum line length: 79 characters
- Use descriptive variable and function names

### Documentation
- Add docstrings to all public functions and classes
- Include parameter types and descriptions
- Provide usage examples where appropriate
- Reference relevant research papers in comments

### Testing
- Write unit tests for all new functionality
- Ensure test coverage > 90%
- Test with both synthetic and real neural data
- Include performance benchmarks

## Compression Algorithm Contributions

When contributing new compression algorithms:

1. **Inherit from BaseCompressor**: All algorithms should inherit from the `BaseCompressor` class
2. **Implement required methods**: `compress()`, `decompress()`, and optionally `fit()`
3. **Optimize for neural data**: Consider temporal correlations and multi-channel structure
4. **Document performance**: Include compression ratio, latency, and quality metrics
5. **Add benchmarks**: Include the algorithm in benchmark comparisons

### Example Algorithm Structure

```python
from bci_compression.core import BaseCompressor
import numpy as np

class MyCompressionAlgorithm(BaseCompressor):
    """
    Brief description of the algorithm.
    
    Parameters
    ----------
    param1 : type
        Description of parameter.
    """
    
    def __init__(self, param1: int = 10):
        super().__init__()
        self.param1 = param1
    
    def compress(self, data: np.ndarray) -> bytes:
        """Compress neural data."""
        # Implementation here
        pass
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress neural data."""
        # Implementation here
        pass
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes following the code standards
3. Add or update tests as needed
4. Update documentation if necessary
5. Run the test suite: `pytest tests/`
6. Run code quality checks:
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```
7. Submit a pull request with a clear description

## Issue Reporting

When reporting issues:

- Use the issue templates provided
- Include system information (OS, Python version, GPU if applicable)
- Provide minimal reproducible examples
- Include error messages and stack traces
- Describe expected vs actual behavior

## Research Contributions

We encourage contributions from the research community:

- Novel compression algorithms for neural data
- Improved benchmarking methodologies
- New evaluation metrics
- Real-world dataset contributions
- Performance optimizations

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Follow the project's technical standards
- Acknowledge contributions from others

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes
- Research publications (when applicable)
- The project website

Thank you for contributing to advancing brain-computer interface technology!
