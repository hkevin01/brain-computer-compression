# Long-Running Integration Tests

This directory contains comprehensive integration tests that take longer to run.

## Purpose

- Full-scale benchmarking with large datasets
- End-to-end pipeline validation
- Performance regression testing
- Comprehensive validation scenarios

## Running These Tests

### Run all long tests:
```bash
pytest tests/long/ -v
```

### Run specific test file:
```bash
pytest tests/long/test_comprehensive_benchmark.py -v
```

### Include long tests in full test suite:
```bash
pytest tests/ -m "not quick"  # Runs everything except quick tests
pytest tests/                 # Runs all tests including slow ones
```

## Test Organization

- `test_comprehensive_benchmark.py` - Full benchmark suite
- `test_integration_workflows.py` - End-to-end pipeline tests
- `test_stress_tests.py` - Stress and load testing

## Quick vs Long Tests

**Quick tests** (`tests/*.py`):
- Run in <30 seconds total
- Small synthetic datasets
- Smoke tests and basic validation
- Run on every commit

**Long tests** (`tests/long/*.py`):
- May take minutes to complete
- Large datasets, multiple runs
- Comprehensive validation
- Run before releases or on-demand

## Marking Tests

Use pytest markers:
```python
import pytest

@pytest.mark.slow
def test_comprehensive_pipeline():
    # Long-running test
    pass

@pytest.mark.quick
def test_basic_functionality():
    # Fast test
    pass
```
