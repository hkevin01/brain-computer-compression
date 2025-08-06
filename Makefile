# Makefile for Brain-Computer Compression Toolkit

.PHONY: help install test test-quick test-standard test-comprehensive clean lint format check-deps benchmark

# Default target
help:
	@echo "Brain-Computer Compression Toolkit - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install the toolkit and dependencies"
	@echo "  install-dev      Install in development mode with dev dependencies"
	@echo "  check-deps       Check if all dependencies are available"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run standard test suite (~10 minutes)"
	@echo "  test-quick       Run quick unit tests (~2 minutes)"
	@echo "  test-standard    Run standard tests with benchmarks (~10 minutes)"
	@echo "  test-comprehensive Run comprehensive validation (~30 minutes)"
	@echo "  test-simple      Run only simple unit tests"
	@echo "  test-performance Run only performance benchmarks"
	@echo "  benchmark        Run detailed performance benchmarks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run code linting checks"
	@echo "  format           Format code with black"
	@echo "  clean            Clean up temporary files and caches"
	@echo ""
	@echo "Development Commands:"
	@echo "  demo             Run EMG demo example"
	@echo "  validate         Run comprehensive validation suite"
	@echo ""

# Installation commands
install:
	pip install -r requirements.txt
	pip install -r requirements-emg.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-emg.txt
	pip install -e ".[dev]"

check-deps:
	cd tests && python run_tests.py --dependencies-only

# Testing commands
test: test-standard

test-quick:
	cd tests && python run_tests.py quick

test-standard:
	cd tests && python run_tests.py standard

test-comprehensive:
	cd tests && python run_tests.py comprehensive

test-simple:
	cd tests && python run_tests.py --test simple

test-performance:
	cd tests && python run_tests.py --test performance

benchmark:
	cd tests && python test_performance_benchmark.py

# Validation commands
validate:
	cd tests && python test_comprehensive_validation_clean.py

# Code quality commands
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black src/ tests/ --line-length=88

# Demo commands
demo:
	python examples/emg_demo.py

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/
	rm -rf test_results/ validation_results/ benchmark_results/

# Advanced testing
test-all: clean test-comprehensive validate benchmark
	@echo "All tests completed!"

# CI/CD simulation
ci:
	make check-deps
	make lint
	make test-standard
	@echo "CI pipeline completed successfully!"

# Development workflow
dev-check:
	make check-deps
	make lint
	make test-quick
	@echo "Development checks passed!"
