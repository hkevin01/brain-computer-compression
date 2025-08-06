# Makefile for Brain-Computer Compression Toolkit

.PHONY: help install test test-quick test-standard test-comprehensive clean lint format check-deps benchmark

# Default target
# Universal Development Makefile
# Provides consistent commands across all environments and projects

.PHONY: help dev-start dev-stop dev-reset dev-status dev-logs test lint build deploy clean

# Default target
help: ## Show this help message
	@echo "🧠 BCI Compression Toolkit - Universal Development Commands"
	@echo "=========================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s
", $$1, $$2}' $(MAKEFILE_LIST)

# ============================================================================
# Development Environment Management
# ============================================================================

dev-start: ## Start all development containers
	@echo "🚀 Starting development environment..."
	@docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"
	@echo "📊 Services available:"
	@echo "   • Jupyter Lab: http://localhost:8888"
	@echo "   • Frontend: http://localhost:3000"
	@echo "   • Backend API: http://localhost:8000"
	@echo "   • Grafana: http://localhost:3001"
	@echo "   • PostgreSQL: localhost:5432"
	@make dev-status

dev-stop: ## Stop all development containers
	@echo "🛑 Stopping development environment..."
	@docker-compose -f docker-compose.dev.yml down
	@echo "✅ Development environment stopped!"

dev-reset: ## Reset development environment (removes volumes)
	@echo "🔄 Resetting development environment..."
	@docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Development environment reset!"

dev-status: ## Show status of all development containers
	@echo "📊 Development Environment Status:"
	@docker-compose -f docker-compose.dev.yml ps

dev-logs: ## Show logs from all development containers
	@docker-compose -f docker-compose.dev.yml logs -f

dev-shell: ## Open shell in backend development container
	@docker-compose -f docker-compose.dev.yml exec backend bash

dev-tools-shell: ## Open shell in tools container
	@docker-compose -f docker-compose.dev.yml exec tools bash

# ============================================================================
# Original Testing Commands (preserved)
# ============================================================================

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
test: ## Run standard tests in containers
	@echo "🧪 Running standard test suite..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec backend python -m pytest tests/ -v --cov=src/bci_compression --cov-report=html --cov-report=term; \
	else \
		python tests/run_tests.py standard; \
	fi
	@echo "✅ Standard tests completed!"

test-quick: ## Run quick validation tests
	@echo "⚡ Running quick validation tests..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec backend python tests/test_simple_validation.py; \
	else \
		python tests/run_tests.py quick; \
	fi
	@echo "✅ Quick tests completed!"

test-standard:
	cd tests && python run_tests.py standard

test-comprehensive: ## Run comprehensive tests
	@echo "🔬 Running comprehensive test suite..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec backend python tests/run_tests.py comprehensive; \
	else \
		python tests/run_tests.py comprehensive; \
	fi
	@echo "✅ Comprehensive tests completed!"

test-emg: ## Run EMG-specific tests
	@echo "🏥 Running EMG compression tests..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec backend python tests/test_emg_integration.py; \
	else \
		python tests/test_emg_integration.py; \
	fi
	@echo "✅ EMG tests completed!"

test-performance: ## Run performance benchmarks
	@echo "📊 Running performance benchmarks..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec backend python tests/test_performance_benchmark.py; \
	else \
		python tests/test_performance_benchmark.py; \
	fi
	@echo "✅ Performance benchmarks completed!"

# Validation commands
validate:
	cd tests && python test_comprehensive_validation_clean.py

# Code quality commands
lint: ## Run code quality checks
	@echo "🔍 Running code quality checks..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec tools black --check /workspace/src/; \
		docker-compose -f docker-compose.dev.yml exec tools flake8 /workspace/src/; \
		docker-compose -f docker-compose.dev.yml exec tools pylint /workspace/src/bci_compression/; \
	else \
		echo "Running local linting..."; \
		python -m black --check src/; \
		python -m flake8 src/; \
	fi
	@echo "✅ Code quality checks completed!"

format:
	black src/ tests/ --line-length=88

lint-fix: ## Fix code formatting issues
	@echo "🔧 Fixing code formatting..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec tools black /workspace/src/; \
		docker-compose -f docker-compose.dev.yml exec tools isort /workspace/src/; \
	else \
		python -m black src/; \
		python -m isort src/; \
	fi
	@echo "✅ Code formatting fixed!"

security-scan: ## Run security vulnerability scans
	@echo "🔒 Running security scans..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml exec tools bandit -r /workspace/src/; \
		docker-compose -f docker-compose.dev.yml exec tools safety check --file /workspace/requirements.txt; \
	else \
		echo "Install bandit and safety for security scanning"; \
	fi
	@echo "✅ Security scan completed!"

# Demo commands
demo:
	python examples/emg_demo.py

# Build and Deployment
build: ## Build production containers
	@echo "🏗️ Building production containers..."
	@docker build -f docker/prod.Dockerfile -t bci-compression:latest .
	@echo "✅ Production containers built!"

build-dev: ## Build development containers
	@echo "🏗️ Building development containers..."
	@docker-compose -f docker-compose.dev.yml build --no-cache
	@echo "✅ Development containers built!"

# ============================================================================
# Monitoring and Maintenance
# ============================================================================

monitor: ## Open monitoring dashboard
	@echo "📈 Opening monitoring dashboard..."
	@open http://localhost:3001 || xdg-open http://localhost:3001 || echo "Open http://localhost:3001 in your browser"

logs-backend: ## Show backend logs
	@docker-compose -f docker-compose.dev.yml logs -f backend

logs-frontend: ## Show frontend logs
	@docker-compose -f docker-compose.dev.yml logs -f frontend

logs-db: ## Show database logs
	@docker-compose -f docker-compose.dev.yml logs -f postgres mongodb redis

system-info: ## Show system resource usage
	@echo "💻 System Resource Usage:"
	@docker stats --no-stream
	@echo ""
	@echo "💾 Disk Usage:"
	@docker system df

# ============================================================================
# Cleanup and Maintenance
# ============================================================================

clean: ## Clean up development environment
	@echo "🧹 Cleaning up development environment..."
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml down; \
		docker system prune -f; \
	fi
	@echo "✅ Cleanup completed!"

clean-all: ## Remove all containers, images, and volumes
	@echo "🧹 Performing deep cleanup..."
	@echo "⚠️  This will remove ALL Docker containers, images, and volumes. Are you sure? [y/N]"
	@read -r CONFIRM && [ "$$CONFIRM" = "y" ] || exit 1
	@docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	@docker system prune -a -f --volumes
	@echo "✅ Deep cleanup completed!"

# ============================================================================
# Development Workflow Shortcuts
# ============================================================================

dev: ## Quick development setup (start + install + test)
	@make dev-start
	@sleep 30
	@make dev-install
	@make test-quick
	@echo "🎉 Development environment ready!"

work: ## Start working session (with monitoring)
	@make dev-start
	@sleep 10
	@make dev-status
	@echo "💼 Work session started!"

# ============================================================================
# CI/CD and Automation
# ============================================================================

ci: ## Run CI/CD pipeline locally
	@echo "🔄 Running CI/CD pipeline..."
	@if [ -f docker-compose.dev.yml ]; then \
		make build-dev; \
		make dev-start; \
		sleep 30; \
		make test; \
		make lint; \
		make security-scan; \
		make dev-stop; \
	else \
		make test; \
		make lint; \
	fi
	@echo "✅ CI/CD pipeline completed!"

# ============================================================================
# Legacy Commands (preserved for backward compatibility)
# ============================================================================

validate: ## Run validation suite
	@make test-comprehensive

benchmark: ## Run benchmarks
	@make test-performance

dev-check: ## Quick development check
	@make test-quick
