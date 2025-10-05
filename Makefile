# Makefile for Brain-Computer Compression Toolkit

.PHONY: help install test test-quick test-standard test-comprehensive clean lint format check-deps benchmark

# Default target
help: ## Show this help message
	@echo "🧠 BCI Compression Toolkit - Development Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ============================================================================
# Development Environment Management
# ============================================================================

dev-start: ## Start development environment
	@echo "🚀 Starting development environment..."
	@./run.sh up
	@echo "✅ Development environment started!"

dev-stop: ## Stop development environment
	@echo "🛑 Stopping development environment..."
	@./run.sh down
	@echo "✅ Development environment stopped!"

dev-status: ## Show system and service status
	@./run.sh status

dev-logs: ## Show service logs
	@./run.sh logs

dev-shell: ## Open shell in backend development container
	@docker-compose -f docker-compose.dev.yml exec backend bash

dev-tools-shell: ## Open shell in tools container
	@docker-compose -f docker-compose.dev.yml exec tools bash

# ============================================================================
# Original Testing Commands (preserved)
# ============================================================================

# Installation commands
setup: ## Install development environment
	@echo "🔧 Setting up development environment..."
	@python -m venv venv || echo "Virtual environment already exists"
	@echo "📦 Installing dependencies..."
	@./venv/bin/pip install --upgrade pip setuptools wheel
	@./venv/bin/pip install -e ".[dev,quality]"
	@echo "✅ Development environment ready!"
	@echo "💡 Activate with: source venv/bin/activate"

install: ## Install package with core dependencies
	@pip install -e .

install-dev: ## Install package with development dependencies
	@pip install -e ".[dev,quality]"

check-deps:
	cd tests && python run_tests.py --dependencies-only

# Testing commands
test: ## Run standard tests with pytest
	@echo "🧪 Running test suite..."
	@pytest tests/ -v --cov=src/bci_compression --cov-report=html --cov-report=term
	@echo "✅ Tests completed!"

test-quick: ## Run quick validation tests
	@echo "⚡ Running quick validation tests..."
	@pytest tests/ -v -k "not slow"
	@echo "✅ Quick tests completed!"

benchmark: ## Run compression benchmarks
	@echo "� Running benchmarks..."
	@./run.sh bench:all
	@echo "✅ Benchmarks completed!"

# Code quality commands
lint: ## Run code quality checks
	@echo "🔍 Running code quality checks..."
	@ruff check src/ tests/
	@black --check src/ tests/
	@mypy src/
	@echo "✅ Code quality checks completed!"

format: ## Format code with black and ruff
	@echo "🔧 Formatting code..."
	@black src/ tests/
	@ruff check --fix src/ tests/
	@echo "✅ Code formatting completed!" \
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

clean: ## Clean up development artifacts
	@echo "🧹 Cleaning up development artifacts..."
	@rm -rf .pytest_cache __pycache__ .coverage htmlcov .mypy_cache .ruff_cache
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@./run.sh clean 2>/dev/null || echo "Docker cleanup skipped"
	@echo "✅ Cleanup completed!"

clean-all: ## Deep cleanup including Docker resources
	@echo "🧹 Performing deep cleanup..."
	@make clean
	@docker system prune -a -f --volumes 2>/dev/null || echo "Docker deep cleanup skipped"
	@echo "✅ Deep cleanup completed!"
	@echo "✅ Deep cleanup completed!"

# ============================================================================
# Development Workflow Shortcuts
# ============================================================================

dev: ## Quick development setup (setup + start + test)
	@make setup
	@make dev-start
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
# CI/CD and Validation
# ============================================================================

ci: ## Run CI/CD pipeline locally
	@echo "🔄 Running CI/CD pipeline..."
	@make test
	@make lint
	@echo "✅ CI/CD pipeline completed!"

validate: ## Run validation suite
	@make test
	@make lint
	@echo "✅ Validation completed!"

check: ## Quick development check
	@make test-quick
	@make lint
	@echo "✅ Development check completed!"
