#!/bin/bash

# Universal Development Environment Setup Script
# Automatically detects and sets up the best development environment

set -e  # Exit on any error

echo "🧠 BCI Compression Toolkit - Universal Development Setup"
echo "======================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."
    
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker found: $DOCKER_VERSION"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            log_success "Docker daemon is running"
            return 0
        else
            log_warning "Docker daemon is not running. Please start Docker."
            return 1
        fi
    else
        log_error "Docker not found. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        return 1
    fi
}

# Check Docker Compose installation
check_docker_compose() {
    log_info "Checking Docker Compose installation..."
    
    if command_exists docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker Compose found: $COMPOSE_VERSION"
        return 0
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker compose version --short)
        log_success "Docker Compose (plugin) found: $COMPOSE_VERSION"
        # Create alias for backward compatibility
        alias docker-compose="docker compose"
        return 0
    else
        log_error "Docker Compose not found. Please install Docker Compose."
        return 1
    fi
}

# Setup Python virtual environment (fallback)
setup_python_env() {
    log_info "Setting up Python virtual environment as fallback..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_info "Python found: $PYTHON_VERSION"
        
        # Check Python version (minimum 3.8)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python version is compatible"
        else
            log_error "Python 3.8+ required. Found: $PYTHON_VERSION"
            return 1
        fi
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            log_info "Creating Python virtual environment..."
            python3 -m venv venv
            log_success "Virtual environment created"
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install dependencies
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
        if [ -f requirements-emg.txt ]; then
            pip install -r requirements-emg.txt
        fi
        pip install -e .
        
        log_success "Python environment setup complete"
        return 0
    else
        log_error "Python 3 not found. Please install Python 3.8+."
        return 1
    fi
}

# Setup development environment
setup_dev_environment() {
    local setup_mode="$1"
    
    case $setup_mode in
        "docker")
            log_info "Setting up Docker development environment..."
            
            # Build development containers
            log_info "Building development containers..."
            docker-compose -f docker-compose.dev.yml build
            
            # Start development environment
            log_info "Starting development environment..."
            docker-compose -f docker-compose.dev.yml up -d
            
            # Wait for services to be ready
            log_info "Waiting for services to start..."
            sleep 30
            
            # Install dependencies
            log_info "Installing dependencies..."
            docker-compose -f docker-compose.dev.yml exec backend pip install -e .
            docker-compose -f docker-compose.dev.yml exec backend pip install -r requirements-emg.txt
            
            # Run quick tests
            log_info "Running quick validation tests..."
            docker-compose -f docker-compose.dev.yml exec backend python tests/test_simple_validation.py
            
            log_success "Docker development environment ready!"
            echo ""
            echo "🎉 Development Environment Started!"
            echo "=================================="
            echo "📊 Available services:"
            echo "   • Jupyter Lab: http://localhost:8888"
            echo "   • Frontend: http://localhost:3000"
            echo "   • Backend API: http://localhost:8000"
            echo "   • Grafana: http://localhost:3001"
            echo "   • PostgreSQL: localhost:5432"
            echo ""
            echo "🔧 Useful commands:"
            echo "   • make dev-status    - Check service status"
            echo "   • make dev-logs      - View logs"
            echo "   • make dev-shell     - Open shell"
            echo "   • make dev-stop      - Stop environment"
            echo "   • make test          - Run tests"
            ;;
            
        "python")
            log_info "Setting up Python virtual environment..."
            setup_python_env
            
            log_success "Python development environment ready!"
            echo ""
            echo "🎉 Python Environment Started!"
            echo "=============================="
            echo "🐍 Virtual environment: venv/"
            echo "📦 Dependencies installed"
            echo ""
            echo "🔧 Useful commands:"
            echo "   • source venv/bin/activate  - Activate environment"
            echo "   • python tests/test_simple_validation.py  - Run tests"
            echo "   • jupyter lab               - Start Jupyter"
            ;;
    esac
}

# Main setup logic
main() {
    echo ""
    log_info "Detecting system configuration..."
    
    OS=$(detect_os)
    log_info "Operating system: $OS"
    
    # Check for Docker first (preferred)
    if check_docker && check_docker_compose; then
        log_success "Docker environment available"
        echo ""
        echo "🐳 Docker development environment detected!"
        echo "This provides the best development experience with:"
        echo "   • Isolated dependencies"
        echo "   • Consistent environment across systems"
        echo "   • All services pre-configured"
        echo "   • Easy cleanup and reset"
        echo ""
        read -p "Use Docker environment? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            setup_dev_environment "python"
        else
            setup_dev_environment "docker"
        fi
    else
        log_warning "Docker not available, falling back to Python virtual environment"
        setup_dev_environment "python"
    fi
    
    echo ""
    log_success "Development environment setup complete!"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Read the documentation: docs/"
    echo "   2. Try the examples: examples/"
    echo "   3. Run the tests: make test"
    echo "   4. Explore the algorithms: src/bci_compression/algorithms/"
    echo ""
    echo "🆘 Need help?"
    echo "   • Check the documentation: README.md"
    echo "   • View available commands: make help"
    echo "   • Open an issue: https://github.com/hkevin01/brain-computer-compression/issues"
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "docker")
        setup_dev_environment "docker"
        ;;
    "python")
        setup_dev_environment "python"
        ;;
    "check")
        log_info "Checking system requirements..."
        check_docker
        check_docker_compose
        if command_exists python3; then
            log_success "Python 3 found: $(python3 --version)"
        else
            log_warning "Python 3 not found"
        fi
        ;;
    "help")
        echo "🧠 BCI Compression Toolkit - Development Setup"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup     - Auto-detect and setup development environment (default)"
        echo "  docker    - Force Docker development environment setup"
        echo "  python    - Force Python virtual environment setup"
        echo "  check     - Check system requirements"
        echo "  help      - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac
