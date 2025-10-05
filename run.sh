#!/usr/bin/env bash
set -euo pipefail

# BCI Compression Toolkit - Modern Orchestration Script
# Supports CPU, CUDA 12.x, and ROCm 6.x backends with Docker profiles

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_NAME="bci-compression"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default backend (cpu|cuda|rocm|auto)
DEFAULT_BACKEND="${BCC_ACCEL:-auto}"

# Ports
BACKEND_PORT="${BACKEND_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-3000}"
METRICS_PORT="${METRICS_PORT:-9090}"

# Environment
ENV_FILE="${ENV_FILE:-.env}"
COMPOSE_FILE="docker/compose/docker-compose.profiles.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "${ROOT_DIR}"

# =============================================================================
# UTILITIES
# =============================================================================

log() {
    echo -e "${GREEN}[BCC]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[BCC WARNING]${NC} $*" >&2
}

error() {
    echo -e "${RED}[BCC ERROR]${NC} $*" >&2
}

detect_backend() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
    elif command -v rocm-smi >/dev/null 2>&1; then
        echo "rocm"
    else
        echo "cpu"
    fi
}

ensure_env_file() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        log "Creating ${ENV_FILE} from template..."
        if [[ -f ".env.example" ]]; then
            cp .env.example "${ENV_FILE}"
        else
            cat > "${ENV_FILE}" << 'ENVEOF'
# BCI Compression Toolkit Environment
BCC_ACCEL=auto
BCC_LOG_LEVEL=INFO
BACKEND_PORT=8000
DASHBOARD_PORT=3000
ENVEOF
        fi
        log "Created ${ENV_FILE}. Please review and customize as needed."
    fi
}

check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is required but not installed"
        exit 1
    fi

    if ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose is required but not available"
        exit 1
    fi
}

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

docker_build() {
    local backend="${1:-auto}"
    local no_cache="${2:-false}"

    log "Building Docker images for backend: ${backend}"

    local build_args=""
    if [[ "${no_cache}" == "true" ]]; then
        build_args="--no-cache"
    fi

    # Build base image first
    docker build ${build_args} -f docker/Dockerfile.base -t brain-compression:base .

    case "${backend}" in
        "cpu"|"auto")
            docker build ${build_args} -f docker/Dockerfile.cpu -t brain-compression:cpu .
            ;;
        "cuda")
            if command -v nvidia-smi >/dev/null 2>&1; then
                docker build ${build_args} -f docker/Dockerfile.cuda -t brain-compression:cuda .
            else
                warn "CUDA not available, falling back to CPU build"
                docker build ${build_args} -f docker/Dockerfile.cpu -t brain-compression:cpu .
            fi
            ;;
        "rocm")
            if command -v rocm-smi >/dev/null 2>&1; then
                docker build ${build_args} -f docker/Dockerfile.rocm -t brain-compression:rocm .
            else
                warn "ROCm not available, falling back to CPU build"
                docker build ${build_args} -f docker/Dockerfile.cpu -t brain-compression:cpu .
            fi
            ;;
        "all")
            docker build ${build_args} -f docker/Dockerfile.cpu -t brain-compression:cpu .
            if command -v nvidia-smi >/dev/null 2>&1; then
                docker build ${build_args} -f docker/Dockerfile.cuda -t brain-compression:cuda .
            fi
            if command -v rocm-smi >/dev/null 2>&1; then
                docker build ${build_args} -f docker/Dockerfile.rocm -t brain-compression:rocm .
            fi
            ;;
    esac

    log "Docker build completed"
}

docker_up() {
    local backend="${1:-auto}"
    local detached="${2:-true}"

    if [[ "${backend}" == "auto" ]]; then
        backend=$(detect_backend)
        log "Auto-detected backend: ${backend}"
    fi

    ensure_env_file

    local compose_args="--profile ${backend}"
    local up_args=""
    if [[ "${detached}" == "true" ]]; then
        up_args="-d"
    fi

    log "Starting services with ${backend} backend..."
    docker compose -f "${COMPOSE_FILE}" ${compose_args} up ${up_args}

    if [[ "${detached}" == "true" ]]; then
        log "Services started successfully!"
        log "Backend API: http://localhost:${BACKEND_PORT}"
        log "Dashboard: http://localhost:${DASHBOARD_PORT}"
        log "Use './run.sh logs' to view logs"
    fi
}

docker_down() {
    log "Stopping all services..."
    docker compose -f "${COMPOSE_FILE}" down
    log "All services stopped"
}

docker_logs() {
    local service="${1:-}"
    local follow="${2:-true}"

    local args=""
    if [[ "${follow}" == "true" ]]; then
        args="-f"
    fi

    if [[ -n "${service}" ]]; then
        docker compose -f "${COMPOSE_FILE}" logs ${args} "${service}"
    else
        docker compose -f "${COMPOSE_FILE}" logs ${args}
    fi
}

# =============================================================================
# BENCHMARKING
# =============================================================================

run_benchmark() {
    local backend="${1:-auto}"
    local algorithms="${2:-lz4,zstd,blosc}"
    local output_file="${3:-logs/benchmark-${backend}.json}"

    if [[ "${backend}" == "auto" ]]; then
        backend=$(detect_backend)
    fi

    log "Running benchmark with ${backend} backend..."

    # Ensure output directory exists
    mkdir -p "$(dirname "${output_file}")"

    # Run benchmark
    if docker compose -f "${COMPOSE_FILE}" ps --services | grep -q "backend-${backend}"; then
        # Use Docker service
        docker compose -f "${COMPOSE_FILE}" exec "backend-${backend}" \
            python scripts/benchmark/bench_stream.py \
            --backend="${backend}" \
            --algorithms $(echo "${algorithms}" | tr ',' ' ') \
            --output="${output_file}"
    else
        # Use local Python
        python scripts/benchmark/bench_stream.py \
            --backend="${backend}" \
            --algorithms $(echo "${algorithms}" | tr ',' ' ') \
            --output="${output_file}"
    fi

    log "Benchmark completed. Results saved to ${output_file}"
}

# =============================================================================
# GUI MANAGEMENT
# =============================================================================

gui_open() {
    local url="http://localhost:${DASHBOARD_PORT}"

    log "Opening dashboard at ${url}..."

    # Try different ways to open URL based on OS
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${url}"
    elif command -v open >/dev/null 2>&1; then
        open "${url}"
    elif command -v start >/dev/null 2>&1; then
        start "${url}"
    else
        log "Please open ${url} in your browser"
    fi
}

# =============================================================================
# SYSTEM STATUS
# =============================================================================

show_status() {
    log "BCI Compression Toolkit Status"
    echo ""

    # System info
    echo "🖥️  System Information:"
    echo "   OS: $(uname -s) $(uname -r)"
    echo "   Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "   Docker: $(docker --version 2>/dev/null || echo 'Not found')"
    echo ""

    # GPU info
    echo "🚀 GPU Information:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "   NVIDIA: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        echo "   CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | cut -c9- || echo 'Not found')"
    else
        echo "   NVIDIA: Not available"
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        echo "   AMD ROCm: Available"
    else
        echo "   AMD ROCm: Not available"
    fi
    echo ""

    # Service status
    echo "🐳 Service Status:"
    if docker compose -f "${COMPOSE_FILE}" ps >/dev/null 2>&1; then
        docker compose -f "${COMPOSE_FILE}" ps
    else
        echo "   No services running"
    fi
    echo ""

    # Capabilities
    echo "🧠 Backend Capabilities:"
    if python3 -c "from bcc.api import capabilities; import json; print(json.dumps(capabilities(), indent=2))" 2>/dev/null; then
        :
    else
        echo "   Package not installed. Run 'make setup' to install."
    fi
}

# =============================================================================
# MAIN COMMAND HANDLER
# =============================================================================

show_help() {
    cat << 'HELPEOF'
BCI Compression Toolkit - Neural Data Compression for Brain-Computer Interfaces

USAGE:
    ./run.sh <command> [options]

COMMANDS:
    help                     Show this help message

    # Docker & Services
    build [backend] [--no-cache]    Build Docker images (cpu|cuda|rocm|all)
    up [backend] [--fg]             Start services (auto-detects backend by default)
    down                            Stop all services
    restart [backend]               Restart services
    logs [service] [--no-follow]    Show service logs
    ps                              Show running services

    # Development
    shell [service]                 Open shell in service container
    exec <service> <command>        Execute command in service

    # Benchmarking
    bench [backend] [algorithms]    Run compression benchmarks
    bench:cpu                       Run CPU-only benchmarks
    bench:cuda                      Run CUDA benchmarks
    bench:rocm                      Run ROCm benchmarks
    bench:all                       Run all backend benchmarks

    # GUI
    gui:open                        Open dashboard in browser

    # System
    status                          Show system and service status
    health                          Check system health
    clean                           Clean Docker artifacts

    # Shortcuts
    dev                             Quick development start (auto backend)
    prod                            Production start (CPU backend)

EXAMPLES:
    ./run.sh up                     # Auto-detect and start services
    ./run.sh up cuda               # Start with CUDA backend
    ./run.sh bench cpu lz4,zstd    # Benchmark CPU with specific algorithms
    ./run.sh logs backend-cuda     # Show CUDA backend logs
    ./run.sh shell backend-cpu     # Open shell in CPU backend

ENVIRONMENT VARIABLES:
    BCC_ACCEL                      Backend preference (cpu|cuda|rocm|auto)
    BACKEND_PORT                   Backend port (default: 8000)
    DASHBOARD_PORT                 Dashboard port (default: 3000)
    ENV_FILE                       Environment file (default: .env)

For more information, visit: https://github.com/hkevin01/brain-computer-compression
HELPEOF
}

main() {
    check_docker

    case "${1:-help}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        "build")
            docker_build "${2:-auto}" "${3:-false}"
            ;;
        "up")
            local fg_mode="true"
            if [[ "${3:-}" == "--fg" ]]; then
                fg_mode="false"
            fi
            docker_up "${2:-auto}" "${fg_mode}"
            ;;
        "down"|"stop")
            docker_down
            ;;
        "restart")
            docker_down
            docker_up "${2:-auto}"
            ;;
        "logs")
            local follow="true"
            if [[ "${3:-}" == "--no-follow" ]]; then
                follow="false"
            fi
            docker_logs "${2:-}" "${follow}"
            ;;
        "ps")
            docker compose -f "${COMPOSE_FILE}" ps
            ;;
        "shell")
            local service="${2:-backend-cpu}"
            docker compose -f "${COMPOSE_FILE}" exec "${service}" bash
            ;;
        "exec")
            local service="${2:-backend-cpu}"
            shift 2
            docker compose -f "${COMPOSE_FILE}" exec "${service}" "$@"
            ;;
        "bench")
            run_benchmark "${2:-auto}" "${3:-lz4,zstd,blosc}"
            ;;
        "bench:cpu")
            run_benchmark "cpu" "lz4,zstd,blosc,neural_lz77"
            ;;
        "bench:cuda")
            run_benchmark "cuda" "lz4,zstd,blosc,neural_lz77"
            ;;
        "bench:rocm")
            run_benchmark "rocm" "lz4,zstd,blosc,neural_lz77"
            ;;
        "bench:all")
            run_benchmark "cpu" "lz4,zstd,blosc,neural_lz77"
            run_benchmark "cuda" "lz4,zstd,blosc,neural_lz77" 2>/dev/null || true
            run_benchmark "rocm" "lz4,zstd,blosc,neural_lz77" 2>/dev/null || true
            ;;
        "gui:open")
            gui_open
            ;;
        "status")
            show_status
            ;;
        "health")
            # Check if API server is running
            if curl -s http://localhost:8000/health >/dev/null 2>&1; then
                echo "✅ API server is healthy"
                curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "API responded successfully"
            else
                echo "❌ API server is not running"
                echo "Start services with: ./run.sh up"
            fi
            ;;
        "clean")
            docker system prune -f
            docker volume prune -f
            log "Docker cleanup completed"
            ;;
        "dev")
            docker_up "auto" "true"
            ;;
        "prod")
            docker_up "cpu" "true"
            ;;
        *)
            error "Unknown command: ${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
