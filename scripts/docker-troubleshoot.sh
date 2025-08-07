#!/bin/bash

# Docker Development Environment Troubleshooting Script
# For Brain-Computer Interface Data Compression Toolkit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if docker-compose.dev.yml exists
COMPOSE_FILE="docker-compose.dev.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "docker-compose.dev.yml not found. Please run from project root."
    exit 1
fi

print_header "Docker Development Environment Diagnostic"
print_info "Brain-Computer Interface Data Compression Toolkit"
print_info "Checking multi-container development environment..."

# 1. Check Docker and Docker Compose
print_header "1. Docker System Check"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed: $DOCKER_VERSION"
else
    print_error "Docker not installed or not in PATH"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
else
    print_error "Docker Compose not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

# 2. Check Docker Compose configuration
print_header "2. Docker Compose Configuration"

if docker-compose -f "$COMPOSE_FILE" config --quiet; then
    print_success "docker-compose.dev.yml syntax is valid"
else
    print_error "docker-compose.dev.yml has syntax errors"
    docker-compose -f "$COMPOSE_FILE" config
    exit 1
fi

# 3. Check port availability
print_header "3. Port Availability Check"

PORTS=(3000 5432 6379 8000 8888 9000 9001 27017)
for port in "${PORTS[@]}"; do
    if lsof -i :$port &> /dev/null; then
        PROCESS=$(lsof -ti :$port)
        print_warning "Port $port is in use by process $PROCESS"
        # Show what's using the port
        lsof -i :$port | head -2
    else
        print_success "Port $port is available"
    fi
done

# 4. Check Docker images
print_header "4. Docker Images Check"

# Check if images exist
if docker images | grep -q "bci.*dev"; then
    print_success "BCI development images found"
    docker images | grep "bci.*dev"
else
    print_warning "BCI development images not found. Need to build."
fi

# 5. Check container status
print_header "5. Container Status"

CONTAINERS=$(docker-compose -f "$COMPOSE_FILE" ps -q)
if [ -z "$CONTAINERS" ]; then
    print_warning "No containers are running"
else
    print_info "Container status:"
    docker-compose -f "$COMPOSE_FILE" ps

    # Check individual service health
    SERVICES=("backend" "frontend" "postgres" "redis" "mongodb" "minio")
    for service in "${SERVICES[@]}"; do
        STATUS=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")
        if [ ! -z "$STATUS" ]; then
            HEALTH=$(docker inspect --format='{{.State.Status}}' "$STATUS" 2>/dev/null || echo "unknown")
            if [ "$HEALTH" = "running" ]; then
                print_success "$service container is running"
            else
                print_warning "$service container status: $HEALTH"
            fi
        else
            print_warning "$service container is not running"
        fi
    done
fi

# 6. Check volumes and file permissions
print_header "6. File System and Permissions"

# Check if key directories exist
DIRS=("src" "tests" "examples" "notebooks" ".devcontainer")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        OWNER=$(stat -c '%U' "$dir" 2>/dev/null || stat -f '%Su' "$dir" 2>/dev/null || echo "unknown")
        print_success "Directory $dir exists (owner: $OWNER)"
    else
        print_warning "Directory $dir does not exist"
    fi
done

# Check if we can write to src directory
if [ -w "src" ]; then
    print_success "src directory is writable"
else
    print_warning "src directory is not writable"
fi

# 7. Test basic connectivity
print_header "7. Service Connectivity Test"

if [ ! -z "$CONTAINERS" ]; then
    # Test backend container
    if docker-compose -f "$COMPOSE_FILE" ps -q backend &> /dev/null; then
        if docker-compose -f "$COMPOSE_FILE" exec -T backend python --version &> /dev/null; then
            print_success "Backend Python environment is accessible"
        else
            print_warning "Backend Python environment is not responding"
        fi

        # Test database connections from backend
        if docker-compose -f "$COMPOSE_FILE" exec -T backend python -c "
import socket
services = [('postgres', 5432), ('redis', 6379), ('mongodb', 27017), ('minio', 9000)]
for host, port in services:
    try:
        socket.create_connection((host, port), timeout=5)
        print(f'✅ {host}:{port} connection successful')
    except Exception as e:
        print(f'❌ {host}:{port} connection failed: {e}')
" 2>/dev/null; then
            print_success "Database connectivity test completed"
        else
            print_warning "Database connectivity test failed"
        fi
    fi
fi

# 8. Test BCI package import
print_header "8. BCI Package Test"

if [ ! -z "$CONTAINERS" ] && docker-compose -f "$COMPOSE_FILE" ps -q backend &> /dev/null; then
    if docker-compose -f "$COMPOSE_FILE" exec -T backend python -c "
import sys
sys.path.insert(0, '/workspace/src')
try:
    import bci_compression
    print('✅ BCI compression package is importable')
except ImportError as e:
    print(f'❌ BCI compression package import failed: {e}')
" 2>/dev/null; then
        print_success "BCI package test completed"
    else
        print_warning "BCI package test failed"
    fi
fi

# 9. Resource usage check
print_header "9. Resource Usage"

if [ ! -z "$CONTAINERS" ]; then
    print_info "Current container resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || print_warning "Could not get container stats"
fi

# Docker system info
DISK_USAGE=$(docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}\t{{.Reclaimable}}" 2>/dev/null || echo "Could not get disk usage")
print_info "Docker disk usage:"
echo "$DISK_USAGE"

# 10. VS Code Dev Container check
print_header "10. VS Code Dev Container Configuration"

if [ -f ".devcontainer/devcontainer.json" ]; then
    print_success ".devcontainer/devcontainer.json exists"
    # Check if it's valid JSON
    if python3 -m json.tool .devcontainer/devcontainer.json > /dev/null 2>&1; then
        print_success "devcontainer.json is valid JSON"
    else
        print_error "devcontainer.json has invalid JSON syntax"
    fi
else
    print_warning ".devcontainer/devcontainer.json not found"
fi

# 11. Recommendations
print_header "11. Recommendations"

if [ -z "$CONTAINERS" ]; then
    print_info "To start the development environment:"
    echo "  docker-compose -f docker-compose.dev.yml up -d"
fi

if ! docker images | grep -q "bci.*dev"; then
    print_info "To build the development images:"
    echo "  docker-compose -f docker-compose.dev.yml build"
fi

# Check if any ports are conflicting
CONFLICTING_PORTS=()
for port in "${PORTS[@]}"; do
    if lsof -i :$port &> /dev/null; then
        CONFLICTING_PORTS+=($port)
    fi
done

if [ ${#CONFLICTING_PORTS[@]} -gt 0 ]; then
    print_info "To resolve port conflicts, either:"
    echo "  1. Stop processes using ports: ${CONFLICTING_PORTS[*]}"
    echo "  2. Modify port mappings in docker-compose.dev.yml"
fi

print_info "For complete troubleshooting guide, see:"
echo "  notebooks/Docker_Development_Environment_Setup.ipynb"

print_header "Diagnostic Complete"
print_info "Check the output above for any issues that need attention."

# Summary
ISSUES=0
if [ ! -z "$(docker-compose -f "$COMPOSE_FILE" config 2>&1 >/dev/null)" ]; then
    ((ISSUES++))
fi

if [ ${#CONFLICTING_PORTS[@]} -gt 0 ]; then
    ((ISSUES++))
fi

if [ -z "$CONTAINERS" ]; then
    print_warning "Development environment is not running"
else
    print_success "Development environment appears to be running"
fi

if [ $ISSUES -eq 0 ]; then
    print_success "No critical issues detected!"
else
    print_warning "$ISSUES potential issues detected. Review output above."
fi
