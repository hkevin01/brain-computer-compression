# Docker Development Environment - GitHub Copilot Instructions

## Context
This is a Brain-Computer Interface (BCI) data compression toolkit with a comprehensive Docker-based development environment. The setup includes multi-container orchestration with backend (Python), frontend (React/Vite), and multiple databases (PostgreSQL, Redis, MongoDB, MinIO).

## Docker Architecture Overview

### Service Stack (docker-compose.dev.yml)
- **Backend**: Python/BCI with PyTorch, NumPy, SciPy (ports: 8888, 8000, 5000)
- **Frontend**: Node.js/React/Vite development server (ports: 3000, 4173, 8080)
- **PostgreSQL**: Primary database (port: 5432)
- **Redis**: Caching and sessions (port: 6379)
- **MongoDB**: Document storage (port: 27017)
- **MinIO**: S3-compatible object storage (ports: 9000, 9001)

### Key Configuration Files
- `docker-compose.dev.yml` - Main orchestration (157 lines, 6 services)
- `docker/dev-backend.Dockerfile` - Python container with GPU support
- `docker/dev-frontend.Dockerfile` - Node.js container with global packages
- `.devcontainer/devcontainer.json` - VS Code Dev Container (140 lines, 20+ extensions)
- `.dockerignore` - Build context exclusions

## Common Issues and Copilot Guidance

### 1. Frontend Build Permission Errors
**Symptoms**: `npm install` EACCES errors, permission denied for global packages
**Root Cause**: npm global packages require root permissions in containers
**Solution Pattern**: Install global packages as root BEFORE switching to user
```dockerfile
# Correct approach - install as root first
RUN npm install -g --force yarn pnpm vite @vitejs/create-vue
# Then switch to user
RUN groupadd -r devuser && useradd -r -g devuser devuser
USER devuser
```

### 2. Port Conflicts
**Symptoms**: "Port already allocated", "Address already in use"
**Solution Pattern**: Check port usage, modify docker-compose ports
```bash
# Check port conflicts
lsof -i :8888
# Kill processes if safe
sudo kill $(lsof -t -i:8888)
```

### 3. Volume Permission Issues
**Symptoms**: Cannot write files, files owned by root
**Solution Pattern**: Fix ownership and ensure user ID matching
```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER src/ tests/
# Check container user ID
docker-compose run --rm backend id
```

### 4. Module Import Errors
**Symptoms**: ModuleNotFoundError for torch/numpy/bci_compression
**Solution Pattern**: Install in development mode, check PYTHONPATH
```bash
# Install package in development mode
docker-compose run --rm backend pip install -e .
# Check Python path
docker-compose run --rm backend python -c "import sys; print(sys.path)"
```

### 5. VS Code Dev Container Issues
**Symptoms**: Container fails to start, extensions not loading
**Solution Pattern**: Validate configuration, rebuild container
```bash
# Validate docker-compose
docker-compose -f docker-compose.dev.yml config --quiet
# Test backend manually
docker-compose run --rm backend python --version
```

## Development Workflow Commands

### Essential Daily Commands
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Open VS Code Dev Container
code . # Then: Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# View service status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Build and Rebuild
```bash
# Build all services
docker-compose -f docker-compose.dev.yml build

# Build specific service
docker-compose -f docker-compose.dev.yml build backend

# Rebuild without cache
docker-compose -f docker-compose.dev.yml build --no-cache frontend

# Rebuild and start
docker-compose -f docker-compose.dev.yml up --build -d
```

### Testing and Debugging
```bash
# Run tests
docker-compose -f docker-compose.dev.yml run --rm backend pytest tests/ -v

# Execute commands in container
docker-compose -f docker-compose.dev.yml exec backend bash

# Run one-time commands
docker-compose -f docker-compose.dev.yml run --rm backend python test_script.py

# Check resource usage
docker stats --no-stream
```

## Neural Data Processing Context

### BCI-Specific Considerations
- Multi-channel neural recordings (32-256+ channels)
- High-frequency sampling (1kHz-30kHz)
- Real-time processing requirements (<1ms latency)
- GPU acceleration for compression algorithms
- Memory-efficient streaming data handling

### Expected Package Environment
- **Core**: NumPy, SciPy, PyTorch, CuPy
- **Signal Processing**: scipy.signal, pywt (wavelets)
- **Compression**: Custom algorithms in `bci_compression` package
- **Data**: HDF5, NEV/NSx format support
- **Development**: Jupyter, pytest, black, mypy

## Troubleshooting Decision Tree

### Container Won't Start
1. Check port conflicts → `lsof -i :PORT`
2. Validate compose file → `docker-compose config`
3. Check Docker daemon → `docker info`
4. Review build logs → `docker-compose build --no-cache`

### Build Failures
1. **Frontend**: Check npm permission fixes in Dockerfile
2. **Backend**: Verify requirements.txt and system packages
3. **General**: Clear build cache, check .dockerignore

### Runtime Errors
1. **Import errors**: Install package in dev mode (`pip install -e .`)
2. **Permission errors**: Fix volume ownership
3. **Network errors**: Check service health and connectivity

### Performance Issues
1. **Memory**: Increase Docker memory limit (16GB+ recommended)
2. **Storage**: Use volume caching (:cached)
3. **Build speed**: Enable BuildKit (`DOCKER_BUILDKIT=1`)

## Emergency Reset Procedure
```bash
# Complete reset (DESTRUCTIVE)
docker-compose -f docker-compose.dev.yml down -v
docker container prune -f
docker image prune -a -f
docker volume prune -f
docker-compose -f docker-compose.dev.yml build --no-cache
```

## Integration Points

### VS Code Dev Container Features
- 20+ extensions pre-configured (Python, Jupyter, Docker, etc.)
- Integrated terminal with container environment
- Port forwarding for all services
- Debugger configuration for Python/Node.js
- Jupyter integration with kernel selection

### Service Health Checks
- Backend: HTTP endpoint check on port 8000
- Databases: TCP connection checks
- Frontend: HTTP endpoint check on port 3000
- MinIO: S3 API endpoint verification

### Development URLs
- Jupyter Lab: http://localhost:8888
- Frontend Dev Server: http://localhost:3000
- Backend API: http://localhost:8000
- MinIO Console: http://localhost:9001

## Quick Validation Tests
```bash
# Test Python environment
docker-compose run --rm backend python -c "import torch, numpy, scipy; print('✅ Core packages OK')"

# Test BCI package
docker-compose run --rm backend python -c "import sys; sys.path.insert(0, '/workspace/src'); import bci_compression; print('✅ BCI package OK')"

# Test database connections
docker-compose run --rm backend python -c "
import socket
for host, port in [('postgres', 5432), ('redis', 6379), ('mongodb', 27017)]:
    try:
        socket.create_connection((host, port), timeout=5)
        print(f'✅ {host} OK')
    except:
        print(f'❌ {host} FAIL')
"
```

## When to Use This Environment
- BCI algorithm development and testing
- Neural data compression research
- Multi-service application development
- Containerized machine learning workflows
- Cross-platform development consistency

This Docker environment provides a complete, isolated, and reproducible development stack for brain-computer interface data compression research and development.
