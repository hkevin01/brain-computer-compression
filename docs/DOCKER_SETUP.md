# 🐳 Docker Development Environment

Complete containerized development environment for the Brain-Computer Interface Data Compression Toolkit.

## 🚀 Quick Start

```bash
# 1. Start all services
docker-compose -f docker-compose.dev.yml up -d

# 2. Open in VS Code
code .
# Then: Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# 3. Access services
# - Jupyter Lab: http://localhost:8888
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - MinIO Console: http://localhost:9001
```

## 🏗️ Architecture

| Service | Container | Ports | Purpose |
|---------|-----------|-------|---------|
| **Backend** | `bci-backend-dev` | 8888, 8000, 5000 | Python/PyTorch API + Jupyter |
| **Frontend** | `bci-frontend-dev` | 3000, 4173, 8080 | React/Vite dev server |
| **PostgreSQL** | `bci-postgres-dev` | 5432 | Primary database |
| **Redis** | `bci-redis-dev` | 6379 | Caching & sessions |
| **MongoDB** | `bci-mongodb-dev` | 27017 | Document storage |
| **MinIO** | `bci-minio-dev` | 9000, 9001 | S3-compatible storage |

## 🛠️ Essential Commands

### Service Management
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Start specific services only
docker-compose -f docker-compose.dev.yml up -d postgres redis mongodb

# Stop all services
docker-compose -f docker-compose.dev.yml down

# Restart a service
docker-compose -f docker-compose.dev.yml restart backend

# View service status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend
```

### Building
```bash
# Build all containers
docker-compose -f docker-compose.dev.yml build

# Build specific container
docker-compose -f docker-compose.dev.yml build backend

# Build without cache (for troubleshooting)
docker-compose -f docker-compose.dev.yml build --no-cache frontend

# Build and start
docker-compose -f docker-compose.dev.yml up --build -d
```

### Development
```bash
# Execute commands in running container
docker-compose -f docker-compose.dev.yml exec backend bash
docker-compose -f docker-compose.dev.yml exec backend python test.py

# Run one-time commands
docker-compose -f docker-compose.dev.yml run --rm backend pytest tests/ -v
docker-compose -f docker-compose.dev.yml run --rm frontend npm test

# Check resource usage
docker stats --no-stream
```

## 🔧 Troubleshooting

### Quick Diagnostic
```bash
# Make script executable and run diagnostic
chmod +x scripts/docker-troubleshoot.sh
./scripts/docker-troubleshoot.sh
```

### Common Issues

#### 1. **Port Conflicts**
```bash
# Check what's using ports
lsof -i :8888
lsof -i :3000

# Kill conflicting processes (if safe)
sudo kill $(lsof -t -i:8888)
```

#### 2. **Frontend Build Failures**
```bash
# Rebuild frontend container (npm permission fix included)
docker-compose -f docker-compose.dev.yml build --no-cache frontend
```

#### 3. **Python Package Errors**
```bash
# Install BCI package in development mode
docker-compose -f docker-compose.dev.yml run --rm backend pip install -e .

# Check Python environment
docker-compose -f docker-compose.dev.yml run --rm backend python -c "import torch, numpy; print('OK')"
```

#### 4. **VS Code Dev Container Issues**
```bash
# Validate configuration
docker-compose -f docker-compose.dev.yml config --quiet

# Rebuild container in VS Code
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

#### 5. **Database Connection Issues**
```bash
# Test database connectivity
docker-compose -f docker-compose.dev.yml run --rm backend python -c "
import socket
for host, port in [('postgres', 5432), ('redis', 6379), ('mongodb', 27017)]:
    socket.create_connection((host, port), timeout=5)
    print(f'{host}:{port} OK')
"
```

#### 6. **Volume Permission Issues** (Linux/Mac)
```bash
# Fix file ownership
sudo chown -R $USER:$USER src/ tests/ examples/

# Check container user
docker-compose -f docker-compose.dev.yml run --rm backend id
```

### Complete Reset (Nuclear Option)
```bash
# ⚠️ WARNING: This destroys all containers, images, and volumes
docker-compose -f docker-compose.dev.yml down -v
docker container prune -f
docker image prune -a -f
docker volume prune -f
docker-compose -f docker-compose.dev.yml build --no-cache
```

## 📋 Validation Checklist

- [ ] All services start: `docker-compose -f docker-compose.dev.yml ps`
- [ ] Ports accessible: 8888 (Jupyter), 3000 (Frontend), 8000 (API)
- [ ] Python imports work: `import torch, numpy, bci_compression`
- [ ] VS Code Dev Container opens successfully
- [ ] Database connections working
- [ ] Tests pass: `pytest tests/` or `make test-quick`

## 📚 Resources

- **Complete Setup Guide**: `notebooks/Docker_Development_Environment_Setup.ipynb`
- **GitHub Copilot Instructions**: `.github/docker-copilot-instructions.md`
- **Troubleshooting Prompts**: `.github/copilot-docker-prompt.md`
- **Diagnostic Script**: `scripts/docker-troubleshoot.sh`

## 🎯 Development Workflow

1. **Start Environment**: `docker-compose -f docker-compose.dev.yml up -d`
2. **Open VS Code**: `code .` → Dev Containers: Reopen in Container
3. **Develop**: Edit code with full IntelliSense and debugging
4. **Test**: Run tests and neural compression algorithms
5. **Debug**: Use integrated debugger and Jupyter notebooks
6. **Stop**: `docker-compose -f docker-compose.dev.yml down`

## 🧠 BCI-Specific Features

- **GPU Support**: CUDA acceleration for neural algorithms
- **Multi-channel Processing**: Handle 32-256+ channel recordings
- **Real-time Requirements**: <1ms latency optimization
- **Neural Data Formats**: NEV, NSx, HDF5 support
- **Compression Algorithms**: Lossless and lossy with configurable quality

## 🆘 Getting Help

1. **Run diagnostic**: `./scripts/docker-troubleshoot.sh`
2. **Check documentation**: Jupyter notebook guide
3. **Use GitHub Copilot**: Copy prompt from `.github/copilot-docker-prompt.md`
4. **Check logs**: `docker-compose logs -f [service]`

The environment is optimized for brain-computer interface development with comprehensive tooling, real-time processing capabilities, and cross-platform consistency.
