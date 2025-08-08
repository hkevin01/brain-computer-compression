# 🐳 Docker Quick Start Guide

The BCI Compression Toolkit includes a comprehensive Docker orchestration system that automatically manages both backend and GUI components.

## Prerequisites

- Docker Desktop or Docker Engine
- Docker Compose v2 (included with Docker Desktop)
- 4GB+ RAM for build process

## Quick Start (30 seconds)

```bash
# Clone and start everything
git clone https://github.com/hkevin01/brain-computer-compression.git
cd brain-computer-compression
./run.sh up         # Builds images and starts services
./run.sh gui:open   # Opens GUI in browser
```

## Core Commands

### Essential Operations
```bash
./run.sh up          # Start all services (backend + GUI)
./run.sh down        # Stop and remove containers
./run.sh restart     # Restart all services
./run.sh status      # Show comprehensive status
./run.sh logs        # Stream backend logs
./run.sh shell       # Open shell in backend container
```

### Development Workflow
```bash
./run.sh build       # Build Docker images
./run.sh ps          # Show running containers
./run.sh exec api "pytest tests/ -v"  # Run tests
./run.sh logs gui    # View GUI logs
./run.sh clean       # Remove unused images
```

### GUI Management
```bash
./run.sh gui:create         # Generate minimal GUI if missing
./run.sh gui:create --force # Overwrite existing GUI
./run.sh gui:open          # Open GUI in browser
```

## Configuration

### Environment Variables

Override defaults by setting environment variables:

```bash
# Port configuration
GUI_PORT=3001 ./run.sh up              # GUI on port 3001
PORTS="8080:8000" ./run.sh up          # Backend on port 8080

# Build options
BUILD_ARGS="USE_FULL_REQ=1" ./run.sh build  # Full ML dependencies
DEV_MODE=false ./run.sh up                   # Production mode
NO_CACHE=1 ./run.sh build                    # Force rebuild

# Image customization
IMAGE_NAME="my-bci:latest" ./run.sh build
GUI_IMAGE_NAME="my-gui:latest" ./run.sh build
```

### Full Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | `bci-compression-backend:local` | Backend Docker image name |
| `GUI_IMAGE_NAME` | `bci-compression-gui:local` | GUI Docker image name |
| `SERVICE_NAME` | `api` | Backend service name |
| `GUI_SERVICE_NAME` | `gui` | GUI service name |
| `PORTS` | `8000:8000` | Backend port mapping |
| `GUI_PORT` | `3000` | GUI port (host side) |
| `API_URL` | `http://localhost:8000` | API URL for browser |
| `API_URL_INTERNAL` | `http://api:8000` | API URL for container-to-container |
| `DEV_MODE` | `true` | Enable development mode |
| `ENV_FILE` | `.env` | Environment file to load |
| `GUI_PATH` | `./dashboard` | Path to GUI source |
| `DOCKER_PLATFORM` | - | Force platform (e.g., `linux/amd64`) |
| `BUILD_ARGS` | - | Docker build arguments |

## Service Access

After running `./run.sh up`:

- **🌐 GUI Dashboard**: http://localhost:3000 (or `$GUI_PORT`)
- **🔌 Backend API**: http://localhost:8000 (or `$PORTS`)
- **❤️ Health Check**: http://localhost:8000/health
- **📚 API Docs**: http://localhost:8000/docs

## GUI Auto-Generation

If no GUI exists at `./dashboard`, the script can generate one:

### Static Vanilla GUI (Default)
```bash
./run.sh gui:create static-vanilla
```
Creates a responsive single-page application with:
- ✅ Mobile-friendly design
- ✅ Real-time API status monitoring
- ✅ Interactive controls for compression plugins
- ✅ Response viewer with syntax highlighting
- ✅ Modern gradient UI with glassmorphism effects

### Working with Existing React/Vite GUI
```bash
# The script auto-detects existing Vite projects
./run.sh up  # Automatically uses existing dashboard/
```

## Development Modes

### Development Mode (Default)
- Source code mounted as volumes
- Auto-reload on file changes
- Debug logging enabled
- Vite dev server for GUI (if React/Vite)

```bash
DEV_MODE=true ./run.sh up
```

### Production Mode
- Optimized builds
- Static asset serving via Nginx
- Minimal logging
- No source mounts

```bash
DEV_MODE=false ./run.sh up
```

## Troubleshooting

### Build Issues
```bash
# Clear build cache
NO_CACHE=1 ./run.sh build

# Use minimal dependencies
BUILD_ARGS="" ./run.sh build

# Check build logs
./run.sh build 2>&1 | tee build.log
```

### Runtime Issues
```bash
# Check service status
./run.sh status

# View logs
./run.sh logs api    # Backend logs
./run.sh logs gui    # GUI logs

# Debug in container
./run.sh shell       # Backend shell
./run.sh exec gui /bin/sh  # GUI shell
```

### Port Conflicts
```bash
# Use different ports
GUI_PORT=3001 PORTS="8080:8000" ./run.sh up
```

### Container Networking
```bash
# Backend accessible from GUI container as:
http://api:8000

# From host browser:
http://localhost:8000
```

## Advanced Usage

### Custom Build Arguments
```bash
# Example: Use PyPI mirror, enable GPU support
BUILD_ARGS="PIP_INDEX_URL=https://pypi.org/simple USE_FULL_REQ=1" ./run.sh build
```

### Volume Mounts (Single Container Mode)
```bash
# Mount additional directories
MOUNTS="./data:/app/data,./models:/app/models" ./run.sh up
```

### Multiple Environments
```bash
# Development
ENV_FILE=.env.dev ./run.sh up

# Production
ENV_FILE=.env.prod DEV_MODE=false ./run.sh up
```

### Platform-Specific Builds
```bash
# Force platform (useful on Apple Silicon)
DOCKER_PLATFORM=linux/amd64 ./run.sh build
```

## Integration with CI/CD

```yaml
# Example GitHub Actions snippet
- name: Test with Docker
  run: |
    ./run.sh build
    ./run.sh up -d
    ./run.sh exec api "pytest tests/ -v"
    ./run.sh down
```

## File Structure

The Docker setup creates/uses these files:

```
brain-computer-compression/
├── run.sh                    # Main orchestration script
├── Dockerfile               # Backend build definition
├── docker-compose.yml       # Service orchestration
├── requirements-backend.txt # Minimal dependencies
├── dashboard/
│   ├── Dockerfile          # GUI build definition
│   └── ...                 # React/Vite source
└── gui/                    # Auto-generated (if created)
    ├── index.html
    ├── app.js
    ├── styles.css
    └── Dockerfile
```

## Performance Tips

- **Use BuildKit**: Automatically enabled for faster builds
- **Layer Caching**: Dependencies installed before source copy
- **Multi-stage Builds**: Minimal runtime images
- **Health Checks**: Ensure services are ready before dependent services start

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Monitor Compression**: Use the GUI at http://localhost:3000
3. **Run Tests**: `./run.sh exec api "pytest tests/ -v"`
4. **Check Logs**: `./run.sh logs`
5. **Customize**: Modify environment variables as needed

For more advanced configuration, see the main [README.md](README.md) and [project documentation](docs/).

---

**🎯 Goal**: Get from git clone to running BCI compression system in under 1 minute with a single command.
