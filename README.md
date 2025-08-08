# Brain-Computer Interface Data Compression Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?style=flat-square&logo=docker)](docker/)
[![Documentation](https://img.shields.io/badge/docs-organized-brightgreen.svg?style=flat-square)](docs/)

> **A state-of-the-art toolkit for neural data compression in brain-computer interfaces**

## 🚀 Quick Start

### Using Docker (Recommended)
```bash
# Build and start services
./run.sh up

# Open GUI in browser
./run.sh gui:open
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run backend server
python scripts/telemetry_server.py
```

## 📁 Project Structure

```
brain-computer-compression/
├── README.md                    # This file
├── requirements*.txt            # Python dependencies
├── pyproject.toml              # Python project config
├── run.sh                      # Main orchestration script
├── docs/                       # 📚 Documentation
│   ├── guides/                 # User guides
│   └── project/               # Project documentation
├── docker/                     # 🐳 Docker configuration
│   ├── Dockerfile             # Main backend image
│   └── compose/               # Docker compose files
├── scripts/                    # 🔧 Scripts and tools
│   ├── setup/                 # Installation scripts
│   └── tools/                 # Utility scripts
├── src/                       # 🧠 Core source code
├── tests/                     # 🧪 Test suite
├── dashboard/                 # 🌐 React GUI
├── examples/                  # 📖 Usage examples
└── notebooks/                 # 📊 Jupyter notebooks
```

## 📚 Documentation

- **[Quick Start Guide](docs/guides/DOCKER_QUICK_START.md)** - Get started with Docker
- **[Docker Troubleshooting](docs/guides/DOCKER_BUILD_FIX.md)** - Fix common Docker issues
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute
- **[Changelog](docs/CHANGELOG.md)** - Version history
- **[Project Status](docs/project/STATUS_REPORT.md)** - Current development status

## 🐳 Docker Usage

All Docker files are now organized in the `docker/` directory:

```bash
# Build images
./run.sh build

# Start services
./run.sh up

# View logs
./run.sh logs

# Stop services
./run.sh down
```

## 🔧 Development Tools

Utility scripts are in `scripts/tools/`:

- **Setup**: `scripts/setup/setup.sh` - Quick environment setup
- **Docker Tools**: `scripts/tools/test_docker_build.sh` - Test Docker builds
- **Cleanup**: `scripts/tools/cleanup_now.sh` - Clean temporary files

## ✨ Key Features

- **Advanced Compression**: Neural-optimized lossless and lossy algorithms
- **Real-Time Processing**: Ultra-low latency (< 1ms for basic algorithms)
- **GPU Acceleration**: CUDA-optimized kernels with CPU fallback
- **Plugin System**: Modular, extensible architecture
- **Web Dashboard**: Real-time monitoring and control interface
- **Docker Ready**: One-command deployment and scaling

## 🏃‍♂️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/hkevin01/brain-computer-compression.git
   cd brain-computer-compression
   ```

2. **Start with Docker** (recommended)
   ```bash
   ./run.sh up
   ```

3. **Or manual setup**
   ```bash
   ./scripts/setup/setup.sh
   ```

4. **Access the dashboard**
   - Open http://localhost:3000 in your browser
   - Or run `./run.sh gui:open`

## 📖 Learn More

- **API Documentation**: http://localhost:8000/docs (when running)
- **Project Guides**: [docs/guides/](docs/guides/)
- **Development Setup**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Architecture Overview**: [docs/project/](docs/project/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**🎯 Goal**: Efficient neural data compression for next-generation brain-computer interfaces.
