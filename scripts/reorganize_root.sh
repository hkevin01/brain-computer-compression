#!/bin/bash

# Comprehensive Root Directory Reorganization Script
# Moves documentation and Docker files to organized subdirectories

set -e

echo "📁 Root Directory Reorganization"
echo "==============================="

# Create target directories
echo "📂 Creating target directories..."
mkdir -p docs/guides
mkdir -p docs/project
mkdir -p docker/compose
mkdir -p scripts/setup
mkdir -p scripts/tools

echo "✅ Created directory structure"

# Phase 1: Move documentation files
echo ""
echo "📝 Phase 1: Moving documentation files..."

DOC_MOVES=(
    "BCI_Validation_Framework_Summary.md:docs/project/"
    "CHANGELOG.md:docs/"
    "CONTRIBUTING.md:docs/"
    "DOCKER_BUILD_FIX.md:docs/guides/"
    "DOCKER_QUICK_START.md:docs/guides/"
    "EMG_UPDATE.md:docs/project/"
    "STATUS_REPORT.md:docs/project/"
    "README_NEW.md:docs/project/"  # If it still exists
)

for move in "${DOC_MOVES[@]}"; do
    file="${move%%:*}"
    dest="${move##*:}"
    if [ -f "$file" ]; then
        mv "$file" "$dest" && echo "  ✅ Moved $file → $dest"
    fi
done

# Phase 2: Move Docker files
echo ""
echo "🐳 Phase 2: Moving Docker files..."

DOCKER_MOVES=(
    "Dockerfile:docker/"
    "Dockerfile.frontend:docker/"
    "Dockerfile.minimal:docker/"
    ".dockerignore:docker/"
    "docker-compose.yml:docker/compose/"
    "docker-compose.dev.yml:docker/compose/"
    "docker-compose.dev.yml.backup:docker/compose/"
)

for move in "${DOCKER_MOVES[@]}"; do
    file="${move%%:*}"
    dest="${move##*:}"
    if [ -f "$file" ]; then
        mv "$file" "$dest" && echo "  ✅ Moved $file → $dest"
    fi
done

# Phase 3: Move setup and tool scripts
echo ""
echo "🔧 Phase 3: Moving scripts..."

SCRIPT_MOVES=(
    "setup.sh:scripts/setup/"
    "setup_and_test.sh:scripts/setup/"
    "setup_and_test.bat:scripts/setup/"
    "cleanup_now.sh:scripts/tools/"
    "cleanup_root.sh:scripts/tools/"
    "comprehensive_fix.sh:scripts/tools/"
    "fix_docker_build.sh:scripts/tools/"
    "test_docker_build.sh:scripts/tools/"
    "test_docker_workflow.sh:scripts/tools/"
)

for move in "${SCRIPT_MOVES[@]}"; do
    file="${move%%:*}"
    dest="${move##*:}"
    if [ -f "$file" ]; then
        mv "$file" "$dest" && echo "  ✅ Moved $file → $dest"
    fi
done

# Phase 4: Move other files
echo ""
echo "📄 Phase 4: Moving other files..."

OTHER_MOVES=(
    "benchmark_results.json:.benchmarks/"
    "test_output.log:logs/"
    "package.json:docs/project/"  # Root package.json if it exists
)

for move in "${OTHER_MOVES[@]}"; do
    file="${move%%:*}"
    dest="${move##*:}"
    if [ -f "$file" ]; then
        mkdir -p "$(dirname "$dest")"
        mv "$file" "$dest" && echo "  ✅ Moved $file → $dest"
    fi
done

echo ""
echo "🔄 Phase 5: Updating file references..."

# Update run.sh to point to new Docker location
if [ -f "run.sh" ]; then
    echo "🔧 Updating run.sh Docker references..."

    # Create backup
    cp run.sh run.sh.backup

    # Update docker build commands to use new Dockerfile location
    sed -i 's|docker build \("${build_extra\[@\]}" \$(build_args_flags)\) -t "${IMAGE_NAME}" \.|docker build \1 -f docker/Dockerfile -t "${IMAGE_NAME}" .|g' run.sh

    # Update compose file discovery function
    sed -i 's|if \[\[ -f docker-compose\.yml \]\]; then echo docker-compose\.yml; return; fi|if [[ -f docker/compose/docker-compose.yml ]]; then echo docker/compose/docker-compose.yml; return; fi|g' run.sh
    sed -i 's|if \[\[ -f docker-compose\.yaml \]\]; then echo docker-compose\.yaml; return; fi|if [[ -f docker/compose/docker-compose.yaml ]]; then echo docker/compose/docker-compose.yaml; return; fi|g' run.sh

    # Update compose file creation
    sed -i 's|COMPOSE_FILE="docker-compose\.yml"|COMPOSE_FILE="docker/compose/docker-compose.yml"|g' run.sh
    sed -i 's|cat > docker-compose\.yml|mkdir -p docker/compose \&\& cat > docker/compose/docker-compose.yml|g' run.sh

    # Update compose file checks
    sed -i 's|if \[\[ -f docker-compose\.yml \]\]; then|if [[ -f docker/compose/docker-compose.yml ]]; then|g' run.sh
    sed -i 's|"Updating docker-compose\.yml|"Updating docker/compose/docker-compose.yml|g' run.sh

    # Update any remaining docker-compose.yml references
    sed -i 's|docker-compose\.yml|docker/compose/docker-compose.yml|g' run.sh

    echo "  ✅ Updated run.sh (backup saved as run.sh.backup)"
fi

# Update any scripts that may have been moved
echo "🔧 Updating documentation references..."

# Update documentation files that have been moved
for doc_file in docs/guides/DOCKER_QUICK_START.md docs/guides/DOCKER_BUILD_FIX.md; do
    if [ -f "$doc_file" ]; then
        # Update Dockerfile references to new location
        sed -i 's|├── Dockerfile|├── docker/Dockerfile|g' "$doc_file"
        sed -i 's|grep -n "pip install\.\*\\\.\$" Dockerfile|grep -n "pip install.*\\.$" docker/Dockerfile|g' "$doc_file"
        sed -i 's|cp Dockerfile\.minimal Dockerfile|cp docker/Dockerfile.minimal docker/Dockerfile|g' "$doc_file"
        # Update file structure documentation
        sed -i 's|├── Dockerfile               # Backend build definition|├── docker/Dockerfile       # Backend build definition|g' "$doc_file"
        sed -i 's|├── docker-compose\.yml       # Service orchestration|├── docker/compose/docker-compose.yml # Service orchestration|g' "$doc_file"
        echo "  ✅ Updated $doc_file"
    fi
done

# Update any .github files if they exist
if [ -f ".github/copilot-docker-prompt.md" ]; then
    sed -i 's|docker/dev-backend\.Dockerfile|docker/dev-backend.Dockerfile|g' .github/copilot-docker-prompt.md
    echo "  ✅ Updated .github/copilot-docker-prompt.md"
fi

# Update VS Code tasks if they reference moved files
if [ -f ".vscode/tasks.json" ]; then
    # Update any Dockerfile references
    sed -i 's|"Dockerfile"|"docker/Dockerfile"|g' .vscode/tasks.json
    echo "  ✅ Updated .vscode/tasks.json"
fi

echo ""
echo "📝 Phase 6: Creating updated README with new structure..."

# Create a simple README that references the docs
cat > README.md << 'EOF'
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
EOF

echo "  ✅ Created organized README.md"

# Update .gitignore to reflect new structure
echo ""
echo "📝 Phase 7: Updating .gitignore..."

cat >> .gitignore << 'EOF'

# Reorganization: Files in new locations
/docs/project/benchmark_results.json
/logs/test_output.log
/docker/Dockerfile.minimal
/docker/Dockerfile.frontend
EOF

echo "  ✅ Updated .gitignore"

echo ""
echo "🎉 Reorganization Complete!"
echo "=========================="
echo ""
echo "📁 New directory structure:"
echo "  docs/guides/          - User guides and documentation"
echo "  docs/project/         - Project-specific documentation"
echo "  docker/               - Docker configuration files"
echo "  docker/compose/       - Docker compose files"
echo "  scripts/setup/        - Installation scripts"
echo "  scripts/tools/        - Utility and maintenance scripts"
echo ""
echo "🔄 Updated references in:"
echo "  ✅ run.sh (Docker paths)"
echo "  ✅ README.md (new structure)"
echo "  ✅ .gitignore (new locations)"
echo ""
echo "📊 Root directory now contains:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"
echo ""
echo "🚀 Next steps:"
echo "  1. Test Docker build: ./run.sh build"
echo "  2. Start services: ./run.sh up"
echo "  3. Check documentation: ls docs/guides/"
