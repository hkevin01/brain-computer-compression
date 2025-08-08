#!/bin/bash

# Emergency fix script for Docker build issues
# This script will reset the Docker setup to a working state

set -e

echo "🔧 Emergency Docker Build Fix"
echo "============================"

# Clear Docker build cache
echo "🧹 Clearing Docker build cache..."
docker builder prune -af || true

# Show current Dockerfile content around problematic line
echo ""
echo "📄 Checking current Dockerfile around line 37..."
if [ -f "Dockerfile" ]; then
    echo "Lines 30-45 of current Dockerfile:"
    sed -n '30,45p' Dockerfile | nl -v30
else
    echo "❌ No Dockerfile found!"
    exit 1
fi

# Check for problematic pip install line
echo ""
echo "🔍 Searching for problematic 'pip install .' commands..."
if grep -n "pip install.*\." Dockerfile; then
    echo "❌ Found problematic pip install lines!"
    echo "These lines need to be removed or fixed."

    # Offer to create a fixed Dockerfile
    echo ""
    echo "🛠️  Creating a fixed Dockerfile..."

    # Backup current Dockerfile
    cp Dockerfile Dockerfile.backup.$(date +%s)
    echo "✅ Backed up current Dockerfile"

    # Create a minimal working Dockerfile
    cat > Dockerfile << 'EOF'
# Minimal Dockerfile that avoids package installation issues
FROM python:3.11-slim AS builder

ARG USE_FULL_REQ=0
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements files
COPY requirements-backend.txt requirements-backend.txt
COPY requirements.txt requirements.txt

# Upgrade pip first
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies only (avoid setup.py issues)
RUN if [ "${USE_FULL_REQ}" = "1" ]; then \
        echo "Installing FULL requirements.txt" && \
        python -m pip install --no-cache-dir -r requirements.txt ; \
    else \
        echo "Installing MINIMAL requirements-backend.txt" && \
        python -m pip install --no-cache-dir -r requirements-backend.txt ; \
    fi

# Copy source code (but don't install as package)
COPY scripts/ /build/scripts/
COPY src/ /build/src/

# Validate dependencies are installed correctly
RUN python -c "import fastapi, uvicorn, numpy, scipy; print('Dependencies OK')"

# Final runtime stage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages and source code
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /build/scripts/ /app/scripts/
COPY --from=builder /build/src/ /app/src/

# Create non-root user
RUN useradd -m -r bciuser && chown -R bciuser:bciuser /app
USER bciuser

# Set Python path to find modules
ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Run the server directly
ENTRYPOINT ["uvicorn"]
CMD ["scripts.telemetry_server:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    echo "✅ Created fixed Dockerfile (original backed up)"

else
    echo "✅ No problematic pip install lines found"
fi

# Check .dockerignore for README exclusion
echo ""
echo "📋 Checking .dockerignore for README exclusion..."
if grep -n "^\*\.md$" .dockerignore; then
    echo "⚠️  Found *.md exclusion in .dockerignore"
    echo "Adding exception for README files..."

    # Backup .dockerignore
    cp .dockerignore .dockerignore.backup.$(date +%s)

    # Fix .dockerignore to allow README files
    sed -i '/^\*\.md$/a !README.md\n!README_NEW.md' .dockerignore
    echo "✅ Fixed .dockerignore to allow README files"
else
    echo "✅ .dockerignore looks OK"
fi

# Test build
echo ""
echo "🧪 Testing Docker build..."
if docker build --no-cache -t bci-test-build . ; then
    echo "✅ Docker build successful!"
    echo ""
    echo "🎉 Problem resolved! You can now run:"
    echo "   ./run.sh build"
    echo "   ./run.sh up"
else
    echo "❌ Build still failing. Please check the output above for errors."
    echo ""
    echo "💡 Additional troubleshooting:"
    echo "   1. Check that requirements-backend.txt exists"
    echo "   2. Ensure scripts/ and src/ directories exist"
    echo "   3. Verify Docker has enough disk space and memory"
    echo "   4. Try: docker system prune -af"
fi
