# Production Container Optimization
# Multi-stage build for minimal, secure production containers
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create wheel directory
WORKDIR /wheels

# Copy requirements
COPY requirements*.txt ./

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt \
    && if [ -f requirements-emg.txt ]; then pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements-emg.txt; fi

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Minimal runtime dependencies
    libpq5 \
    libhdf5-103 \
    # Security updates
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Create non-root user for security
RUN groupadd -r bciuser && useradd -r -g bciuser bciuser
RUN mkdir -p /app && chown bciuser:bciuser /app

# Copy wheels from builder stage
COPY --from=builder /wheels /wheels

# Install packages from wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/* \
    && rm -rf /wheels

# Switch to non-root user
USER bciuser
WORKDIR /app

# Copy application code
COPY --chown=bciuser:bciuser src/ ./src/
COPY --chown=bciuser:bciuser setup.py ./
COPY --chown=bciuser:bciuser README.md ./

# Install application
RUN pip install --user --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/bciuser/.local/bin:${PATH}"

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from bci_compression.algorithms import create_neural_lz_compressor; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "bci_compression"]
