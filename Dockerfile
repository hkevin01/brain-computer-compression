# Use a multi-stage build for smaller final image
FROM python:3.9-slim AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with build requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Run tests to verify build
RUN python -m pytest tests/ -v --tb=short

# Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed packages and source from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /build/src/ /app/src/

# Create non-root user for security
RUN useradd -m -r bciuser && \
    chown -R bciuser:bciuser /app

# Switch to non-root user
USER bciuser

# Environment variables
ENV PYTHONPATH=/app/src
ENV PORT=8000
ENV WORKERS=4
ENV LOG_LEVEL=info

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["python", "-m"]
CMD ["bci_compression.api.server", "--port", "8000", "--workers", "4"]
