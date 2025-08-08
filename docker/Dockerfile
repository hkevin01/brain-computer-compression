# Use a multi-stage build for smaller final image
FROM python:3.11-slim AS builder

ARG PIP_INDEX_URL
ARG USE_FULL_REQ=0
ENV DEBIAN_FRONTEND=noninteractive

# Install system build & runtime deps (add curl for healthcheck, gcc for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy minimal backend requirements and full requirements separately for cache efficiency
COPY requirements-backend.txt requirements-backend.txt
COPY requirements.txt requirements.txt

# Upgrade pip & build tools first (prevents BackendUnavailable / legacy issues)
RUN python -m pip install --upgrade pip setuptools wheel

# Install minimal or full dependencies based on ARG
RUN if [ "${USE_FULL_REQ}" = "1" ]; then \
    echo "Installing FULL requirements.txt" && \
    python -m pip install --no-cache-dir -r requirements.txt ; \
    else \
    echo "Installing MINIMAL requirements-backend.txt (set USE_FULL_REQ=1 to override)" && \
    python -m pip install --no-cache-dir -r requirements-backend.txt ; \
    fi

# Copy source code after deps for better layer caching
COPY . .

# For Docker builds, skip full package install to avoid setup.py/pyproject.toml issues
# Instead, just ensure scripts/ and src/ are available in PYTHONPATH
# (Optional) lightweight smoke tests (skip heavy test suite to speed image build)
RUN python - <<'PY'
import importlib, sys
mods = ["fastapi","uvicorn","numpy"]
for m in mods:
    if importlib.util.find_spec(m) is None:
        print(f"Missing critical dependency: {m}", file=sys.stderr)
        sys.exit(1)
print("Smoke checks passed")
PY

# Final runtime stage
FROM python:3.11-slim

ARG PIP_INDEX_URL
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    LOG_LEVEL=info \
    WORKERS=4 \
    APP_MODULE=scripts.telemetry_server:app

# System runtime deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed site-packages and source (for dynamic plugins / scripts)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /build/scripts/ /app/scripts/
COPY --from=builder /build/src/ /app/src/
COPY --from=builder /build/requirements-backend.txt /app/
COPY --from=builder /build/requirements.txt /app/

# Create non-root user
RUN useradd -m -r bciuser && chown -R bciuser:bciuser /app
USER bciuser

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Default command runs uvicorn directly (faster than python -m for this case)
ENTRYPOINT ["uvicorn"]
CMD ["scripts.telemetry_server:app", "--host", "0.0.0.0", "--port", "8000"]
