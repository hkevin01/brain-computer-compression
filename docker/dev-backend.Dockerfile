# Universal Backend Development Container
# Multi-purpose development environment for Python, Node.js, and scientific computing
FROM python:3.11-slim as base

# Install system dependencies and development tools
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Development tools
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    jq \
    # Database clients
    postgresql-client \
    mysql-client \
    redis-tools \
    # Scientific computing dependencies
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # Neural/EMG data processing
    libhdf5-dev \
    libsndfile1-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS for any frontend tooling
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Create development user
RUN groupadd -r devuser && useradd -r -g devuser devuser
RUN mkdir -p /home/devuser && chown devuser:devuser /home/devuser
USER devuser
WORKDIR /workspace

# Set up Python development environment
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python development tools
COPY requirements*.txt ./
RUN pip install --user --no-cache-dir \
    # Core development tools
    pip-tools \
    black \
    flake8 \
    pytest \
    pytest-cov \
    pytest-xdist \
    mypy \
    pre-commit \
    # Jupyter for data analysis
    jupyter \
    jupyterlab \
    ipykernel \
    # Scientific stack
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    # Neural/EMG specific
    h5py \
    scikit-learn \
    && pip install --user --no-cache-dir -r requirements.txt \
    && if [ -f requirements-emg.txt ]; then pip install --user --no-cache-dir -r requirements-emg.txt; fi

# Add user bin to PATH
ENV PATH="/home/devuser/.local/bin:${PATH}"

# Expose common development ports
EXPOSE 8000 8080 8888 5000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command for development
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
