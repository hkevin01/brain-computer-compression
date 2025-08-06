# Development Tools Container
# Code quality, security, and CI/CD tooling
FROM ubuntu:22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core development tools
    git \
    curl \
    wget \
    jq \
    unzip \
    # Build tools
    build-essential \
    # Python for tooling
    python3 \
    python3-pip \
    # Node.js for JS tooling
    nodejs \
    npm \
    # Security scanning tools
    gnupg \
    software-properties-common \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Go for additional tooling
RUN wget -O go.tar.gz https://go.dev/dl/go1.21.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go.tar.gz \
    && rm go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Create development user
RUN groupadd -r devuser && useradd -r -g devuser devuser
RUN mkdir -p /home/devuser && chown devuser:devuser /home/devuser
USER devuser
WORKDIR /workspace

# Install Python development tools
RUN pip3 install --user \
    # Code quality
    black \
    flake8 \
    pylint \
    mypy \
    isort \
    # Security scanning
    bandit \
    safety \
    # Documentation
    sphinx \
    mkdocs \
    # Testing
    pytest \
    pytest-cov \
    # Pre-commit hooks
    pre-commit

# Install JavaScript/TypeScript tools
RUN npm install -g \
    # Code quality
    eslint \
    prettier \
    tslint \
    # Security scanning
    audit-ci \
    snyk \
    # Documentation
    jsdoc \
    typedoc \
    # Build analysis
    webpack-bundle-analyzer \
    # API tools
    @apidevtools/swagger-cli

# Install Go development tools
ENV GOPATH="/home/devuser/go"
ENV PATH="${GOPATH}/bin:${PATH}"
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest \
    && go install github.com/securecodewarrior/sast-scan@latest

# Install Docker tools for container analysis
RUN curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /home/devuser/.local/bin \
    && curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /home/devuser/.local/bin

# Add user bin to PATH
ENV PATH="/home/devuser/.local/bin:${PATH}"

# Set up git configuration template
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 --version && node --version

# Default command
CMD ["bash"]
