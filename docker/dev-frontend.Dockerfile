# Frontend Development Container
# Optimized for modern JavaScript/TypeScript development with all major frameworks
FROM node:18-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    python3 \
    make \
    g++ \
    # Development utilities
    git \
    curl \
    wget \
    vim \
    tree \
    # Browser automation dependencies
    libnss3-dev \
    libatk-bridge2.0-dev \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create development user
RUN groupadd -r devuser && useradd -r -g devuser devuser
RUN mkdir -p /home/devuser && chown devuser:devuser /home/devuser
USER devuser
WORKDIR /workspace

# Install global development tools
RUN npm install -g \
    # Package managers
    yarn \
    pnpm \
    # Build tools
    vite \
    webpack-cli \
    parcel \
    esbuild \
    # Development tools
    typescript \
    ts-node \
    nodemon \
    # Code quality
    eslint \
    prettier \
    # Testing
    jest \
    playwright \
    cypress \
    # Documentation
    typedoc \
    # Utilities
    serve \
    http-server \
    concurrently

# Install Playwright browsers (for testing)
RUN npx playwright install --with-deps chromium firefox webkit

# Set up development environment
ENV NODE_ENV=development
ENV NPM_CONFIG_PREFIX=/home/devuser/.npm-global
ENV PATH="/home/devuser/.npm-global/bin:${PATH}"

# Expose common frontend development ports
EXPOSE 3000 3001 4000 4173 5173 8080 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node --version

# Default command for development server
CMD ["npm", "run", "dev"]
