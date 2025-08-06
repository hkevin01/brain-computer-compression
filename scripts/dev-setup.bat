@echo off
REM Universal Development Environment Setup Script for Windows
REM Automatically detects and sets up the best development environment

echo 🧠 BCI Compression Toolkit - Universal Development Setup
echo =======================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Docker found
    docker info >nul 2>&1
    if %ERRORLEVEL% == 0 (
        echo ✅ Docker daemon is running
        goto :docker_setup
    ) else (
        echo ⚠️  Docker daemon is not running. Please start Docker.
        goto :python_setup
    )
) else (
    echo ❌ Docker not found. Using Python virtual environment.
    goto :python_setup
)

:docker_setup
echo 🐳 Setting up Docker development environment...

REM Build development containers
echo 📦 Building development containers...
docker-compose -f docker-compose.dev.yml build

REM Start development environment
echo 🚀 Starting development environment...
docker-compose -f docker-compose.dev.yml up -d

REM Wait for services
echo ⏱️  Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Install dependencies
echo 📚 Installing dependencies...
docker-compose -f docker-compose.dev.yml exec backend pip install -e .
docker-compose -f docker-compose.dev.yml exec backend pip install -r requirements-emg.txt

REM Run quick tests
echo 🧪 Running quick validation tests...
docker-compose -f docker-compose.dev.yml exec backend python tests/test_simple_validation.py

echo.
echo 🎉 Docker Development Environment Ready!
echo =======================================
echo 📊 Available services:
echo    • Jupyter Lab: http://localhost:8888
echo    • Frontend: http://localhost:3000
echo    • Backend API: http://localhost:8000
echo    • Grafana: http://localhost:3001
echo    • PostgreSQL: localhost:5432
echo.
echo 🔧 Useful commands:
echo    • make dev-status    - Check service status
echo    • make dev-logs      - View logs
echo    • make dev-shell     - Open shell
echo    • make dev-stop      - Stop environment
echo    • make test          - Run tests
goto :end

:python_setup
echo 🐍 Setting up Python virtual environment...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Python found
) else (
    echo ❌ Python not found. Please install Python 3.8+
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing Python dependencies...
pip install -r requirements.txt
if exist requirements-emg.txt (
    pip install -r requirements-emg.txt
)
pip install -e .

echo.
echo 🎉 Python Development Environment Ready!
echo ======================================
echo 🐍 Virtual environment: venv\
echo 📦 Dependencies installed
echo.
echo 🔧 Useful commands:
echo    • venv\Scripts\activate.bat     - Activate environment
echo    • python tests\test_simple_validation.py  - Run tests
echo    • jupyter lab                   - Start Jupyter

:end
echo.
echo ✅ Development environment setup complete!
echo.
echo 📚 Next steps:
echo    1. Read the documentation: docs\
echo    2. Try the examples: examples\
echo    3. Run the tests: make test
echo    4. Explore the algorithms: src\bci_compression\algorithms\
echo.
echo 🆘 Need help?
echo    • Check the documentation: README.md
echo    • View available commands: make help
echo    • Open an issue: https://github.com/hkevin01/brain-computer-compression/issues
pause
