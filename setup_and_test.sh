#!/bin/bash

# Brain-Computer Compression Toolkit - Quick Setup and Validation Script

set -e  # Exit on any error

echo "=========================================="
echo "Brain-Computer Compression Toolkit Setup"
echo "=========================================="

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing basic dependencies..."
    pip install numpy scipy matplotlib
fi

# Install EMG requirements if available
if [ -f "requirements-emg.txt" ]; then
    echo "📦 Installing EMG requirements..."
    pip install -r requirements-emg.txt
fi

# Install toolkit in development mode
echo "📦 Installing toolkit..."
pip install -e .

echo ""
echo "✅ Installation completed!"
echo ""

# Run dependency check
echo "🔍 Checking dependencies..."
cd tests
if python3 run_tests.py --dependencies-only; then
    echo "✅ All dependencies available"
else
    echo "❌ Some dependencies missing"
    exit 1
fi

echo ""
echo "🧪 Running quick validation tests..."

# Run quick tests
if python3 run_tests.py quick; then
    echo ""
    echo "🎉 SETUP SUCCESSFUL!"
    echo ""
    echo "The Brain-Computer Compression Toolkit is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment: source venv/bin/activate"
    echo "  2. Run standard tests: make test"
    echo "  3. Try the EMG demo: python examples/emg_demo.py"
    echo "  4. Run comprehensive tests: make test-comprehensive"
    echo ""
    echo "Available test commands:"
    echo "  make test-quick       # 2 minutes - basic functionality"
    echo "  make test-standard    # 10 minutes - recommended for development"
    echo "  make test-comprehensive # 30 minutes - full validation"
    echo ""
else
    echo ""
    echo "❌ SETUP VALIDATION FAILED"
    echo ""
    echo "Some quick tests failed. This could indicate:"
    echo "  - Missing dependencies"
    echo "  - Installation issues"
    echo "  - System compatibility problems"
    echo ""
    echo "Please check the error messages above and try:"
    echo "  1. Manual dependency installation: pip install numpy scipy matplotlib"
    echo "  2. Check Python version: python3 --version (need 3.8+)"
    echo "  3. Try individual tests: python tests/test_simple_validation.py"
    echo ""
    exit 1
fi
