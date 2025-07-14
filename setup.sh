#!/bin/bash
# Setup script for BCI Compression Toolkit

echo "Setting up Brain-Computer Interface Compression Toolkit..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip
pip install numpy scipy matplotlib pandas PyWavelets scikit-learn

# Try to install optional dependencies
echo "Installing optional dependencies..."
pip install torch || echo "Warning: PyTorch installation failed"

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest black flake8

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run tests, use: python tests/validate_phase2.py"
