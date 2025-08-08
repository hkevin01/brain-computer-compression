#!/bin/bash

# Test Docker build and diagnose issues
echo "🔍 Diagnosing Docker build issues..."

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found"
    exit 1
fi

# Check if requirements files exist
echo "📋 Checking dependencies..."
if [ ! -f "requirements-backend.txt" ]; then
    echo "❌ requirements-backend.txt not found"
    exit 1
else
    echo "✅ requirements-backend.txt found"
fi

if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt not found"
else
    echo "✅ requirements.txt found"
fi

# Check if source directories exist
echo "📁 Checking source structure..."
if [ ! -d "scripts" ]; then
    echo "❌ scripts/ directory not found"
    exit 1
else
    echo "✅ scripts/ directory found"
fi

if [ ! -d "src" ]; then
    echo "❌ src/ directory not found"
    exit 1
else
    echo "✅ src/ directory found"
fi

# Check if README.md exists
if [ ! -f "README.md" ]; then
    echo "❌ README.md not found"
    exit 1
else
    echo "✅ README.md found"
fi

# Try to build the Docker image
echo ""
echo "🔨 Attempting Docker build..."
echo "Command: docker build --no-cache -t bci-test ."

# Clear any cached layers first
docker builder prune -f > /dev/null 2>&1

if docker build --no-cache -t bci-test . ; then
    echo "✅ Docker build successful!"

    # Test the image
    echo "🧪 Testing the built image..."
    if docker run --rm bci-test python -c "import fastapi, uvicorn, numpy; print('All imports successful')" ; then
        echo "✅ Image test successful!"
        echo "🎉 Docker build is working correctly"
    else
        echo "❌ Image test failed"
    fi

    # Clean up test image
    docker rmi bci-test > /dev/null 2>&1

else
    echo "❌ Docker build failed"
    echo ""
    echo "💡 Common solutions:"
    echo "1. Check requirements-backend.txt for invalid dependencies"
    echo "2. Ensure all source files are present"
    echo "3. Check disk space: docker system df"
    echo "4. Clear Docker cache: docker system prune -af"
    echo "5. Check setup.py for file reading issues"
fi
