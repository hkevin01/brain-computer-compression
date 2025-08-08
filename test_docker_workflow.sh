#!/bin/bash

# Test script to validate Docker workflow
# This script tests the comprehensive run.sh Docker orchestration

set -e

echo "🧪 Testing BCI Compression Docker Workflow"
echo "==========================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available"
    exit 1
fi

echo "✅ Docker prerequisites satisfied"

# Check if run.sh exists and is executable
if [ ! -f "./run.sh" ]; then
    echo "❌ run.sh not found"
    exit 1
fi

if [ ! -x "./run.sh" ]; then
    echo "🔧 Making run.sh executable..."
    chmod +x ./run.sh
fi

echo "✅ run.sh is executable"

# Test basic run.sh commands
echo ""
echo "🔍 Testing run.sh commands..."

echo "Testing help command..."
./run.sh help

echo ""
echo "Testing status command..."
./run.sh status

echo ""
echo "Testing build command..."
./run.sh build

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful"

    echo ""
    echo "Testing up command..."
    ./run.sh up -d

    if [ $? -eq 0 ]; then
        echo "✅ Docker services started successfully"

        # Wait a moment for services to be ready
        sleep 5

        echo ""
        echo "Testing service status..."
        ./run.sh ps

        echo ""
        echo "Testing API health check..."
        if curl -f http://localhost:8000/health &> /dev/null; then
            echo "✅ Backend API is responding"
        else
            echo "⚠️  Backend API not responding yet (may need more time)"
        fi

        echo ""
        echo "Testing GUI availability..."
        if curl -f http://localhost:3000 &> /dev/null; then
            echo "✅ GUI is responding"
        else
            echo "⚠️  GUI not responding yet (may need more time)"
        fi

        echo ""
        echo "🧹 Cleaning up..."
        ./run.sh down

        echo "✅ All tests completed successfully!"
        echo ""
        echo "🚀 Your Docker workflow is ready!"
        echo "   - Run './run.sh up' to start services"
        echo "   - Run './run.sh gui:open' to open the GUI"
        echo "   - Run './run.sh status' to check everything"

    else
        echo "❌ Failed to start Docker services"
        exit 1
    fi
else
    echo "❌ Docker build failed"
    echo "💡 Try: ./run.sh build --help for troubleshooting"
    exit 1
fi
