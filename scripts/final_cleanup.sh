#!/bin/bash

echo "🧹 Manual cleanup of remaining files..."
echo ""

# Move config files that are still in root
echo "📁 Moving configuration files..."
mkdir -p config/build
mkdir -p config/development

# Move build config files
if [ -f "pyproject.toml" ]; then
    mv pyproject.toml config/build/ && echo "  ✅ Moved pyproject.toml → config/build/"
fi

if [ -f "pytest.ini" ]; then
    mv pytest.ini config/build/ && echo "  ✅ Moved pytest.ini → config/build/"
fi

if [ -f "setup.py" ]; then
    mv setup.py config/build/ && echo "  ✅ Moved setup.py → config/build/"
fi

# Move development config files
if [ -f ".flake8" ]; then
    mv .flake8 config/development/ && echo "  ✅ Moved .flake8 → config/development/"
fi

# Move environment files
mkdir -p config/environments
if [ -f ".env.production" ]; then
    mv .env.production config/environments/ && echo "  ✅ Moved .env.production → config/environments/"
fi

if [ -f ".env.template" ]; then
    mv .env.template config/environments/ && echo "  ✅ Moved .env.template → config/environments/"
fi

# Move remaining scripts to scripts/tools
echo ""
echo "🔧 Moving remaining scripts..."
mkdir -p scripts/tools

CLEANUP_SCRIPTS=(
    "completion_summary.sh"
    "execute_cleanup.sh"
    "execute_reorganization.sh"
    "reorganize_root.sh"
    "test_reorganization.sh"
    "run_cleanup_now.sh"
)

for script in "${CLEANUP_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" scripts/tools/ && echo "  ✅ Moved $script → scripts/tools/"
    fi
done

# Move deployment files
echo ""
echo "🐳 Moving deployment files..."
mkdir -p deployment

if [ -f "Makefile" ]; then
    mv Makefile deployment/ && echo "  ✅ Moved Makefile → deployment/"
fi

if [ -d "kubernetes" ]; then
    mv kubernetes deployment/ && echo "  ✅ Moved kubernetes/ → deployment/"
fi

echo ""
echo "📊 Updated root directory contents:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"

echo ""
echo "✅ Manual cleanup complete!"
echo ""
echo "📁 New organized structure:"
echo "  config/build/         - Build configuration files"
echo "  config/development/   - Development tools config"
echo "  config/environments/  - Environment configurations"
echo "  deployment/           - Deployment files"
echo "  scripts/tools/        - Utility and cleanup scripts"

echo ""
echo "🚀 Test the organized project:"
echo "  ./run.sh status       # Check project status"
echo "  ./run.sh build        # Build Docker images"
