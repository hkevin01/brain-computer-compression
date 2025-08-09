#!/bin/bash

# Test script to validate the reorganization worked correctly
echo "🧪 Testing Reorganization Results"
echo "================================"

# Check that directories were created
echo "📁 Checking directory structure..."

EXPECTED_DIRS=(
    "docs/guides"
    "docs/project"
    "docker"
    "docker/compose"
    "scripts/setup"
    "scripts/tools"
)

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ exists"
    else
        echo "  ❌ $dir/ missing"
    fi
done

# Check that key files were moved correctly
echo ""
echo "📄 Checking file movements..."

FILE_CHECKS=(
    "docker/Dockerfile:Main Dockerfile"
    "docker/compose/docker-compose.yml:Docker Compose file"
    "docs/guides/DOCKER_QUICK_START.md:Docker guide"
    "docs/CHANGELOG.md:Changelog"
    "scripts/setup/setup.sh:Setup script"
    "scripts/tools/test_docker_build.sh:Docker test script"
)

for check in "${FILE_CHECKS[@]}"; do
    file="${check%%:*}"
    desc="${check##*:}"
    if [ -f "$file" ]; then
        echo "  ✅ $desc: $file"
    else
        echo "  ⚠️  $desc: $file (missing)"
    fi
done

# Check run.sh was updated correctly
echo ""
echo "🔧 Checking run.sh updates..."

if [ -f "run.sh" ]; then
    if grep -q "docker/Dockerfile" run.sh; then
        echo "  ✅ run.sh references docker/Dockerfile"
    else
        echo "  ⚠️  run.sh may not reference new Dockerfile location"
    fi

    if grep -q "docker/compose/docker-compose.yml" run.sh; then
        echo "  ✅ run.sh references docker/compose/docker-compose.yml"
    else
        echo "  ⚠️  run.sh may not reference new compose file location"
    fi
else
    echo "  ❌ run.sh not found"
fi

# Check root directory cleanliness
echo ""
echo "📊 Root directory analysis..."

ROOT_FILES=$(ls -la | grep -E '^-' | wc -l)
ROOT_DIRS=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)

echo "  Files in root: $ROOT_FILES"
echo "  Directories in root: $ROOT_DIRS"

if [ "$ROOT_FILES" -lt 15 ]; then
    echo "  ✅ Root directory is clean (< 15 files)"
else
    echo "  ⚠️  Root directory still has many files"
fi

# Test Docker build (if possible)
echo ""
echo "🐳 Testing Docker functionality..."

if [ -f "docker/Dockerfile" ]; then
    echo "  ✅ Docker/Dockerfile exists"

    # Test if Docker is available
    if command -v docker >/dev/null 2>&1; then
        echo "  ✅ Docker command available"

        # Test build (dry run)
        if docker build --help >/dev/null 2>&1; then
            echo "  ✅ Docker build command works"
            echo "  💡 You can now test: ./run.sh build"
        else
            echo "  ⚠️  Docker build command issues"
        fi
    else
        echo "  ⚠️  Docker not available for testing"
    fi
else
    echo "  ❌ Docker/Dockerfile missing"
fi

echo ""
echo "🎉 Reorganization Test Complete!"

# Count remaining issues
ISSUES=0
[ ! -d "docs/guides" ] && ((ISSUES++))
[ ! -d "docker" ] && ((ISSUES++))
[ ! -f "docker/Dockerfile" ] && ((ISSUES++))
[ ! -f "run.sh" ] && ((ISSUES++))

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ All critical components in place"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Test build: ./run.sh build"
    echo "  2. Start services: ./run.sh up"
    echo "  3. Check documentation: ls docs/guides/"
else
    echo "⚠️  Found $ISSUES critical issues"
    echo "💡 Re-run the reorganization script if needed"
fi
