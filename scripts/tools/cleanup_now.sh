#!/bin/bash

echo "🧹 Automated Root Directory Cleanup"
echo "=================================="

# Remove duplicate/unnecessary files
echo "🗑️  Removing unnecessary files..."

FILES_REMOVED=0

# Remove README_NEW.md (duplicate)
if [ -f "README_NEW.md" ]; then
    rm "README_NEW.md" && echo "  ✅ Removed README_NEW.md" && ((FILES_REMOVED++))
fi

# Remove backup docker-compose files
if [ -f "docker-compose.dev.yml.backup" ]; then
    rm "docker-compose.dev.yml.backup" && echo "  ✅ Removed docker-compose.dev.yml.backup" && ((FILES_REMOVED++))
fi

# Remove temporary test files
if [ -f "test_output.log" ]; then
    rm "test_output.log" && echo "  ✅ Removed test_output.log" && ((FILES_REMOVED++))
fi

# Remove coverage file (will be regenerated)
if [ -f ".coverage" ]; then
    rm ".coverage" && echo "  ✅ Removed .coverage" && ((FILES_REMOVED++))
fi

# Remove root package.json (should only be in dashboard/)
if [ -f "package.json" ] && [ -f "dashboard/package.json" ]; then
    rm "package.json" && echo "  ✅ Removed root package.json (kept dashboard/package.json)" && ((FILES_REMOVED++))
fi

# Remove temporary Dockerfile variants
if [ -f "Dockerfile.minimal" ]; then
    rm "Dockerfile.minimal" && echo "  ✅ Removed Dockerfile.minimal" && ((FILES_REMOVED++))
fi

if [ -f "Dockerfile.frontend" ]; then
    rm "Dockerfile.frontend" && echo "  ✅ Removed Dockerfile.frontend" && ((FILES_REMOVED++))
fi

# Move benchmark results to proper location
if [ -f "benchmark_results.json" ]; then
    mkdir -p .benchmarks
    mv "benchmark_results.json" ".benchmarks/" && echo "  ✅ Moved benchmark_results.json to .benchmarks/" && ((FILES_REMOVED++))
fi

echo "🗑️  Removing cache directories..."

DIRS_REMOVED=0

# Remove Python cache directories
if [ -d ".mypy_cache" ]; then
    rm -rf ".mypy_cache" && echo "  ✅ Removed .mypy_cache/" && ((DIRS_REMOVED++))
fi

if [ -d ".pytest_cache" ]; then
    rm -rf ".pytest_cache" && echo "  ✅ Removed .pytest_cache/" && ((DIRS_REMOVED++))
fi

# Remove virtual environments if they exist in root
if [ -d "venv" ]; then
    rm -rf "venv" && echo "  ✅ Removed venv/" && ((DIRS_REMOVED++))
fi

if [ -d ".venv" ]; then
    rm -rf ".venv" && echo "  ✅ Removed .venv/" && ((DIRS_REMOVED++))
fi

echo ""
echo "📊 Cleanup Summary:"
echo "  Files removed: $FILES_REMOVED"
echo "  Directories removed: $DIRS_REMOVED"

# Update .gitignore to prevent future clutter
echo "📝 Updating .gitignore to prevent future clutter..."

# Check if cleanup section already exists
if ! grep -q "# Cleanup: Prevent future clutter" .gitignore; then
    cat >> .gitignore << 'EOF'

# Cleanup: Prevent future clutter
test_output.log
benchmark_results.json
.coverage
README_NEW.md
Dockerfile.minimal
Dockerfile.frontend
*.backup
/package.json
EOF
    echo "  ✅ Added cleanup rules to .gitignore"
else
    echo "  ✅ .gitignore already has cleanup rules"
fi

echo ""
echo "✅ Root directory cleanup complete!"
echo ""
echo "📁 Current root directory structure:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"
