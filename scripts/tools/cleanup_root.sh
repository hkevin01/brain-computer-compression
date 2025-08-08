#!/bin/bash

# Root Directory Cleanup Script for brain-computer-compression
echo "🧹 Starting root directory cleanup..."

# Files to remove (duplicates, temporary, or unnecessary)
FILES_TO_REMOVE=(
    "README_NEW.md"                    # Duplicate of README.md
    "Dockerfile.frontend"              # Superseded by dashboard/Dockerfile
    "Dockerfile.minimal"               # Created as backup, not needed
    "docker-compose.dev.yml.backup"   # Backup file
    "test_output.log"                  # Temporary test output
    "benchmark_results.json"           # Should be in .benchmarks/
    ".coverage"                        # Coverage data, regenerated
    "package.json"                     # Should be in dashboard/ only
)

# Directories to remove (if empty or redundant)
DIRS_TO_REMOVE=(
    ".mypy_cache"                      # Temporary mypy cache
    ".pytest_cache"                    # Temporary pytest cache
    "venv"                             # User virtual environment
    ".venv"                            # Alternative venv location
)

echo "📋 Files marked for removal:"
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo "📋 Directories marked for removal:"
for dir in "${DIRS_TO_REMOVE[@]}"; do
    if [ -d "$dir" ]; then
        echo "  - $dir/"
    fi
done

echo ""
read -p "Proceed with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo "🗑️  Removing files..."
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        rm "$file" && echo "  ✅ Removed $file"
    fi
done

echo "🗑️  Removing directories..."
for dir in "${DIRS_TO_REMOVE[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir" && echo "  ✅ Removed $dir/"
    fi
done

# Move files to better locations
echo "📁 Reorganizing files..."

# Move benchmark results if it exists
if [ -f "benchmark_results.json" ]; then
    mkdir -p .benchmarks
    mv benchmark_results.json .benchmarks/ && echo "  ✅ Moved benchmark_results.json to .benchmarks/"
fi

# Create gitignore entries for cleanup
echo "📝 Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Cleanup: Prevent future clutter
test_output.log
benchmark_results.json
.coverage
README_NEW.md
Dockerfile.minimal
Dockerfile.frontend
*.backup
EOF

echo "✅ Root directory cleanup complete!"
echo ""
echo "📊 Remaining files in root:"
ls -la | grep -E '^-' | wc -l
echo "📊 Remaining directories in root:"
ls -la | grep -E '^d' | wc -l
