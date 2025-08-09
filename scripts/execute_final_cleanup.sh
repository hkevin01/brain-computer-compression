#!/bin/bash

echo "🧹 Executing final repository cleanup..."
echo ""

# Function to move file if it exists
move_if_exists() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        mv "$src" "$dest" && echo "  ✅ Moved $src → $dest"
        return 0
    else
        echo "  ⚪ $src already moved or doesn't exist"
        return 1
    fi
}

# Function to move directory if it exists
move_dir_if_exists() {
    local src="$1"
    local dest="$2"
    if [ -d "$src" ]; then
        mv "$src" "$dest" && echo "  ✅ Moved $src/ → $dest/"
        return 0
    else
        echo "  ⚪ $src/ already moved or doesn't exist"
        return 1
    fi
}

echo "📁 Moving build configuration files to config/build/..."
move_if_exists "pyproject.toml" "config/build/"
move_if_exists "pytest.ini" "config/build/"
move_if_exists "setup.py" "config/build/"

echo ""
echo "🔧 Moving development configuration files to config/development/..."
move_if_exists ".flake8" "config/development/"

echo ""
echo "🌍 Moving environment files to config/environments/..."
move_if_exists ".env.production" "config/environments/"
move_if_exists ".env.template" "config/environments/"

echo ""
echo "🐳 Moving deployment files to deployment/..."
mkdir -p deployment
move_if_exists "Makefile" "deployment/"
move_dir_if_exists "kubernetes" "deployment/"

echo ""
echo "🔧 Moving cleanup scripts to scripts/tools/..."
move_if_exists "completion_summary.sh" "scripts/tools/"
move_if_exists "execute_cleanup.sh" "scripts/tools/"
move_if_exists "execute_reorganization.sh" "scripts/tools/"
move_if_exists "reorganize_root.sh" "scripts/tools/"
move_if_exists "test_reorganization.sh" "scripts/tools/"
move_if_exists "run_cleanup_now.sh" "scripts/tools/"
move_if_exists "final_cleanup.sh" "scripts/tools/"

echo ""
echo "📊 Final root directory analysis:"
echo -n "  Files in root: "
ls -la | grep -E '^-' | wc -l
echo -n "  Directories in root: "
ls -la | grep -E '^d' | grep -v '^total' | wc -l

echo ""
echo "✅ Repository reorganization complete!"
echo ""
echo "📁 New clean structure achieved:"
echo "  config/           - Configuration files organized by type"
echo "  deployment/       - Deployment and infrastructure files"
echo "  docker/           - Docker configuration and compose files"
echo "  docs/             - Documentation organized by purpose"
echo "  scripts/          - Utility scripts organized by function"
echo ""
echo "🚀 Test the reorganized project:"
echo "  ./run.sh status   # Check project status"
echo "  ./run.sh build    # Build Docker images"
echo "  ./run.sh up       # Start services"
