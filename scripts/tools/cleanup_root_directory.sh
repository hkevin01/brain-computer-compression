#!/bin/bash

echo "🧹 Cleaning up root directory - moving files to proper subdirectories"
echo "=================================================================="
echo ""

# Function to move file with confirmation
move_file() {
    local src="$1"
    local dest="$2"

    if [ -f "$src" ]; then
        echo "Moving $src → $dest"
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
    else
        echo "⚠️  $src not found, skipping"
    fi
}

echo "📄 Moving .md files to docs/ directory..."

# Move documentation files to docs/
move_file "CONTRIBUTING.md" "docs/CONTRIBUTING.md"
move_file "BCI_Validation_Framework_Summary.md" "docs/project/BCI_Validation_Framework_Summary.md"
move_file "EMG_UPDATE.md" "docs/project/EMG_UPDATE.md"
move_file "STATUS_REPORT.md" "docs/project/STATUS_REPORT.md"
move_file "README_NEW.md" "docs/archive/README_NEW.md"

# Move Docker guides to proper location
move_file "DOCKER_BUILD_FIX.md" "docs/guides/DOCKER_BUILD_FIX.md"
move_file "DOCKER_QUICK_START.md" "docs/guides/DOCKER_QUICK_START.md"

echo ""
echo "🐳 Moving Docker files to docker/ directory..."

# Move Docker files
move_file "Dockerfile" "docker/Dockerfile.backup"
move_file "Dockerfile.frontend" "docker/Dockerfile.frontend"
move_file "Dockerfile.minimal" "docker/Dockerfile.minimal"
move_file "docker-compose.yml" "docker/compose/docker-compose.main.yml"
move_file "docker-compose.dev.yml" "docker/compose/docker-compose.dev.yml"

echo ""
echo "🔧 Moving script files to scripts/ directory..."

# Move setup and utility scripts
move_file "setup.sh" "scripts/setup/setup.sh"
move_file "setup_and_test.sh" "scripts/tools/setup_and_test.sh"
move_file "setup_and_test.bat" "scripts/tools/setup_and_test.bat"

# Move cleanup scripts
move_file "cleanup_now.sh" "scripts/tools/cleanup_now.sh"
move_file "cleanup_root.sh" "scripts/tools/cleanup_root.sh"
move_file "execute_cleanup.sh" "scripts/tools/execute_cleanup.sh"
move_file "execute_final_cleanup.sh" "scripts/tools/execute_final_cleanup.sh"
move_file "final_cleanup.sh" "scripts/tools/final_cleanup.sh"
move_file "run_cleanup_now.sh" "scripts/tools/run_cleanup_now.sh"

# Move reorganization scripts
move_file "execute_reorganization.sh" "scripts/tools/execute_reorganization.sh"
move_file "reorganization_complete.sh" "scripts/tools/reorganization_complete.sh"
move_file "reorganize_root.sh" "scripts/tools/reorganize_root.sh"

# Move testing and verification scripts
move_file "test_docker_build.sh" "scripts/tools/test_docker_build.sh"
move_file "test_docker_workflow.sh" "scripts/tools/test_docker_workflow.sh"
move_file "test_reorganization.sh" "scripts/tools/test_reorganization.sh"
move_file "verify_reorganization.sh" "scripts/tools/verify_reorganization.sh"

# Move Docker build scripts
move_file "fix_docker_build.sh" "scripts/tools/fix_docker_build.sh"

# Move completion and summary scripts
move_file "completion_summary.sh" "scripts/tools/completion_summary.sh"
move_file "comprehensive_fix.sh" "scripts/tools/comprehensive_fix.sh"

echo ""
echo "⚙️  Moving configuration files to config/ directory..."

# Move configuration files
move_file ".dockerignore" "docker/.dockerignore"

echo ""
echo "📊 Root directory status after cleanup:"

# Count remaining files in root
file_count=$(ls -la | grep -E '^-' | wc -l)
dir_count=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)

echo "  Files in root: $file_count"
echo "  Directories in root: $dir_count"

echo ""
echo "✅ Essential files remaining in root:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $'

echo ""
echo "🎉 Root directory cleanup complete!"
echo ""
echo "Files should now be organized as follows:"
echo "  📚 docs/ - All documentation and guides"
echo "  🐳 docker/ - All Docker-related files"
echo "  🔧 scripts/ - All utility and setup scripts"
echo "  ⚙️  config/ - Configuration files"
echo "  📦 Root - Only essential project files"
