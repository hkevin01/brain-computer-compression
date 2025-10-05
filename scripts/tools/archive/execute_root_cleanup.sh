#!/bin/bash

echo "🧹 Executing Root Directory Cleanup"
echo "===================================="
echo ""

cd /home/kevin/Projects/brain-computer-compression

# Function to safely move files
safe_move() {
    local src="$1"
    local dest="$2"

    if [ -f "$src" ]; then
        echo "Moving: $src → $dest"
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
        return 0
    else
        echo "⚠️  File not found: $src"
        return 1
    fi
}

echo "📄 Phase 1: Moving .md files to docs/"
echo ""

# Move documentation files
safe_move "CONTRIBUTING.md" "docs/CONTRIBUTING.md"
safe_move "BCI_Validation_Framework_Summary.md" "docs/project/BCI_Validation_Framework_Summary.md"
safe_move "EMG_UPDATE.md" "docs/project/EMG_UPDATE.md"
safe_move "STATUS_REPORT.md" "docs/project/STATUS_REPORT.md"
safe_move "README_NEW.md" "docs/archive/README_NEW.md"

# Move Docker guides (if not already moved)
safe_move "DOCKER_BUILD_FIX.md" "docs/guides/DOCKER_BUILD_FIX.md"
safe_move "DOCKER_QUICK_START.md" "docs/guides/DOCKER_QUICK_START.md"

echo ""
echo "🔧 Phase 2: Moving .sh scripts to scripts/"
echo ""

# Move setup scripts
safe_move "setup.sh" "scripts/setup/setup.sh"
safe_move "setup_and_test.sh" "scripts/tools/setup_and_test.sh"
safe_move "setup_and_test.bat" "scripts/tools/setup_and_test.bat"

# Move cleanup scripts
safe_move "cleanup_now.sh" "scripts/tools/cleanup_now.sh"
safe_move "cleanup_root.sh" "scripts/tools/cleanup_root.sh"
safe_move "execute_cleanup.sh" "scripts/tools/execute_cleanup.sh"
safe_move "execute_final_cleanup.sh" "scripts/tools/execute_final_cleanup.sh"
safe_move "final_cleanup.sh" "scripts/tools/final_cleanup.sh"
safe_move "run_cleanup_now.sh" "scripts/tools/run_cleanup_now.sh"

# Move reorganization scripts
safe_move "execute_reorganization.sh" "scripts/tools/execute_reorganization.sh"
safe_move "reorganization_complete.sh" "scripts/tools/reorganization_complete.sh"
safe_move "reorganize_root.sh" "scripts/tools/reorganize_root.sh"

# Move test and verification scripts
safe_move "test_docker_build.sh" "scripts/tools/test_docker_build.sh"
safe_move "test_docker_workflow.sh" "scripts/tools/test_docker_workflow.sh"
safe_move "test_reorganization.sh" "scripts/tools/test_reorganization.sh"
safe_move "verify_reorganization.sh" "scripts/tools/verify_reorganization.sh"

# Move build and fix scripts
safe_move "fix_docker_build.sh" "scripts/tools/fix_docker_build.sh"
safe_move "completion_summary.sh" "scripts/tools/completion_summary.sh"
safe_move "comprehensive_fix.sh" "scripts/tools/comprehensive_fix.sh"

echo ""
echo "🐳 Phase 3: Moving Docker files (if any remaining in root)"
echo ""

# Move any remaining Docker files in root
safe_move "Dockerfile" "docker/Dockerfile.root-backup"
safe_move "Dockerfile.frontend" "docker/Dockerfile.frontend.backup"
safe_move "Dockerfile.minimal" "docker/Dockerfile.minimal.backup"
safe_move "docker-compose.yml" "docker/compose/docker-compose.main.yml"
safe_move "docker-compose.dev.yml" "docker/compose/docker-compose.dev-backup.yml"
safe_move ".dockerignore" "docker/.dockerignore.backup"

echo ""
echo "⚙️  Phase 4: Moving config files"
echo ""

# Keep .env files in root but organize others
# These should be moved if they exist:
# (Most are likely already in proper locations)

echo ""
echo "📊 Phase 5: Final root directory analysis"
echo ""

file_count=$(ls -la | grep -E '^-' | wc -l)
dir_count=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)

echo "Root directory now contains:"
echo "  📄 Files: $file_count"
echo "  📁 Directories: $dir_count"

echo ""
echo "✅ Essential files in root:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | head -20

echo ""
echo "🎉 Root directory cleanup complete!"

# Remove this cleanup script itself
echo ""
echo "🗑️  Self-cleanup: removing this script"
rm -f "/home/kevin/Projects/brain-computer-compression/scripts/tools/cleanup_root_directory.sh"

echo "✅ Cleanup finished!"
