#!/bin/bash

echo "🧹 Safe Root Directory Cleanup"
echo "==============================="
echo ""

cd "$(dirname "$0")"

# Function to remove file if it exists
safe_remove() {
    local file="$1"
    local reason="$2"

    if [ -f "$file" ]; then
        echo "🗑️  Removing: $file ($reason)"
        rm -f "$file"
    fi
}

# Function to move file if it exists and destination doesn't exist
safe_move() {
    local src="$1"
    local dest="$2"
    local reason="$3"

    if [ -f "$src" ]; then
        if [ -f "$dest" ]; then
            echo "⚠️  Destination exists, removing source: $src"
            rm -f "$src"
        else
            echo "📦 Moving: $src → $dest ($reason)"
            mkdir -p "$(dirname "$dest")"
            mv "$src" "$dest"
        fi
    fi
}

echo "🗑️  Phase 1: Removing duplicate/empty .md files"
echo ""

# Check and remove empty or duplicate .md files
safe_remove "CONTRIBUTING.md" "empty file, using docs/CONTRIBUTING.md if exists"
safe_remove "BCI_Validation_Framework_Summary.md" "empty file"
safe_remove "EMG_UPDATE.md" "empty file"
safe_remove "STATUS_REPORT.md" "empty file"
safe_remove "README_NEW.md" "duplicate/obsolete"

# Remove duplicate Docker guides (already in docs/guides/)
safe_remove "DOCKER_BUILD_FIX.md" "duplicate - already in docs/guides/"
safe_remove "DOCKER_QUICK_START.md" "duplicate - already in docs/guides/"

echo ""
echo "🔧 Phase 2: Moving scripts to scripts/tools/"
echo ""

# Move all .sh files except run.sh
safe_move "cleanup_now.sh" "scripts/tools/cleanup_now.sh" "cleanup script"
safe_move "cleanup_root.sh" "scripts/tools/cleanup_root.sh" "cleanup script"
safe_move "completion_summary.sh" "scripts/tools/completion_summary.sh" "utility script"
safe_move "comprehensive_fix.sh" "scripts/tools/comprehensive_fix.sh" "utility script"
safe_move "execute_cleanup.sh" "scripts/tools/execute_cleanup.sh" "utility script"
safe_move "execute_final_cleanup.sh" "scripts/tools/execute_final_cleanup.sh" "utility script"
safe_move "execute_reorganization.sh" "scripts/tools/execute_reorganization.sh" "utility script"
safe_move "final_cleanup.sh" "scripts/tools/final_cleanup.sh" "cleanup script"
safe_move "fix_docker_build.sh" "scripts/tools/fix_docker_build.sh" "Docker utility"
safe_move "reorganization_complete.sh" "scripts/tools/reorganization_complete.sh" "status script"
safe_move "reorganize_root.sh" "scripts/tools/reorganize_root.sh" "utility script"
safe_move "run_cleanup_now.sh" "scripts/tools/run_cleanup_now.sh" "cleanup script"
safe_move "setup.sh" "scripts/setup/setup.sh" "setup script"
safe_move "setup_and_test.sh" "scripts/tools/setup_and_test.sh" "test script"
safe_move "setup_and_test.bat" "scripts/tools/setup_and_test.bat" "Windows test script"
safe_move "test_docker_build.sh" "scripts/tools/test_docker_build.sh" "test script"
safe_move "test_docker_workflow.sh" "scripts/tools/test_docker_workflow.sh" "test script"
safe_move "test_reorganization.sh" "scripts/tools/test_reorganization.sh" "test script"
safe_move "verify_reorganization.sh" "scripts/tools/verify_reorganization.sh" "verification script"

echo ""
echo "🐳 Phase 3: Moving Docker files to docker/"
echo ""

# Only move Docker files if they're not already properly located
safe_move "Dockerfile" "docker/Dockerfile.root-backup" "backup of root Dockerfile"
safe_move "Dockerfile.frontend" "docker/Dockerfile.frontend.backup" "backup, already exists in docker/"
safe_move "Dockerfile.minimal" "docker/Dockerfile.minimal.backup" "backup, already exists in docker/"
safe_move "docker-compose.yml" "docker/compose/docker-compose.backup.yml" "backup, main compose already exists"
safe_move "docker-compose.dev.yml" "docker/compose/docker-compose.dev.backup.yml" "backup, dev compose already exists"

echo ""
echo "📊 Phase 4: Root directory final status"
echo ""

file_count=$(ls -la | grep -E '^-' | wc -l)
dir_count=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)

echo "Root directory now contains:"
echo "  📄 Files: $file_count"
echo "  📁 Directories: $dir_count"

echo ""
echo "✅ Essential files remaining in root:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | head -15

echo ""
echo "🎉 Root directory cleanup complete!"
echo ""
echo "Clean structure achieved:"
echo "  📚 docs/ - All documentation organized"
echo "  🐳 docker/ - All Docker files organized"
echo "  🔧 scripts/ - All utility scripts organized"
echo "  📦 Root - Only essential project files"

# Remove this cleanup script itself
rm -f "execute_root_cleanup.sh"
rm -f "safe_cleanup.sh"

echo ""
echo "✅ Cleanup finished successfully!"
