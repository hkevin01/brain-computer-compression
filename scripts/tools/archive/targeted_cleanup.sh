#!/bin/bash

echo "🧹 Targeted Root Directory Cleanup"
echo "==================================="
echo "Removing empty/duplicate files and moving essential scripts"
echo ""

# Remove empty .md files
echo "📄 Removing empty .md files..."
[ -f "CONTRIBUTING.md" ] && [ ! -s "CONTRIBUTING.md" ] && rm "CONTRIBUTING.md" && echo "  🗑️  Removed empty CONTRIBUTING.md"
[ -f "BCI_Validation_Framework_Summary.md" ] && [ ! -s "BCI_Validation_Framework_Summary.md" ] && rm "BCI_Validation_Framework_Summary.md" && echo "  🗑️  Removed empty BCI_Validation_Framework_Summary.md"
[ -f "EMG_UPDATE.md" ] && [ ! -s "EMG_UPDATE.md" ] && rm "EMG_UPDATE.md" && echo "  🗑️  Removed empty EMG_UPDATE.md"
[ -f "STATUS_REPORT.md" ] && [ ! -s "STATUS_REPORT.md" ] && rm "STATUS_REPORT.md" && echo "  🗑️  Removed empty STATUS_REPORT.md"
[ -f "README_NEW.md" ] && rm "README_NEW.md" && echo "  🗑️  Removed obsolete README_NEW.md"

# Remove duplicate Docker guides (already in docs/guides/)
[ -f "DOCKER_BUILD_FIX.md" ] && rm "DOCKER_BUILD_FIX.md" && echo "  🗑️  Removed duplicate DOCKER_BUILD_FIX.md"
[ -f "DOCKER_QUICK_START.md" ] && rm "DOCKER_QUICK_START.md" && echo "  🗑️  Removed duplicate DOCKER_QUICK_START.md"

echo ""
echo "🔧 Moving scripts to proper locations..."

# Move reorganization script that has content
[ -f "reorganization_complete.sh" ] && mv "reorganization_complete.sh" "scripts/tools/reorganization_complete.sh" && echo "  📦 Moved reorganization_complete.sh → scripts/tools/"

# Move setup scripts
[ -f "setup.sh" ] && mv "setup.sh" "scripts/setup/setup.sh" && echo "  📦 Moved setup.sh → scripts/setup/"
[ -f "setup_and_test.sh" ] && mv "setup_and_test.sh" "scripts/tools/setup_and_test.sh" && echo "  📦 Moved setup_and_test.sh → scripts/tools/"
[ -f "setup_and_test.bat" ] && mv "setup_and_test.bat" "scripts/tools/setup_and_test.bat" && echo "  📦 Moved setup_and_test.bat → scripts/tools/"

# Move utility scripts
[ -f "completion_summary.sh" ] && mv "completion_summary.sh" "scripts/tools/completion_summary.sh" && echo "  📦 Moved completion_summary.sh → scripts/tools/"
[ -f "execute_cleanup.sh" ] && mv "execute_cleanup.sh" "scripts/tools/execute_cleanup.sh" && echo "  📦 Moved execute_cleanup.sh → scripts/tools/"
[ -f "execute_final_cleanup.sh" ] && mv "execute_final_cleanup.sh" "scripts/tools/execute_final_cleanup.sh" && echo "  📦 Moved execute_final_cleanup.sh → scripts/tools/"
[ -f "execute_reorganization.sh" ] && mv "execute_reorganization.sh" "scripts/tools/execute_reorganization.sh" && echo "  📦 Moved execute_reorganization.sh → scripts/tools/"
[ -f "final_cleanup.sh" ] && mv "final_cleanup.sh" "scripts/tools/final_cleanup.sh" && echo "  📦 Moved final_cleanup.sh → scripts/tools/"
[ -f "reorganize_root.sh" ] && mv "reorganize_root.sh" "scripts/tools/reorganize_root.sh" && echo "  📦 Moved reorganize_root.sh → scripts/tools/"
[ -f "run_cleanup_now.sh" ] && mv "run_cleanup_now.sh" "scripts/tools/run_cleanup_now.sh" && echo "  📦 Moved run_cleanup_now.sh → scripts/tools/"
[ -f "test_reorganization.sh" ] && mv "test_reorganization.sh" "scripts/tools/test_reorganization.sh" && echo "  📦 Moved test_reorganization.sh → scripts/tools/"
[ -f "verify_reorganization.sh" ] && mv "verify_reorganization.sh" "scripts/tools/verify_reorganization.sh" && echo "  📦 Moved verify_reorganization.sh → scripts/tools/"

# Remove empty cleanup scripts (content is already in scripts/tools/)
[ -f "cleanup_now.sh" ] && [ ! -s "cleanup_now.sh" ] && rm "cleanup_now.sh" && echo "  🗑️  Removed empty cleanup_now.sh"
[ -f "cleanup_root.sh" ] && [ ! -s "cleanup_root.sh" ] && rm "cleanup_root.sh" && echo "  🗑️  Removed empty cleanup_root.sh"

echo ""
echo "🐳 Moving Docker files..."

# Move Docker files (keep as backups since docker/ already has proper files)
[ -f "Dockerfile" ] && mv "Dockerfile" "docker/Dockerfile.root-backup" && echo "  📦 Moved Dockerfile → docker/Dockerfile.root-backup"
[ -f "Dockerfile.frontend" ] && mv "Dockerfile.frontend" "docker/Dockerfile.frontend.backup2" && echo "  📦 Moved Dockerfile.frontend → docker/Dockerfile.frontend.backup2"
[ -f "Dockerfile.minimal" ] && mv "Dockerfile.minimal" "docker/Dockerfile.minimal.backup2" && echo "  📦 Moved Dockerfile.minimal → docker/Dockerfile.minimal.backup2"
[ -f "docker-compose.yml" ] && mv "docker-compose.yml" "docker/compose/docker-compose.root-backup.yml" && echo "  📦 Moved docker-compose.yml → docker/compose/docker-compose.root-backup.yml"
[ -f "docker-compose.dev.yml" ] && mv "docker-compose.dev.yml" "docker/compose/docker-compose.dev-backup2.yml" && echo "  📦 Moved docker-compose.dev.yml → docker/compose/docker-compose.dev-backup2.yml"

echo ""
echo "📊 Root directory status:"

file_count=$(ls -la | grep -E '^-' | wc -l)
echo "  📄 Files remaining in root: $file_count"

echo ""
echo "✅ Essential files remaining:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | head -15

echo ""
echo "🎉 Cleanup complete! Root directory is now organized."

# Remove cleanup scripts
rm -f "execute_root_cleanup.sh" 2>/dev/null
rm -f "safe_cleanup.sh" 2>/dev/null
rm -f "targeted_cleanup.sh" 2>/dev/null
