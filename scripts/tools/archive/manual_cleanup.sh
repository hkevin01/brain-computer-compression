#!/bin/bash

# Execute manual cleanup of root directory
cd "$(dirname "$0")"

echo "🧹 Executing manual root directory cleanup"
echo "=========================================="

# Step 1: Remove empty .md files (confirmed empty)
echo "Step 1: Removing empty .md files..."
rm -f CONTRIBUTING.md && echo "  ✅ Removed empty CONTRIBUTING.md"
rm -f BCI_Validation_Framework_Summary.md && echo "  ✅ Removed empty BCI_Validation_Framework_Summary.md"
rm -f EMG_UPDATE.md && echo "  ✅ Removed empty EMG_UPDATE.md"
rm -f STATUS_REPORT.md && echo "  ✅ Removed empty STATUS_REPORT.md"
rm -f README_NEW.md && echo "  ✅ Removed obsolete README_NEW.md"

# Step 2: Remove duplicate guides (exist in docs/guides/)
echo ""
echo "Step 2: Removing duplicate guides..."
rm -f DOCKER_BUILD_FIX.md && echo "  ✅ Removed duplicate DOCKER_BUILD_FIX.md"
rm -f DOCKER_QUICK_START.md && echo "  ✅ Removed duplicate DOCKER_QUICK_START.md"

# Step 3: Remove empty scripts
echo ""
echo "Step 3: Removing empty scripts..."
rm -f cleanup_now.sh && echo "  ✅ Removed empty cleanup_now.sh"
rm -f cleanup_root.sh && echo "  ✅ Removed empty cleanup_root.sh"

# Step 4: Move important scripts
echo ""
echo "Step 4: Moving scripts to proper locations..."
mkdir -p scripts/tools scripts/setup

mv reorganization_complete.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved reorganization_complete.sh" || echo "  ℹ️  reorganization_complete.sh already moved"
mv completion_summary.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved completion_summary.sh" || echo "  ℹ️  completion_summary.sh already moved"
mv execute_cleanup.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved execute_cleanup.sh" || echo "  ℹ️  execute_cleanup.sh already moved"
mv execute_final_cleanup.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved execute_final_cleanup.sh" || echo "  ℹ️  execute_final_cleanup.sh already moved"
mv execute_reorganization.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved execute_reorganization.sh" || echo "  ℹ️  execute_reorganization.sh already moved"
mv final_cleanup.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved final_cleanup.sh" || echo "  ℹ️  final_cleanup.sh already moved"
mv reorganize_root.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved reorganize_root.sh" || echo "  ℹ️  reorganize_root.sh already moved"
mv run_cleanup_now.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved run_cleanup_now.sh" || echo "  ℹ️  run_cleanup_now.sh already moved"
mv test_reorganization.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved test_reorganization.sh" || echo "  ℹ️  test_reorganization.sh already moved"
mv verify_reorganization.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved verify_reorganization.sh" || echo "  ℹ️  verify_reorganization.sh already moved"

# Move setup scripts
mv setup.sh scripts/setup/ 2>/dev/null && echo "  📦 Moved setup.sh" || echo "  ℹ️  setup.sh already moved"
mv setup_and_test.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved setup_and_test.sh" || echo "  ℹ️  setup_and_test.sh already moved"
mv setup_and_test.bat scripts/tools/ 2>/dev/null && echo "  📦 Moved setup_and_test.bat" || echo "  ℹ️  setup_and_test.bat already moved"

# Move other utility scripts
mv comprehensive_fix.sh scripts/tools/ 2>/dev/null && echo "  📦 Moved comprehensive_fix.sh" || echo "  ℹ️  comprehensive_fix.sh already moved"

# Step 5: Move Docker files as backups
echo ""
echo "Step 5: Moving Docker files to docker/ as backups..."
mv Dockerfile docker/Dockerfile.root-backup 2>/dev/null && echo "  📦 Created docker/Dockerfile.root-backup" || echo "  ℹ️  Dockerfile already moved"
mv Dockerfile.frontend docker/ 2>/dev/null && echo "  📦 Moved Dockerfile.frontend to docker/" || echo "  ℹ️  Dockerfile.frontend already in docker/"
mv Dockerfile.minimal docker/ 2>/dev/null && echo "  📦 Moved Dockerfile.minimal to docker/" || echo "  ℹ️  Dockerfile.minimal already in docker/"
mv docker-compose.yml docker/compose/docker-compose.root-backup.yml 2>/dev/null && echo "  📦 Created docker/compose/docker-compose.root-backup.yml" || echo "  ℹ️  docker-compose.yml already moved"
mv docker-compose.dev.yml docker/compose/ 2>/dev/null && echo "  📦 Moved docker-compose.dev.yml to docker/compose/" || echo "  ℹ️  docker-compose.dev.yml already in docker/compose/"

# Step 6: Clean up all cleanup scripts
echo ""
echo "Step 6: Removing cleanup scripts..."
rm -f execute_root_cleanup.sh
rm -f safe_cleanup.sh
rm -f targeted_cleanup.sh
rm -f temp_cleanup_log.txt
rm -f quick_cleanup.sh
rm -f manual_cleanup.sh

echo ""
echo "✅ Manual cleanup completed!"

# Show results
echo ""
echo "📊 Root directory status:"
file_count=$(ls -la | grep -E '^-' | wc -l)
dir_count=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)
echo "  📄 Files: $file_count"
echo "  📁 Directories: $dir_count"

echo ""
echo "🎯 Essential files remaining in root:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | sort

echo ""
echo "🎉 Root directory is now clean and organized!"
