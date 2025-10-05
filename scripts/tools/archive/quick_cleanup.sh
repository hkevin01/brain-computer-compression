#!/bin/bash
echo "Starting root directory cleanup..."

# Remove empty or duplicate .md files
echo "Removing empty/duplicate .md files..."
rm -f CONTRIBUTING.md  # Empty, exists in docs/
rm -f BCI_Validation_Framework_Summary.md  # Empty
rm -f EMG_UPDATE.md  # Empty
rm -f STATUS_REPORT.md  # Empty
rm -f README_NEW.md  # Duplicate/obsolete
rm -f DOCKER_BUILD_FIX.md  # Duplicate, exists in docs/guides/
rm -f DOCKER_QUICK_START.md  # Duplicate, exists in docs/guides/

# Remove empty shell scripts
echo "Removing empty shell scripts..."
rm -f cleanup_now.sh  # Empty
rm -f cleanup_root.sh  # Empty

# Move important scripts to scripts/tools/
echo "Moving scripts to scripts/tools/..."
mv reorganization_complete.sh scripts/tools/ 2>/dev/null || echo "reorganization_complete.sh already moved or doesn't exist"
mv completion_summary.sh scripts/tools/ 2>/dev/null || echo "completion_summary.sh already moved or doesn't exist"
mv execute_cleanup.sh scripts/tools/ 2>/dev/null || echo "execute_cleanup.sh already moved or doesn't exist"
mv execute_final_cleanup.sh scripts/tools/ 2>/dev/null || echo "execute_final_cleanup.sh already moved or doesn't exist"
mv execute_reorganization.sh scripts/tools/ 2>/dev/null || echo "execute_reorganization.sh already moved or doesn't exist"
mv final_cleanup.sh scripts/tools/ 2>/dev/null || echo "final_cleanup.sh already moved or doesn't exist"
mv reorganize_root.sh scripts/tools/ 2>/dev/null || echo "reorganize_root.sh already moved or doesn't exist"
mv run_cleanup_now.sh scripts/tools/ 2>/dev/null || echo "run_cleanup_now.sh already moved or doesn't exist"
mv test_reorganization.sh scripts/tools/ 2>/dev/null || echo "test_reorganization.sh already moved or doesn't exist"
mv verify_reorganization.sh scripts/tools/ 2>/dev/null || echo "verify_reorganization.sh already moved or doesn't exist"

# Move setup scripts
echo "Moving setup scripts..."
mv setup.sh scripts/setup/ 2>/dev/null || echo "setup.sh already moved or doesn't exist"
mv setup_and_test.sh scripts/tools/ 2>/dev/null || echo "setup_and_test.sh already moved or doesn't exist"
mv setup_and_test.bat scripts/tools/ 2>/dev/null || echo "setup_and_test.bat already moved or doesn't exist"

# Move Docker files (as backups since docker/ already organized)
echo "Moving Docker files to docker/ as backups..."
mv Dockerfile docker/Dockerfile.root-backup 2>/dev/null || echo "Dockerfile already moved or doesn't exist"
mv Dockerfile.frontend docker/Dockerfile.frontend.backup2 2>/dev/null || echo "Dockerfile.frontend already moved or doesn't exist"
mv Dockerfile.minimal docker/Dockerfile.minimal.backup2 2>/dev/null || echo "Dockerfile.minimal already moved or doesn't exist"
mv docker-compose.yml docker/compose/docker-compose.root-backup.yml 2>/dev/null || echo "docker-compose.yml already moved or doesn't exist"
mv docker-compose.dev.yml docker/compose/docker-compose.dev-backup2.yml 2>/dev/null || echo "docker-compose.dev.yml already moved or doesn't exist"

# Move comprehensive_fix.sh if it exists
mv comprehensive_fix.sh scripts/tools/ 2>/dev/null || echo "comprehensive_fix.sh already moved or doesn't exist"

# Clean up the cleanup scripts themselves
echo "Removing cleanup scripts..."
rm -f execute_root_cleanup.sh
rm -f safe_cleanup.sh
rm -f targeted_cleanup.sh
rm -f temp_cleanup_log.txt
rm -f quick_cleanup.sh

echo "✅ Root directory cleanup completed!"

# Show final status
echo ""
echo "📊 Root directory now contains:"
ls -la | grep -E '^-' | wc -l | xargs echo "Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "Directories:"

echo ""
echo "✅ Essential files remaining:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | head -15
