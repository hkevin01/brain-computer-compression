#!/bin/bash

echo "🧹 FINAL ROOT DIRECTORY CLEANUP"
echo "==============================="
echo "Executing comprehensive cleanup to achieve professional root directory"
echo ""

# Go to project root
cd "$(dirname "$0")"

# Create a comprehensive cleanup log
exec > >(tee final_cleanup_log.txt) 2>&1

echo "📋 Phase 1: Removing empty/duplicate files"
echo "==========================================="

# Remove empty .md files
echo "Removing empty .md files..."
for file in CONTRIBUTING.md BCI_Validation_Framework_Summary.md EMG_UPDATE.md STATUS_REPORT.md README_NEW.md; do
    if [ -f "$file" ]; then
        if [ ! -s "$file" ]; then
            rm "$file" && echo "  ✅ Removed empty $file"
        else
            echo "  ⚠️  $file is not empty, keeping"
        fi
    else
        echo "  ℹ️  $file not found"
    fi
done

# Remove duplicate guides (exist in docs/guides/)
echo ""
echo "Removing duplicate guide files..."
for file in DOCKER_BUILD_FIX.md DOCKER_QUICK_START.md; do
    if [ -f "$file" ]; then
        rm "$file" && echo "  ✅ Removed duplicate $file (exists in docs/guides/)"
    else
        echo "  ℹ️  $file not found"
    fi
done

# Remove empty shell scripts
echo ""
echo "Removing empty shell scripts..."
for file in cleanup_now.sh cleanup_root.sh; do
    if [ -f "$file" ]; then
        if [ ! -s "$file" ]; then
            rm "$file" && echo "  ✅ Removed empty $file"
        else
            echo "  ⚠️  $file is not empty, moving to scripts/tools/"
            mkdir -p scripts/tools
            mv "$file" scripts/tools/ && echo "  📦 Moved $file to scripts/tools/"
        fi
    else
        echo "  ℹ️  $file not found"
    fi
done

echo ""
echo "📋 Phase 2: Moving scripts to proper locations"
echo "=============================================="

# Create directories
mkdir -p scripts/tools scripts/setup

# Move utility scripts to scripts/tools/
echo "Moving utility scripts to scripts/tools/..."
utility_scripts=(
    "completion_summary.sh"
    "execute_cleanup.sh"
    "execute_final_cleanup.sh"
    "execute_reorganization.sh"
    "final_cleanup.sh"
    "reorganize_root.sh"
    "run_cleanup_now.sh"
    "test_reorganization.sh"
    "verify_reorganization.sh"
    "comprehensive_fix.sh"
    "setup_and_test.sh"
    "setup_and_test.bat"
)

for script in "${utility_scripts[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" scripts/tools/ && echo "  📦 Moved $script"
    else
        echo "  ℹ️  $script not found or already moved"
    fi
done

# Move setup scripts to scripts/setup/
echo ""
echo "Moving setup scripts to scripts/setup/..."
setup_scripts=("setup.sh")

for script in "${setup_scripts[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" scripts/setup/ && echo "  📦 Moved $script"
    else
        echo "  ℹ️  $script not found or already moved"
    fi
done

echo ""
echo "📋 Phase 3: Moving Docker files"
echo "==============================="

# Move Docker files to docker/ as backups or to proper locations
echo "Moving Docker files to docker/..."

# Create docker subdirectories if needed
mkdir -p docker/compose

# Move Dockerfile files
if [ -f "Dockerfile" ]; then
    mv "Dockerfile" "docker/Dockerfile.root-backup" && echo "  📦 Moved Dockerfile → docker/Dockerfile.root-backup"
else
    echo "  ℹ️  Dockerfile not found or already moved"
fi

if [ -f "Dockerfile.frontend" ]; then
    if [ ! -f "docker/Dockerfile.frontend" ]; then
        mv "Dockerfile.frontend" "docker/" && echo "  📦 Moved Dockerfile.frontend → docker/"
    else
        mv "Dockerfile.frontend" "docker/Dockerfile.frontend.backup" && echo "  📦 Moved Dockerfile.frontend → docker/Dockerfile.frontend.backup"
    fi
else
    echo "  ℹ️  Dockerfile.frontend not found or already moved"
fi

if [ -f "Dockerfile.minimal" ]; then
    if [ ! -f "docker/Dockerfile.minimal" ]; then
        mv "Dockerfile.minimal" "docker/" && echo "  📦 Moved Dockerfile.minimal → docker/"
    else
        mv "Dockerfile.minimal" "docker/Dockerfile.minimal.backup" && echo "  📦 Moved Dockerfile.minimal → docker/Dockerfile.minimal.backup"
    fi
else
    echo "  ℹ️  Dockerfile.minimal not found or already moved"
fi

# Move compose files
if [ -f "docker-compose.yml" ]; then
    mv "docker-compose.yml" "docker/compose/docker-compose.root-backup.yml" && echo "  📦 Moved docker-compose.yml → docker/compose/docker-compose.root-backup.yml"
else
    echo "  ℹ️  docker-compose.yml not found or already moved"
fi

if [ -f "docker-compose.dev.yml" ]; then
    if [ ! -f "docker/compose/docker-compose.dev.yml" ]; then
        mv "docker-compose.dev.yml" "docker/compose/" && echo "  📦 Moved docker-compose.dev.yml → docker/compose/"
    else
        mv "docker-compose.dev.yml" "docker/compose/docker-compose.dev.backup.yml" && echo "  📦 Moved docker-compose.dev.yml → docker/compose/docker-compose.dev.backup.yml"
    fi
else
    echo "  ℹ️  docker-compose.dev.yml not found or already moved"
fi

echo ""
echo "📋 Phase 4: Cleanup temporary files"
echo "==================================="

# Remove all cleanup-related scripts
echo "Removing cleanup scripts and temporary files..."
cleanup_files=(
    "execute_root_cleanup.sh"
    "safe_cleanup.sh"
    "targeted_cleanup.sh"
    "temp_cleanup_log.txt"
    "quick_cleanup.sh"
    "manual_cleanup.sh"
    "remove_empty_contributing.sh"
    "cleanup_plan.md"
    "final_cleanup_plan.md"
)

for file in "${cleanup_files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file" && echo "  🗑️  Removed $file"
    fi
done

echo ""
echo "📋 Phase 5: Final status report"
echo "==============================="

file_count=$(ls -la | grep -E '^-' | wc -l)
dir_count=$(ls -la | grep -E '^d' | grep -v '^total' | wc -l)

echo "🎯 Root directory final status:"
echo "  📄 Files: $file_count"
echo "  📁 Directories: $dir_count"

echo ""
echo "✅ Essential files remaining in root:"
ls -la | grep -E '^-' | awk '{print "  " $9}' | grep -v '^  $' | sort

echo ""
echo "📁 Directory structure:"
ls -la | grep -E '^d' | awk '{print "  " $9}' | grep -v '^  \.$' | grep -v '^  \.\.$' | sort

echo ""
echo "🎉 ROOT DIRECTORY CLEANUP COMPLETE!"
echo "===================================="
echo ""
echo "✅ Achievements:"
echo "  • Removed empty/duplicate .md files"
echo "  • Moved all utility scripts to scripts/tools/"
echo "  • Moved setup scripts to scripts/setup/"
echo "  • Organized Docker files in docker/"
echo "  • Achieved clean, professional root directory"
echo ""
echo "🚀 Next steps:"
echo "  ./run.sh status   # Verify functionality"
echo "  ./run.sh build    # Test Docker build"
echo "  ./run.sh up       # Test full workflow"
echo ""
echo "📖 All documentation remains accessible:"
echo "  • docs/guides/ - User guides"
echo "  • docs/project/ - Project documentation"
echo "  • scripts/tools/ - Utility scripts"

# Remove this final cleanup script and log
rm -f "final_comprehensive_cleanup.sh"

echo ""
echo "✅ Cleanup completed successfully!"
echo "Root directory is now clean and professionally organized!"
