#!/bin/bash

# Execute the reorganization step by step
echo "🚀 Executing Root Directory Reorganization"
echo "=========================================="

# Step 1: Move documentation files
echo "📝 Moving documentation files..."

# Move main docs to docs/
[ -f "CHANGELOG.md" ] && mv "CHANGELOG.md" "docs/" && echo "  ✅ CHANGELOG.md → docs/"
[ -f "CONTRIBUTING.md" ] && mv "CONTRIBUTING.md" "docs/" && echo "  ✅ CONTRIBUTING.md → docs/"

# Move guides to docs/guides/
[ -f "DOCKER_BUILD_FIX.md" ] && mv "DOCKER_BUILD_FIX.md" "docs/guides/" && echo "  ✅ DOCKER_BUILD_FIX.md → docs/guides/"
[ -f "DOCKER_QUICK_START.md" ] && mv "DOCKER_QUICK_START.md" "docs/guides/" && echo "  ✅ DOCKER_QUICK_START.md → docs/guides/"

# Move project docs to docs/project/
[ -f "BCI_Validation_Framework_Summary.md" ] && mv "BCI_Validation_Framework_Summary.md" "docs/project/" && echo "  ✅ BCI_Validation_Framework_Summary.md → docs/project/"
[ -f "EMG_UPDATE.md" ] && mv "EMG_UPDATE.md" "docs/project/" && echo "  ✅ EMG_UPDATE.md → docs/project/"
[ -f "STATUS_REPORT.md" ] && mv "STATUS_REPORT.md" "docs/project/" && echo "  ✅ STATUS_REPORT.md → docs/project/"
[ -f "README_NEW.md" ] && mv "README_NEW.md" "docs/project/" && echo "  ✅ README_NEW.md → docs/project/"

echo "✅ Documentation files moved"

# Step 2: Move Docker files
echo ""
echo "🐳 Moving Docker files..."

[ -f "Dockerfile" ] && mv "Dockerfile" "docker/" && echo "  ✅ Dockerfile → docker/"
[ -f "Dockerfile.frontend" ] && mv "Dockerfile.frontend" "docker/" && echo "  ✅ Dockerfile.frontend → docker/"
[ -f "Dockerfile.minimal" ] && mv "Dockerfile.minimal" "docker/" && echo "  ✅ Dockerfile.minimal → docker/"
[ -f ".dockerignore" ] && mv ".dockerignore" "docker/" && echo "  ✅ .dockerignore → docker/"

[ -f "docker-compose.yml" ] && mv "docker-compose.yml" "docker/compose/" && echo "  ✅ docker-compose.yml → docker/compose/"
[ -f "docker-compose.dev.yml" ] && mv "docker-compose.dev.yml" "docker/compose/" && echo "  ✅ docker-compose.dev.yml → docker/compose/"
[ -f "docker-compose.dev.yml.backup" ] && mv "docker-compose.dev.yml.backup" "docker/compose/" && echo "  ✅ docker-compose.dev.yml.backup → docker/compose/"

echo "✅ Docker files moved"

# Step 3: Move scripts
echo ""
echo "🔧 Moving scripts..."

[ -f "setup.sh" ] && mv "setup.sh" "scripts/setup/" && echo "  ✅ setup.sh → scripts/setup/"
[ -f "setup_and_test.sh" ] && mv "setup_and_test.sh" "scripts/setup/" && echo "  ✅ setup_and_test.sh → scripts/setup/"
[ -f "setup_and_test.bat" ] && mv "setup_and_test.bat" "scripts/setup/" && echo "  ✅ setup_and_test.bat → scripts/setup/"

[ -f "cleanup_now.sh" ] && mv "cleanup_now.sh" "scripts/tools/" && echo "  ✅ cleanup_now.sh → scripts/tools/"
[ -f "cleanup_root.sh" ] && mv "cleanup_root.sh" "scripts/tools/" && echo "  ✅ cleanup_root.sh → scripts/tools/"
[ -f "comprehensive_fix.sh" ] && mv "comprehensive_fix.sh" "scripts/tools/" && echo "  ✅ comprehensive_fix.sh → scripts/tools/"
[ -f "fix_docker_build.sh" ] && mv "fix_docker_build.sh" "scripts/tools/" && echo "  ✅ fix_docker_build.sh → scripts/tools/"
[ -f "test_docker_build.sh" ] && mv "test_docker_build.sh" "scripts/tools/" && echo "  ✅ test_docker_build.sh → scripts/tools/"
[ -f "test_docker_workflow.sh" ] && mv "test_docker_workflow.sh" "scripts/tools/" && echo "  ✅ test_docker_workflow.sh → scripts/tools/"
[ -f "reorganize_root.sh" ] && mv "reorganize_root.sh" "scripts/tools/" && echo "  ✅ reorganize_root.sh → scripts/tools/"
[ -f "test_reorganization.sh" ] && mv "test_reorganization.sh" "scripts/tools/" && echo "  ✅ test_reorganization.sh → scripts/tools/"

echo "✅ Scripts moved"

# Step 4: Move other files
echo ""
echo "📄 Moving other files..."

[ -f "benchmark_results.json" ] && mkdir -p ".benchmarks" && mv "benchmark_results.json" ".benchmarks/" && echo "  ✅ benchmark_results.json → .benchmarks/"
[ -f "test_output.log" ] && mkdir -p "logs" && mv "test_output.log" "logs/" && echo "  ✅ test_output.log → logs/"
[ -f "package.json" ] && [ -f "dashboard/package.json" ] && mv "package.json" "docs/project/" && echo "  ✅ package.json → docs/project/ (root copy removed)"

echo "✅ Other files moved"

echo ""
echo "📊 Current root directory:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"

echo ""
echo "🎉 File movement complete!"
echo "📁 New structure created successfully"
