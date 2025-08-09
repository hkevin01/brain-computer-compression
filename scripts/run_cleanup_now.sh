#!/bin/bash

# Simple execution wrapper
cd /home/kevin/Projects/brain-computer-compression

# Make scripts executable
chmod +x reorganize_root.sh 2>/dev/null || true
chmod +x test_reorganization.sh 2>/dev/null || true
chmod +x execute_cleanup.sh 2>/dev/null || true

# Check current directory state
echo "📊 Current root directory contents:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"

echo ""
echo "🚀 Executing reorganization now..."
echo ""

# Run the reorganization
if [ -f "reorganize_root.sh" ]; then
    echo "🔧 Running reorganize_root.sh..."
    bash reorganize_root.sh
    echo ""
    echo "✅ Reorganization script completed!"
else
    echo "⚠️  reorganize_root.sh not found"
fi

echo ""
echo "📊 Final root directory contents:"
ls -la | grep -E '^-' | wc -l | xargs echo "  Files:"
ls -la | grep -E '^d' | grep -v '^total' | wc -l | xargs echo "  Directories:"

echo ""
echo "✅ Cleanup execution complete!"
