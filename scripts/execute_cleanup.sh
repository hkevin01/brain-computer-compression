#!/bin/bash

# Execute the reorganization script
echo "🚀 Running root directory reorganization..."
echo ""

if [ -f "reorganize_root.sh" ]; then
    chmod +x reorganize_root.sh
    ./reorganize_root.sh
    echo ""
    echo "✅ Reorganization complete!"
else
    echo "❌ reorganize_root.sh not found"
    exit 1
fi

echo ""
echo "🧪 Testing reorganization results..."
if [ -f "test_reorganization.sh" ]; then
    chmod +x test_reorganization.sh
    ./test_reorganization.sh
else
    echo "⚠️ test_reorganization.sh not found, skipping tests"
fi

echo ""
echo "🎉 All done! Root directory has been reorganized."
