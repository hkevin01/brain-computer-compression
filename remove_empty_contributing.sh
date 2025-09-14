#!/bin/bash
echo "Removing empty CONTRIBUTING.md file (proper version exists in docs/)"
if [ -f "CONTRIBUTING.md" ] && [ ! -s "CONTRIBUTING.md" ]; then
    rm "CONTRIBUTING.md"
    echo "✅ Removed empty CONTRIBUTING.md"
else
    echo "ℹ️  CONTRIBUTING.md not found or not empty"
fi
