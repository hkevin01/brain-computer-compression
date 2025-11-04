#!/bin/bash
# Quick Test Runner
# Run fast unit tests for rapid development iteration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== BCI Compression - Quick Test Suite ===${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install dev dependencies first:${NC}"
    echo "  pip install -e '.[dev]'"
    exit 1
fi

# Run quick tests only (exclude slow tests)
echo -e "${YELLOW}Running quick unit tests (excluding slow tests)...${NC}"
echo ""

# Run with:
# - Mark selection: not slow
# - Verbose output
# - Coverage disabled for speed
# Note: Add --timeout=10 --timeout-method=thread when pytest-timeout is installed
pytest tests/ \
    -m "not slow" \
    -v \
    --tb=short \
    --maxfail=5 \
    "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All quick tests passed!${NC}"
else
    echo ""
    echo -e "${RED}✗ Some tests failed (exit code: $exit_code)${NC}"
fi

exit $exit_code
