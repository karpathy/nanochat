#!/bin/bash
#
# Run all unit tests for auto-discovery functionality
#

echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="
echo ""

# Run pytest with verbose output
pytest tests/test_auto_batch_size.py -v --tb=short

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All unit tests passed!"
else
    echo "✗ Some unit tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
