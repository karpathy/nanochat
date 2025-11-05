#!/bin/bash
#
# Make all test scripts executable
#

echo "Making test scripts executable..."

chmod +x tests/run_unit_tests.sh
chmod +x tests/run_integration_tests.sh
chmod +x tests/integration/*.sh

echo "✓ Done!"
echo ""
echo "You can now run:"
echo "  bash tests/run_unit_tests.sh"
echo "  bash tests/run_integration_tests.sh"
