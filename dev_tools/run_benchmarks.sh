#!/bin/bash
# Starlighter Performance Benchmarking Script
# Run this script to check for performance regressions

set -e

echo "🚀 Running Starlighter Performance Benchmarks"
echo "============================================="

# Check if baseline exists
if [ ! -f "dev_tools/baseline.json" ]; then
    echo "⚠️  No baseline found. Creating baseline..."
    uv run python -m dev_tools.benchmark.runner --save-baseline
    echo "✅ Baseline created successfully"
    exit 0
fi

# Run benchmarks with regression check
echo "📊 Running benchmarks and checking for regressions..."
uv run python -m dev_tools.benchmark.runner --check-regression --output results.json

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All benchmarks passed - no regressions detected!"
else
    echo "❌ Performance regression detected!"
    echo "Review results.json for details"
    exit 1
fi

# Optional: Update baseline if requested
if [ "$1" == "--update-baseline" ]; then
    echo "📝 Updating baseline..."
    uv run python -m dev_tools.benchmark.runner --save-baseline
    echo "✅ Baseline updated"
fi