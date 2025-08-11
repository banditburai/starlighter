"""
Starlighter benchmarking framework (development tools).

Provides tools for performance measurement, regression detection,
and A/B testing of code changes. This is a development tool and
not included in the distributed package.
"""

from .framework import Benchmark, BenchmarkResult, PerformanceTracker
from .metrics import (
    calculate_percentiles,
    detect_outliers,
    analyze_scaling,
    calculate_speedup,
    generate_report,
    MemoryProfiler,
    PerformanceProfile,
)
from .runner import BenchmarkSuite

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "PerformanceTracker",
    "BenchmarkSuite",
    "calculate_percentiles",
    "detect_outliers",
    "analyze_scaling",
    "calculate_speedup",
    "generate_report",
    "MemoryProfiler",
    "PerformanceProfile",
]
