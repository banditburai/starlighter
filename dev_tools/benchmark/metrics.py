"""
Performance metrics and analysis utilities for benchmarking.

Provides statistical analysis, visualization data, and performance
characterization for benchmark results.
"""

import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class PerformanceProfile:
    """Performance characteristics of code."""

    complexity_class: str  # O(1), O(n), O(n log n), etc.
    base_time: float  # Time for minimal input
    scaling_factor: float  # How time scales with input
    variance: float  # Consistency of performance
    outlier_ratio: float  # Ratio of outliers to normal samples


def calculate_percentiles(samples: List[float]) -> Dict[str, float]:
    """
    Calculate various percentiles from samples.

    Args:
        samples: List of timing samples in milliseconds

    Returns:
        Dictionary with percentile values
    """
    sorted_samples = sorted(samples)
    n = len(samples)

    return {
        "p1": sorted_samples[max(0, n * 1 // 100)],
        "p5": sorted_samples[n * 5 // 100],
        "p10": sorted_samples[n * 10 // 100],
        "p25": sorted_samples[n * 25 // 100],
        "p50": sorted_samples[n * 50 // 100],  # median
        "p75": sorted_samples[n * 75 // 100],
        "p90": sorted_samples[n * 90 // 100],
        "p95": sorted_samples[n * 95 // 100],
        "p99": sorted_samples[min(n * 99 // 100, n - 1)],
        "p999": sorted_samples[min(n * 999 // 1000, n - 1)]
        if n >= 1000
        else sorted_samples[-1],
    }


def detect_outliers(samples: List[float]) -> Tuple[List[float], List[float]]:
    """
    Detect outliers using IQR method.

    Args:
        samples: List of timing samples

    Returns:
        Tuple of (normal_samples, outliers)
    """
    sorted_samples = sorted(samples)
    n = len(samples)

    q1 = sorted_samples[n // 4]
    q3 = sorted_samples[3 * n // 4]
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    normal = []
    outliers = []

    for s in samples:
        if lower_bound <= s <= upper_bound:
            normal.append(s)
        else:
            outliers.append(s)

    return normal, outliers


def analyze_scaling(timings_by_size: Dict[int, List[float]]) -> PerformanceProfile:
    """
    Analyze how performance scales with input size.

    Args:
        timings_by_size: Dictionary mapping input size to timing samples

    Returns:
        PerformanceProfile with complexity analysis
    """
    if not timings_by_size:
        return PerformanceProfile("unknown", 0, 0, 0, 0)

    # Get median time for each size
    sizes = sorted(timings_by_size.keys())
    medians = [statistics.median(timings_by_size[size]) for size in sizes]

    # Detect complexity class by fitting curves
    complexity_class = detect_complexity(sizes, medians)

    # Calculate base time (smallest input)
    base_time = medians[0] if medians else 0

    # Calculate scaling factor
    if len(sizes) > 1:
        scaling_factor = (medians[-1] - medians[0]) / (sizes[-1] - sizes[0])
    else:
        scaling_factor = 0

    # Calculate variance across all samples
    all_samples = []
    for samples in timings_by_size.values():
        all_samples.extend(samples)

    variance = (
        statistics.stdev(all_samples) / statistics.mean(all_samples)
        if all_samples
        else 0
    )

    # Calculate outlier ratio
    normal, outliers = detect_outliers(all_samples)
    outlier_ratio = len(outliers) / len(all_samples) if all_samples else 0

    return PerformanceProfile(
        complexity_class=complexity_class,
        base_time=base_time,
        scaling_factor=scaling_factor,
        variance=variance,
        outlier_ratio=outlier_ratio,
    )


def detect_complexity(sizes: List[int], times: List[float]) -> str:
    """
    Detect algorithmic complexity from size/time data.

    Uses curve fitting to identify O(1), O(log n), O(n), O(n log n), O(n²).

    Args:
        sizes: Input sizes
        times: Corresponding median times

    Returns:
        String representation of complexity class
    """
    if len(sizes) < 3:
        return "insufficient data"

    # Normalize data
    max_size = max(sizes)
    max_time = max(times)
    norm_sizes = [s / max_size for s in sizes]
    norm_times = [t / max_time for t in times]

    # Calculate correlation with different complexity curves
    correlations = {}

    # O(1) - constant
    const_fit = sum((t - norm_times[0]) ** 2 for t in norm_times)
    correlations["O(1)"] = 1 / (1 + const_fit)

    # O(log n)
    log_curve = [
        math.log(s + 1) / math.log(max_size + 1) if s > 0 else 0 for s in sizes
    ]
    log_fit = sum((norm_times[i] - log_curve[i]) ** 2 for i in range(len(sizes)))
    correlations["O(log n)"] = 1 / (1 + log_fit)

    # O(n) - linear
    linear_fit = sum((norm_times[i] - norm_sizes[i]) ** 2 for i in range(len(sizes)))
    correlations["O(n)"] = 1 / (1 + linear_fit)

    # O(n log n)
    nlogn_curve = [
        s * math.log(s + 1) / (max_size * math.log(max_size + 1)) if s > 0 else 0
        for s in sizes
    ]
    nlogn_fit = sum((norm_times[i] - nlogn_curve[i]) ** 2 for i in range(len(sizes)))
    correlations["O(n log n)"] = 1 / (1 + nlogn_fit)

    # O(n²) - quadratic
    quad_curve = [(s / max_size) ** 2 for s in sizes]
    quad_fit = sum((norm_times[i] - quad_curve[i]) ** 2 for i in range(len(sizes)))
    correlations["O(n²)"] = 1 / (1 + quad_fit)

    # Return best fit
    return max(correlations, key=correlations.get)


def calculate_speedup(baseline: List[float], current: List[float]) -> Dict[str, float]:
    """
    Calculate speedup metrics between baseline and current.

    Args:
        baseline: Baseline timing samples
        current: Current timing samples

    Returns:
        Dictionary with speedup metrics
    """
    baseline_mean = statistics.mean(baseline)
    current_mean = statistics.mean(current)

    baseline_p50 = statistics.median(baseline)
    current_p50 = statistics.median(current)

    baseline_p99 = sorted(baseline)[len(baseline) * 99 // 100]
    current_p99 = sorted(current)[len(current) * 99 // 100]

    return {
        "mean_speedup": baseline_mean / current_mean,
        "median_speedup": baseline_p50 / current_p50,
        "p99_speedup": baseline_p99 / current_p99,
        "percent_faster": (baseline_mean - current_mean) / baseline_mean * 100,
        "absolute_improvement": baseline_mean - current_mean,
    }


def generate_report(results: Dict[str, Any], format: str = "text") -> str:
    """
    Generate a performance report from benchmark results.

    Args:
        results: Benchmark results dictionary
        format: Output format ('text', 'markdown', 'json')

    Returns:
        Formatted report string
    """
    if format == "text":
        return _text_report(results)
    elif format == "markdown":
        return _markdown_report(results)
    else:
        import json

        return json.dumps(results, indent=2)


def _text_report(results: Dict[str, Any]) -> str:
    """Generate plain text report."""
    lines = []
    lines.append("Performance Report")
    lines.append("=" * 50)

    if "name" in results:
        lines.append(f"Benchmark: {results['name']}")

    if "mean" in results:
        lines.append(f"Mean:      {results['mean']:.3f}ms")
    if "median" in results:
        lines.append(f"Median:    {results['median']:.3f}ms")
    if "p95" in results:
        lines.append(f"P95:       {results['p95']:.3f}ms")
    if "p99" in results:
        lines.append(f"P99:       {results['p99']:.3f}ms")

    if "speedup" in results:
        lines.append(f"\nSpeedup:   {results['speedup']:.2f}x")

    if "profile" in results:
        p = results["profile"]
        lines.append(f"\nComplexity: {p.get('complexity_class', 'unknown')}")
        lines.append(f"Variance:   {p.get('variance', 0):.2%}")

    return "\n".join(lines)


def _markdown_report(results: Dict[str, Any]) -> str:
    """Generate Markdown report."""
    lines = []
    lines.append("# Performance Report")

    if "name" in results:
        lines.append(f"## {results['name']}")

    lines.append("\n### Timing Statistics")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    if "mean" in results:
        lines.append(f"| Mean | {results['mean']:.3f}ms |")
    if "median" in results:
        lines.append(f"| Median | {results['median']:.3f}ms |")
    if "p95" in results:
        lines.append(f"| P95 | {results['p95']:.3f}ms |")
    if "p99" in results:
        lines.append(f"| P99 | {results['p99']:.3f}ms |")

    if "speedup" in results:
        lines.append(f"\n**Speedup:** {results['speedup']:.2f}x")

    return "\n".join(lines)


class MemoryProfiler:
    """
    Simple memory profiling for benchmarks.

    Tracks memory usage during benchmark execution.
    """

    def __init__(self):
        """Initialize memory profiler."""
        self.baseline = 0
        self.peak = 0
        self.current = 0

    def start(self):
        """Start memory profiling."""
        import tracemalloc

        tracemalloc.start()
        self.baseline = tracemalloc.get_traced_memory()[0]

    def sample(self):
        """Sample current memory usage."""
        import tracemalloc

        current, peak = tracemalloc.get_traced_memory()
        self.current = current
        self.peak = max(self.peak, peak)

    def stop(self) -> Dict[str, float]:
        """
        Stop profiling and return results.

        Returns:
            Dictionary with memory statistics in MB
        """
        import tracemalloc

        self.sample()
        tracemalloc.stop()

        return {
            "baseline_mb": self.baseline / 1024 / 1024,
            "peak_mb": self.peak / 1024 / 1024,
            "current_mb": self.current / 1024 / 1024,
            "allocated_mb": (self.peak - self.baseline) / 1024 / 1024,
        }
