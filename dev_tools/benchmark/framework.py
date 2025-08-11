"""
Core benchmarking framework for Starlighter.

Provides consistent, statistically rigorous performance measurement
with A/B testing capabilities and regression detection.
"""

import time
import statistics
import json
import gc
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    samples: List[float]
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    timestamp: str
    metadata: Dict[str, Any]

    @classmethod
    def from_samples(cls, name: str, samples: List[float], metadata: Dict = None):
        """Create BenchmarkResult from raw samples."""
        sorted_samples = sorted(samples)
        n = len(samples)

        return cls(
            name=name,
            samples=samples,
            mean=statistics.mean(samples),
            median=statistics.median(samples),
            stdev=statistics.stdev(samples) if n > 1 else 0,
            min=min(samples),
            max=max(samples),
            p50=sorted_samples[n * 50 // 100],
            p75=sorted_samples[n * 75 // 100],
            p90=sorted_samples[n * 90 // 100],
            p95=sorted_samples[n * 95 // 100],
            p99=sorted_samples[n * 99 // 100] if n >= 100 else sorted_samples[-1],
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Benchmark:
    """
    Core benchmark runner with statistical analysis.

    Features:
    - Warmup runs
    - Statistical sampling
    - Garbage collection control
    - Memory measurement
    - JSON output for tracking
    """

    def __init__(
        self, warmup_runs: int = 5, sample_runs: int = 100, gc_enabled: bool = False
    ):
        """
        Initialize benchmark runner.

        Args:
            warmup_runs: Number of warmup iterations before measurement
            sample_runs: Number of samples to collect
            gc_enabled: Whether to enable GC during benchmarks
        """
        self.warmup_runs = warmup_runs
        self.sample_runs = sample_runs
        self.gc_enabled = gc_enabled
        self.results: List[BenchmarkResult] = []

    def run(self, func: Callable, *args, name: str = None, **kwargs) -> BenchmarkResult:
        """
        Run a benchmark on a function.

        Args:
            func: Function to benchmark
            *args: Positional arguments for func
            name: Name for this benchmark
            **kwargs: Keyword arguments for func

        Returns:
            BenchmarkResult with timing statistics
        """
        name = name or func.__name__

        # Disable GC if requested
        gc_was_enabled = gc.isenabled()
        if not self.gc_enabled:
            gc.disable()

        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                func(*args, **kwargs)

            # Collect samples
            samples = []
            for _ in range(self.sample_runs):
                # Force GC before measurement if GC is disabled
                if not self.gc_enabled:
                    gc.collect()

                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()

                samples.append((end - start) * 1000)  # Convert to ms

            # Create result
            result = BenchmarkResult.from_samples(
                name=name,
                samples=samples,
                metadata={
                    "warmup_runs": self.warmup_runs,
                    "sample_runs": self.sample_runs,
                    "gc_enabled": self.gc_enabled,
                },
            )

            self.results.append(result)
            return result

        finally:
            # Restore GC state
            if gc_was_enabled:
                gc.enable()
            else:
                gc.disable()

    def compare(
        self,
        func_a: Callable,
        func_b: Callable,
        *args,
        name_a: str = "A",
        name_b: str = "B",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        A/B comparison of two functions.

        Args:
            func_a: First function to compare
            func_b: Second function to compare
            *args: Common arguments for both functions
            name_a: Name for first function
            name_b: Name for second function
            **kwargs: Common keyword arguments

        Returns:
            Dictionary with comparison results
        """
        # Run both benchmarks
        result_a = self.run(func_a, *args, name=name_a, **kwargs)
        result_b = self.run(func_b, *args, name=name_b, **kwargs)

        # Calculate speedup/slowdown
        speedup_mean = result_b.mean / result_a.mean
        speedup_p99 = result_b.p99 / result_a.p99

        # Statistical significance test (simplified t-test)
        pooled_stdev = ((result_a.stdev**2 + result_b.stdev**2) / 2) ** 0.5
        effect_size = (
            abs(result_a.mean - result_b.mean) / pooled_stdev if pooled_stdev > 0 else 0
        )

        return {
            "result_a": result_a,
            "result_b": result_b,
            "speedup_mean": speedup_mean,
            "speedup_p99": speedup_p99,
            "effect_size": effect_size,
            "significant": effect_size > 0.5,  # Medium effect size threshold
            "winner": name_a if speedup_mean > 1 else name_b,
            "summary": f"{name_a if speedup_mean > 1 else name_b} is {abs(speedup_mean - 1) * 100:.1f}% faster",
        }

    def save_results(self, filepath: Path):
        """Save benchmark results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "warmup_runs": self.warmup_runs,
                "sample_runs": self.sample_runs,
                "gc_enabled": self.gc_enabled,
            },
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_baseline(self, filepath: Path) -> List[BenchmarkResult]:
        """Load baseline results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        baseline = []
        for r in data["results"]:
            baseline.append(BenchmarkResult(**r))

        return baseline

    def check_regression(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        threshold: float = 1.1,
    ) -> bool:
        """
        Check if current result is a regression from baseline.

        Args:
            current: Current benchmark result
            baseline: Baseline benchmark result
            threshold: Regression threshold (1.1 = 10% slower is regression)

        Returns:
            True if regression detected
        """
        # Check P99 regression
        if current.p99 > baseline.p99 * threshold:
            return True

        # Check mean regression
        if current.mean > baseline.mean * threshold:
            return True

        return False

    def summary(self) -> str:
        """Generate summary of all benchmark results."""
        if not self.results:
            return "No benchmark results"

        lines = ["Benchmark Summary", "=" * 50]

        for result in self.results:
            lines.append(f"\n{result.name}:")
            lines.append(f"  Mean: {result.mean:.3f}ms")
            lines.append(f"  P50:  {result.p50:.3f}ms")
            lines.append(f"  P90:  {result.p90:.3f}ms")
            lines.append(f"  P95:  {result.p95:.3f}ms")
            lines.append(f"  P99:  {result.p99:.3f}ms")
            lines.append(f"  Min:  {result.min:.3f}ms")
            lines.append(f"  Max:  {result.max:.3f}ms")

        return "\n".join(lines)


class PerformanceTracker:
    """
    Track performance over time and detect regressions.

    Maintains historical performance data and provides
    regression detection and trend analysis.
    """

    def __init__(self, history_file: Path = None):
        """
        Initialize performance tracker.

        Args:
            history_file: Path to JSON file for storing history
        """
        self.history_file = history_file or Path("benchmark_history.json")
        self.history: List[Dict] = []

        if self.history_file.exists():
            self.load_history()

    def load_history(self):
        """Load benchmark history from file."""
        with open(self.history_file, "r") as f:
            self.history = json.load(f)

    def save_history(self):
        """Save benchmark history to file."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to history."""
        self.history.append(result.to_dict())
        self.save_history()

    def get_trend(
        self, name: str, metric: str = "p99", last_n: int = 10
    ) -> List[float]:
        """
        Get performance trend for a specific benchmark.

        Args:
            name: Benchmark name
            metric: Metric to track (mean, p99, etc.)
            last_n: Number of recent results to include

        Returns:
            List of metric values over time
        """
        values = []
        for entry in self.history:
            if entry["name"] == name:
                values.append(entry[metric])

        return values[-last_n:] if len(values) > last_n else values

    def detect_regression(
        self, result: BenchmarkResult, window: int = 5, threshold: float = 1.1
    ) -> bool:
        """
        Detect if result is a regression from recent history.

        Args:
            result: Current benchmark result
            window: Number of recent results to compare against
            threshold: Regression threshold

        Returns:
            True if regression detected
        """
        trend = self.get_trend(result.name, "p99", window)

        if not trend:
            return False

        # Compare against average of recent results
        avg_baseline = statistics.mean(trend)

        return result.p99 > avg_baseline * threshold
