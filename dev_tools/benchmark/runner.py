"""
Benchmark runner for Starlighter performance testing.

Provides standardized benchmark suites and test cases for
consistent performance measurement.
"""

from pathlib import Path
from typing import Dict, List, Callable, Any
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from starlighter import highlight
from dev_tools.benchmark.framework import Benchmark, BenchmarkResult
from dev_tools.benchmark.metrics import MemoryProfiler


class BenchmarkSuite:
    """
    Standard benchmark suite for Starlighter.

    Includes test cases for different code sizes and complexity.
    """

    # Standard test sizes
    SIZES = {
        "tiny": 10,  # 10 lines
        "small": 50,  # 50 lines
        "medium": 250,  # 250 lines
        "large": 500,  # 500 lines
        "xlarge": 1000,  # 1000 lines
        "huge": 5000,  # 5000 lines
    }

    @staticmethod
    def generate_simple_code(lines: int) -> str:
        """Generate simple Python code with given number of lines."""
        code_lines = []
        for i in range(lines):
            if i % 10 == 0:
                code_lines.append(f"# Section {i // 10}")
            elif i % 5 == 0:
                code_lines.append(f"def func_{i}():")
                code_lines.append(f"    return {i} * 2")
            else:
                code_lines.append(f"x_{i} = {i} + 1")
        return "\n".join(code_lines)

    @staticmethod
    def generate_complex_code(lines: int) -> str:
        """Generate complex Python code with various constructs."""
        code_lines = []

        for i in range(lines):
            pattern = i % 20

            if pattern == 0:
                code_lines.append(f"@decorator_{i}")
            elif pattern == 1:
                code_lines.append(f"class Class_{i}:")
            elif pattern == 2:
                code_lines.append(f'    """Docstring for class {i}"""')
            elif pattern == 3:
                code_lines.append(f"    def __init__(self, param_{i}: int):")
            elif pattern == 4:
                code_lines.append(f"        self.value = param_{i}")
            elif pattern == 5:
                code_lines.append("        self.data = []")
            elif pattern == 6:
                code_lines.append(f"    def method_{i}(self) -> str:")
            elif pattern == 7:
                code_lines.append(f'        return f"Result: {{self.value * {i}}}"')
            elif pattern == 8:
                code_lines.append(f"# Comment about section {i}")
            elif pattern == 9:
                code_lines.append(f"data_show = 'visible_{i}'  # DataStar attribute")
            elif pattern == 10:
                code_lines.append(f"Button('Click {i}')  # StarHTML element")
            elif pattern == 11:
                code_lines.append(f"value = {i} * 3.14159")
            elif pattern == 12:
                code_lines.append(f"if value > {i * 10}:")
            elif pattern == 13:
                code_lines.append("    print('Large value:', value)")
            elif pattern == 14:
                code_lines.append("else:")
            elif pattern == 15:
                code_lines.append("    print('Small value')")
            elif pattern == 16:
                code_lines.append(f"items = [x for x in range({i})]")
            elif pattern == 17:
                code_lines.append("result = sum(items) / len(items) if items else 0")
            elif pattern == 18:
                code_lines.append(f"text = 'String with {{}} placeholder'.format({i})")
            else:
                code_lines.append(f"var_{i} = lambda x: x + {i}")

        return "\n".join(code_lines)

    @staticmethod
    def generate_edge_case_code() -> Dict[str, str]:
        """Generate edge case test codes."""
        return {
            "empty": "",
            "single_line": "x = 1",
            "only_comments": "\n".join(["# Comment " + str(i) for i in range(100)]),
            "only_strings": "\n".join([f'text_{i} = "String {i}"' for i in range(100)]),
            "nested_strings": 'text = "She said \\"Hello\\" to me"',
            "unicode": 'message = "Hello 世界 🌍"',
            "long_line": "x = " + " + ".join([str(i) for i in range(500)]),
            "deeply_nested": "\n".join(
                ["    " * min(i, 20) + f"if x_{i}:" for i in range(50)]
            ),
            "unterminated_string": 'text = "This string never ends...',
            "mixed_quotes": '''text1 = "double"\ntext2 = 'single'\ntext3 = """triple"""''',
        }

    def __init__(self, warmup: int = 5, samples: int = 100):
        """
        Initialize benchmark suite.

        Args:
            warmup: Number of warmup runs
            samples: Number of samples to collect
        """
        self.benchmark = Benchmark(warmup_runs=warmup, sample_runs=samples)
        self.results: Dict[str, BenchmarkResult] = {}

    def run_size_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for different code sizes."""
        results = {}

        for size_name, line_count in self.SIZES.items():
            if line_count > 1000:
                # Skip huge sizes for quick tests
                continue

            code = self.generate_simple_code(line_count)

            result = self.benchmark.run(
                highlight, code, name=f"simple_{size_name}_{line_count}_lines"
            )

            results[f"simple_{size_name}"] = result

            # Also test complex code
            complex_code = self.generate_complex_code(line_count)

            complex_result = self.benchmark.run(
                highlight, complex_code, name=f"complex_{size_name}_{line_count}_lines"
            )

            results[f"complex_{size_name}"] = complex_result

        self.results.update(results)
        return results

    def run_edge_case_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for edge cases."""
        results = {}
        edge_cases = self.generate_edge_case_code()

        for case_name, code in edge_cases.items():
            result = self.benchmark.run(highlight, code, name=f"edge_{case_name}")
            results[case_name] = result

        self.results.update(results)
        return results

    def run_memory_benchmarks(self) -> Dict[str, Dict]:
        """Run memory profiling benchmarks."""
        results = {}
        profiler = MemoryProfiler()

        for size_name, line_count in self.SIZES.items():
            if line_count > 1000:
                continue

            code = self.generate_complex_code(line_count)

            profiler.start()

            # Run highlight multiple times to see memory growth
            for _ in range(10):
                highlight(code)
                profiler.sample()

            memory_stats = profiler.stop()
            results[size_name] = memory_stats

        return results

    def run_all(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Running size benchmarks...")
        size_results = self.run_size_benchmarks()

        print("Running edge case benchmarks...")
        edge_results = self.run_edge_case_benchmarks()

        print("Running memory benchmarks...")
        memory_results = self.run_memory_benchmarks()

        return {
            "size_benchmarks": {k: v.to_dict() for k, v in size_results.items()},
            "edge_benchmarks": {k: v.to_dict() for k, v in edge_results.items()},
            "memory_benchmarks": memory_results,
            "summary": self.benchmark.summary(),
        }

    def compare_versions(
        self, version_a: Callable, version_b: Callable, test_size: str = "medium"
    ) -> Dict:
        """
        Compare two versions of the highlight function.

        Args:
            version_a: First version to test
            version_b: Second version to test
            test_size: Size of test code to use

        Returns:
            Comparison results
        """
        line_count = self.SIZES[test_size]
        code = self.generate_complex_code(line_count)

        return self.benchmark.compare(
            version_a, version_b, code, name_a="Version A", name_b="Version B"
        )

    def save_baseline(self, filepath: Path = None):
        """Save current results as baseline."""
        filepath = filepath or Path("baseline.json")
        self.benchmark.save_results(filepath)
        print(f"Baseline saved to {filepath}")

    def check_regression(self, baseline_file: Path = None) -> List[str]:
        """
        Check for performance regressions against baseline.

        Args:
            baseline_file: Path to baseline JSON file

        Returns:
            List of regression warnings
        """
        baseline_file = baseline_file or Path("baseline.json")

        if not baseline_file.exists():
            return ["No baseline file found"]

        baseline_results = self.benchmark.load_baseline(baseline_file)
        warnings = []

        # Map baseline results by name
        baseline_map = {r.name: r for r in baseline_results}

        # Check each current result against baseline
        for name, current in self.results.items():
            if name in baseline_map:
                baseline = baseline_map[name]

                if self.benchmark.check_regression(current, baseline):
                    warnings.append(
                        f"Regression in {name}: "
                        f"P99 {current.p99:.2f}ms vs baseline {baseline.p99:.2f}ms "
                        f"({(current.p99 / baseline.p99 - 1) * 100:.1f}% slower)"
                    )

        return warnings


def main():
    """Run default benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Starlighter benchmarks")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--samples", type=int, default=100, help="Sample runs")
    parser.add_argument("--save-baseline", action="store_true", help="Save as baseline")
    parser.add_argument(
        "--check-regression", action="store_true", help="Check for regressions"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Create and run benchmark suite
    suite = BenchmarkSuite(warmup=args.warmup, samples=args.samples)

    print("Starlighter Benchmark Suite")
    print("=" * 50)

    # Run benchmarks
    results = suite.run_all()

    # Print summary
    print("\n" + results["summary"])

    # Check regressions if requested
    if args.check_regression:
        warnings = suite.check_regression()
        if warnings:
            print("\n⚠️  Performance Regressions Detected:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n✅ No regressions detected")

    # Save baseline if requested
    if args.save_baseline:
        suite.save_baseline()

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0 if not (args.check_regression and suite.check_regression()) else 1


if __name__ == "__main__":
    sys.exit(main())
