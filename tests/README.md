# Starlighter v2 Test Suite Documentation

## Overview

This document provides comprehensive documentation for the Starlighter v2 test suite, organized using a behavior-driven testing approach that focuses on input-output validation rather than implementation details.

## Test Organization Strategy

The test suite follows a **consolidated, behavior-focused approach** with these key principles:

1. **Behavior Testing**: Tests validate input → output behavior, not internal implementation
2. **Consolidation**: Related tests are grouped into comprehensive files to eliminate duplication
3. **TDD Approach**: Tests drive implementation and serve as living documentation
4. **Performance Integration**: Performance requirements are embedded throughout the test suite
5. **Framework Integration**: Tests validate real-world usage patterns with FastHTML/StarHTML

## Test File Organization

### Core Parser Tests

#### `test_parser_consolidated.py` (306 lines)
**Purpose**: Consolidated parser behavior tests focusing on HTML output validation

**Coverage Areas**:
- Basic Python syntax highlighting (keywords, identifiers, operators)
- String handling (single/double quotes, f-strings, multiline strings)
- Number parsing (integers, floats, scientific notation, binary/hex)
- Comment processing and special token recognition
- DataStar attribute recognition (`data_*` attributes)
- StarHTML element highlighting
- Error handling and edge cases
- Performance validation (inline performance checks)

**Key Test Classes**:
- `TestBasicSyntax`: Core Python syntax elements
- `TestStringHandling`: All string types and edge cases
- `TestNumberParsing`: Numeric literal recognition
- `TestDataStarAttributes`: DataStar attribute highlighting
- `TestStarHtmlElements`: StarHTML element recognition
- `TestEdgeCasesAndErrors`: Error handling and malformed input

#### `test_integration_core.py` (589 lines)
**Purpose**: Core integration and end-to-end workflow validation

**Coverage Areas**:
- Complete pipeline from raw code to HTML output
- API integration across all components
- Theme system integration
- Security validation (XSS prevention, input sanitization)
- Performance characteristics and regression detection
- Memory usage and garbage collection behavior
- Real-world code sample processing

**Key Test Classes**:
- `TestEndToEndWorkflow`: Complete processing pipeline
- `TestApiIntegration`: Integration between components
- `TestThemeIntegration`: Theme system functionality
- `TestSecurityValidation`: Security feature verification
- `TestPerformanceCharacteristics`: Performance regression detection
- `TestMemoryManagement`: Memory usage validation
- `TestRealWorldSamples`: Real code processing

#### `test_integration_features.py` (445 lines)
**Purpose**: DataStar/StarHTML feature integration validation

**Coverage Areas**:
- DataStar attribute recognition and highlighting
- StarHTML element processing
- Mixed usage patterns and complex scenarios
- Framework integration patterns (FastHTML integration)
- Real-world usage validation
- Performance of feature-specific code paths

**Key Test Classes**:
- `TestDataStarAttributeRecognition`: DataStar attribute handling
- `TestStarHtmlElementRecognition`: StarHTML element processing
- `TestMixedUsagePatterns`: Complex real-world scenarios
- `TestFrameworkIntegration`: FastHTML integration patterns
- `TestFeaturePerformance`: Feature-specific performance validation

### API and Function Tests

#### `test_highlight_api.py` (233 lines)
**Purpose**: Main API entry point validation

**Coverage Areas**:
- `highlight()` function behavior and signature
- Error handling and input validation
- Backward compatibility maintenance
- Performance requirements (<5ms for typical files)
- Edge case handling

**Key Test Classes**:
- `TestHighlightApiBasic`: Core `highlight()` function behavior
- `TestHighlightApiPerformance`: Performance validation
- `TestHighlightApiEdgeCases`: Error handling and edge cases
- `TestHighlightApiCompatibility`: Backward compatibility

#### `test_highlight_function.py` (269 lines)
**Purpose**: Highlight function implementation details

**Coverage Areas**:
- Function parameter validation
- Return value structure and consistency
- Integration with parser components
- Performance characteristics
- Error propagation and handling

### Component Tests

#### `test_themes.py` (135 lines)
**Purpose**: Theme system functionality validation

**Coverage Areas**:
- Theme loading and application
- CSS class generation
- Color scheme validation
- Default theme behavior
- Custom theme support

**Key Test Classes**:
- `TestThemeSystem`: Core theme functionality
- `TestDefaultTheme`: Default theme validation
- `TestThemeCustomization`: Custom theme support

#### `test_error_handling.py` (55 lines)
**Purpose**: Comprehensive error handling validation

**Coverage Areas**:
- Exception type validation (`InputError`, `ParseError`, `RenderError`)
- Error message clarity and consistency
- Recovery behavior and graceful degradation
- Error boundary testing

### Performance Tests

#### `test_performance.py` (183 lines)
**Purpose**: Core performance validation and benchmarking

**Coverage Areas**:
- Latency requirements validation (P99 < 5ms for 500 lines)
- Memory usage profiling
- Performance regression detection
- Scaling behavior analysis
- Garbage collection impact

**Key Test Classes**:
- `TestCorePerformance`: Basic performance validation
- `TestPerformanceRegression`: Regression detection
- `TestMemoryProfiling`: Memory usage analysis
- `TestScalingBehavior`: Performance scaling validation

#### `test_performance_enhanced.py` (312 lines)
**Purpose**: Advanced performance analysis and CI integration

**Coverage Areas**:
- Detailed latency analysis and scaling studies
- CI/CD performance gates
- PRD requirement validation
- Statistical performance analysis
- Performance reporting and metrics

**Key Test Classes**:
- `TestPRDRequirements`: PRD requirement validation
- `TestDetailedLatencyAnalysis`: Advanced timing analysis
- `TestMemoryProfiler`: Detailed memory profiling
- `TestP99LatencyBenchmarks`: P99 latency validation
- `TestCIPerformanceGates`: CI/CD integration gates

## How to Run Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_parser_consolidated.py -v

# Run specific test class
uv run pytest tests/test_integration_core.py::TestEndToEndWorkflow -v

# Run specific test method
uv run pytest tests/test_highlight_api.py::TestHighlightApiBasic::test_highlight_uses_unified_parser -v
```

### Coverage Testing

```bash
# Generate coverage report
uv run pytest --cov=starlighter --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=starlighter --cov-report=html

# Coverage with specific minimum threshold
uv run pytest --cov=starlighter --cov-fail-under=85
```

### Performance Testing

```bash
# Run performance tests only
uv run pytest tests/test_performance.py tests/test_performance_enhanced.py -v

# Run performance tests with detailed output
uv run pytest tests/test_performance_enhanced.py::TestPRDRequirements -v -s

# Quick performance validation
uv run pytest tests/test_performance.py::TestCorePerformance::test_latency_requirement -v
```

## How to Run Benchmarks

### Built-in Benchmarking Framework

```bash
# Run comprehensive benchmarks using the integrated framework
uv run python -c "from starlighter.benchmark.runner import BenchmarkRunner; BenchmarkRunner().run_comprehensive_suite()"

# Quick benchmark check
uv run python benchmark_final.py

# Run baseline comparison
uv run python -c "
from starlighter.benchmark.framework import PerformanceBenchmark
from starlighter import highlight
benchmark = PerformanceBenchmark()
code = 'def hello(): return \"world\"' * 100
result = benchmark.benchmark_function(highlight, code)
print(f'Performance: {result[\"mean\"]*1000:.3f}ms ± {result[\"std\"]*1000:.3f}ms')
"
```

### Shell Script Benchmarks

```bash
# Run benchmarks with CI reporting
./run_benchmarks.sh

# Generate new baseline (after performance improvements)
uv run python -c "
from starlighter.benchmark.runner import BenchmarkRunner
runner = BenchmarkRunner()
runner.create_baseline('baseline.json')
"
```

## Performance Requirements

The test suite enforces these performance requirements based on the PRD:

### Core Performance Targets

| Metric | Requirement | Test Validation |
|--------|-------------|-----------------|
| **P99 Latency** | < 5ms for 500 lines | `test_performance.py::TestCorePerformance` |
| **Cold Start** | < 100ms import time | `test_performance_enhanced.py::TestPRDRequirements` |
| **Memory Usage** | < 50MB peak usage | Memory profiler tests |
| **Package Size** | < 200KB total | CI performance gates |
| **Accuracy** | ≥ 99.9% correct highlighting | Accuracy benchmark tests |

### Scaling Requirements

- **Linear Scaling**: Performance should scale roughly linearly with input size
- **Memory Efficiency**: No memory leaks or excessive garbage collection
- **Consistent Performance**: P99 latency should remain stable across different input types

### Test Integration

Performance requirements are validated through:

1. **Inline Performance Checks**: Each major test file includes performance assertions
2. **Dedicated Performance Tests**: Comprehensive performance test suite
3. **CI Performance Gates**: Automated regression detection
4. **Benchmark Integration**: Regular benchmarking with baseline comparisons

## Test Fixtures and Utilities

### Shared Fixtures

#### `tests/fixtures/sample_code.py`
Standard code samples used across multiple test files for consistency.

#### `tests/fixtures/complex_examples.py`
Real-world complex code examples for integration testing.

#### `tests/fixtures/accuracy_benchmark.py`
Accuracy validation tools and test corpus generation.

#### `tests/fixtures/xss_attack_vectors.py`
Security testing vectors for XSS prevention validation.

### Test Utilities

#### `tests/benchmarks/sample_files.py`
Performance testing utilities and sample file generation.

## Key Testing Patterns

### 1. Behavior-Driven Testing

```python
def test_keyword_highlighting(self):
    """Test Python keywords are properly highlighted."""
    code = "def function(): return None"
    html = highlight(code)
    
    # Test behavior: input code → expected HTML output
    assert 'token-keyword">def</span>' in html
    assert 'token-keyword">return</span>' in html
    assert 'token-keyword">None</span>' in html
```

### 2. Performance Integration

```python
def test_with_performance_validation(self):
    """Test functionality with embedded performance check."""
    code = generate_test_code(lines=500)
    
    start_time = time.perf_counter()
    html = highlight(code)
    execution_time = time.perf_counter() - start_time
    
    # Validate functionality
    assert '<pre><code class="language-python">' in html
    
    # Validate performance (P99 requirement)
    assert execution_time < 0.005, f"Performance regression: {execution_time*1000:.3f}ms > 5ms"
```

### 3. Security Validation

```python
def test_xss_prevention(self):
    """Test XSS attack prevention in output."""
    malicious_code = '<script>alert("xss")</script>'
    html = highlight(malicious_code)
    
    # Ensure dangerous content is escaped
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
```

## Maintenance Guidelines

### Adding New Tests

1. **Choose Appropriate File**: Add tests to existing consolidated files when possible
2. **Follow Naming Convention**: Use descriptive test method names starting with `test_`
3. **Include Performance Checks**: Add timing assertions for new functionality
4. **Document Expected Behavior**: Include clear docstrings explaining test purpose
5. **Use Shared Fixtures**: Leverage existing fixtures for consistency

### Performance Test Maintenance

1. **Update Baselines**: Regenerate baselines after legitimate performance improvements
2. **Monitor CI Gates**: Review failed CI performance gates for regressions
3. **Scaling Analysis**: Regularly review scaling behavior with different input sizes
4. **Memory Profiling**: Include memory usage validation for new features

### Coverage Maintenance

1. **Target 85%+ Coverage**: Maintain high test coverage across all modules
2. **Identify Gaps**: Use coverage reports to find untested code paths
3. **Quality over Quantity**: Focus on meaningful tests rather than coverage metrics
4. **Regular Reviews**: Periodically review coverage reports for gaps

## Troubleshooting

### Common Test Issues

#### Performance Test Failures

```bash
# Check if system is under load
uv run pytest tests/test_performance.py::TestCorePerformance::test_latency_requirement -v -s

# Run with fewer iterations for debugging
uv run pytest tests/test_performance_enhanced.py -k "not test_latency_scaling_analysis" -v
```

#### Coverage Issues

```bash
# Generate detailed coverage report
uv run pytest --cov=starlighter --cov-report=html --cov-report=term-missing

# Check specific module coverage
uv run pytest --cov=starlighter.parser --cov-report=term-missing
```

#### Import/Setup Issues

```bash
# Verify project setup
uv sync

# Check Python path
uv run python -c "import starlighter; print(starlighter.__file__)"

# Verify test discovery
uv run pytest --collect-only tests/
```

### Performance Debugging

When performance tests fail:

1. **Check System Load**: Ensure system isn't under heavy load
2. **Review Recent Changes**: Check for code changes that might impact performance
3. **Run Profiling**: Use built-in profiler tools for detailed analysis
4. **Compare Baselines**: Check if baseline needs updating after improvements
5. **Isolate Issues**: Run individual test methods to isolate specific problems

---

This test suite provides comprehensive validation of the Starlighter v2 codebase while maintaining focus on behavior rather than implementation, ensuring robust and maintainable code quality.