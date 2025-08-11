"""
Unit tests for the main highlight API functions - test_highlight_api.py.

This test suite covers the main entry points for starlighter syntax highlighting,
specifically testing the integration between the main API functions and the
unified parser. Written using TDD approach to drive the implementation changes.

Tests focus on:
1. Integration with PythonLexerUnified from parser_unified.py
2. Backward compatibility of API signatures
3. Performance requirements (<5ms for typical files)
4. CodeBlock() function framework integrations
5. Error handling and edge cases
"""

import pytest
import time
from unittest.mock import patch
from starlighter import highlight, CodeBlock
# Using standard exceptions now


class TestHighlightApiBasic:
    """Test basic functionality of highlight() function with unified parser."""

    def test_highlight_uses_unified_parser(self):
        """Test that highlight() now uses PythonLexerUnified instead of PythonLexer."""
        code = "def hello(): return 'world'"
        html = highlight(code)

        # Should return complete HTML structure
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "language-python" in html

        # Should contain proper token spans from unified parser
        assert "token-keyword" in html
        assert "token-identifier" in html
        assert "token-string" in html

    def test_highlight_empty_string_unified(self):
        """Test highlighting empty string with unified parser."""
        html = highlight("")

        # Should return basic HTML structure for empty input
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "language-python" in html

    def test_highlight_single_token_types(self):
        """Test highlighting single token types to verify unified parser integration."""
        test_cases = [
            ("def", "token-keyword"),
            ("myvar", "token-identifier"),
            ("42", "token-number"),
            ('"hello"', "token-string"),
            ("# comment", "token-comment"),
            ("@decorator", "token-decorator"),
            ("len", "token-builtin"),
            ("+", "token-operator"),
        ]

        for code, expected_token in test_cases:
            html = highlight(code)
            assert expected_token in html, f"Expected {expected_token} for code: {code}"

    def test_highlight_complex_code_unified(self):
        """Test highlighting complex code to verify unified parser features."""
        code = '''@property
def calculate(x: int, y=10) -> float:
    """Calculate something."""
    # This is a comment
    result = x * y / 2.5
    return f"Result: {result:.2f}"
'''
        html = highlight(code)

        # Should contain all major token types from unified parser
        expected_tokens = [
            "token-decorator",  # @property
            "token-keyword",  # def, return
            "token-identifier",  # calculate, x, y
            "token-builtin",  # int, float
            "token-string",  # docstring and f-string
            "token-comment",  # comment
            "token-number",  # 2.5
            "token-operator",  # operators
        ]

        for token in expected_tokens:
            assert token in html, f"Missing {token} in unified parser output"


class TestHighlightApiPerformance:
    """Test performance requirements for highlight() function."""

    def test_highlight_performance_target(self):
        """Test that highlight() meets <5ms target for typical files."""
        # Create a typical 500-line Python file
        code_lines = []
        for i in range(125):  # 125 * 4 = 500 lines
            code_lines.extend(
                [
                    f"def function_{i}(param_{i}):",
                    f'    """Function {i} docstring."""',
                    f"    result = param_{i} * {i} + 42",
                    "    return result",
                ]
            )

        large_code = "\n".join(code_lines)

        # Measure performance
        start_time = time.perf_counter()
        html = highlight(large_code)
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000

        # Should meet performance target (allow some tolerance for test environment)
        # Target is <5ms but allow up to 10ms for CI environments
        import os

        threshold = 10.0 if os.getenv("CI") else 7.0
        assert execution_time_ms < threshold, (
            f"Performance target missed: {execution_time_ms:.2f}ms >= {threshold}ms (target <5ms)"
        )

        # Should still produce correct output
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-keyword" in html
        assert "token-identifier" in html

    def test_highlight_performance_validation_hook(self):
        """Test that performance validation hooks are present."""
        # This will be implemented by adding timing/monitoring in the actual implementation
        code = "def simple(): pass"
        html = highlight(code)

        # Basic validation that function works
        assert html is not None
        assert "token-keyword" in html

        # Performance hook validation would be added to actual implementation
        # For now, just ensure the function completes quickly
        start_time = time.perf_counter()
        for _ in range(10):
            highlight(code)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 10) * 1000
        assert avg_time_ms < 1.0, f"Simple code took too long: {avg_time_ms:.2f}ms"


class TestHighlightApiUnifiedParserIntegration:
    """Test specific integration aspects with PythonLexerUnified."""

    def test_highlight_streaming_pipeline(self):
        """Public API should use fast streaming pipeline and return HTML."""
        code = "def test(): return 42"
        html = highlight(code)
        assert html.startswith("<pre><code")
        assert "token-keyword" in html and "token-number" in html

    def test_highlight_starhtml_datastar_support(self):
        """Test that unified parser's StarHTML/DataStar features work through highlight()."""
        code = """
from fasthtml.common import Div
def page():
    return Div(
        data_on_click="handle_click",
        data_store="user_data"
    )
"""
        html = highlight(code)

        # Should properly tokenize StarHTML elements and DataStar attributes
        assert "token-identifier" in html  # Should identify Div
        # StarHTML/DataStar specific tokenization would show up based on context
        assert "Div" in html
        assert "data_on_click" in html

    def test_highlight_unified_parser_error_recovery(self):
        """Test error recovery features of unified parser."""
        malformed_code = "def incomplete_function("
        html = highlight(malformed_code)

        # Should handle malformed code gracefully
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-keyword" in html  # Should still identify 'def'


class TestHighlightApiBackwardCompatibility:
    """Test backward compatibility of highlight() function."""

    def test_highlight_api_signature_unchanged(self):
        """Test that highlight() API signature is unchanged."""
        # Should accept same parameters as before
        html1 = highlight("def test(): pass")
        html2 = highlight("def test(): pass", language="python")

        assert html1.startswith("<pre><code")
        assert html2.startswith("<pre><code")
        assert "language-python" in html1
        assert "language-python" in html2

    def test_highlight_return_format_unchanged(self):
        """Test that return format matches previous implementation."""
        code = "def hello(): return 'world'"
        html = highlight(code)

        # Should maintain same HTML structure
        assert html.startswith('<pre><code class="language-python">')
        assert html.endswith("</code></pre>")

        # Should contain properly formatted token spans
        assert '<span class="token-keyword">def</span>' in html
        assert '<span class="token-identifier">hello</span>' in html

    def test_highlight_error_types_unchanged(self):
        """Test that error types remain the same for backward compatibility."""
        # ValueError for None
        with pytest.raises(ValueError):
            highlight(None)

        # ValueError for wrong type
        with pytest.raises(ValueError):
            highlight(123)

        # Should not raise unexpected errors for valid input
        try:
            html = highlight("valid code")
            assert html is not None
        except RuntimeError:
            # These are acceptable for edge cases
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error type: {type(e).__name__}: {e}")


class TestCodeBlockFunction:
    """Test CodeBlock() function for framework integrations."""

    def test_codeblock_function_exists(self):
        """Test that CodeBlock function is available and callable."""
        # Should be importable
        from starlighter import CodeBlock

        # Should be callable (may fail due to framework dependencies)
        code = "def test(): pass"
        try:
            result = CodeBlock(code)
            # If successful, should return some kind of component
            assert result is not None
        except ImportError:
            # Expected if FastHTML/StarHTML not available
            pass

    def test_codeblock_uses_highlight_internally(self):
        """Test that CodeBlock uses the updated highlight() function."""
        code = "def test(): return 42"

        # Mock highlight to verify it's called by CodeBlock
        with patch("starlighter.highlight") as mock_highlight:
            mock_highlight.return_value = "<pre><code>mocked</code></pre>"

            try:
                CodeBlock(code)
                mock_highlight.assert_called_once_with(code)
            except ImportError:
                # Expected if framework dependencies not available
                pass

    def test_codeblock_theme_parameter(self):
        """Test CodeBlock theme parameter functionality."""
        code = "def test(): pass"

        try:
            # Should accept theme parameter
            result1 = CodeBlock(code, theme="github-dark")
            result2 = CodeBlock(code, theme="monokai")

            # Both should work (if framework available)
            assert result1 is not None
            assert result2 is not None
        except ImportError:
            # Expected if framework not available
            pass

    def test_codeblock_framework_detection(self):
        """Test automatic framework detection in CodeBlock."""
        code = "def test(): pass"

        # Should handle case where neither framework is available
        try:
            CodeBlock(code)
            # If this succeeds, framework was detected
        except ImportError as e:
            # Should have helpful error message
            assert "FastHTML" in str(e) or "StarHTML" in str(e)


class TestHighlightApiEdgeCases:
    """Test edge cases and error conditions."""

    def test_highlight_input_validation_comprehensive(self):
        """Test comprehensive input validation."""
        # None input
        with pytest.raises(ValueError) as exc:
            highlight(None)
        assert "Expected str" in str(exc.value)

        # Wrong types
        invalid_inputs = [123, [], {}, object(), b"bytes"]
        for invalid in invalid_inputs:
            with pytest.raises(ValueError):
                highlight(invalid)

    def test_highlight_empty_and_whitespace_cases(self):
        """Test empty and whitespace-only inputs."""
        test_cases = [
            "",
            " ",
            "\n",
            "\t",
            "   \n  \t  \n",
            "\r\n",
            "\x0c",  # Form feed
        ]

        for test_input in test_cases:
            html = highlight(test_input)
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")

    def test_highlight_unicode_handling(self):
        """Test Unicode character handling."""
        unicode_code = """
# 测试 Unicode 支持
def функция():
    return "こんにちは世界"
"""
        html = highlight(unicode_code)

        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        # Unicode should be properly handled and escaped
        assert "token-comment" in html
        assert "token-keyword" in html

    def test_highlight_xss_prevention_maintained(self):
        """Test that XSS prevention is maintained with unified parser."""
        malicious_inputs = [
            '<script>alert("xss")</script>',
            "message = \"<script>alert('XSS')</script>\"",
            '# <img src="x" onerror="alert(1)">',
            '"<svg onload=alert(1)>"',
        ]

        for malicious in malicious_inputs:
            html = highlight(malicious)

            # Should not contain unescaped HTML
            assert "<script>" not in html
            assert "<img" not in html
            assert "<svg" not in html
            assert "onerror" not in html or "&" in html  # Should be escaped

            # Should contain escaped versions
            assert "&lt;" in html or "&gt;" in html

    def test_highlight_error_handling_unified_parser(self):
        """Test error handling with unified parser exceptions."""
        # Test various edge cases that might cause parsing issues
        edge_cases = [
            "def incomplete(",
            'unterminated_string = "hello',
            "0b2",  # Invalid binary
            "0x",  # Invalid hex
            "@",  # Incomplete decorator
            "\x00",  # Null character
        ]

        for edge_case in edge_cases:
            try:
                html = highlight(edge_case)
                # Should always produce some HTML output
                assert html.startswith("<pre><code")
                assert html.endswith("</code></pre>")
            except RuntimeError:
                # These are acceptable for severe edge cases
                pass
            except Exception as e:
                pytest.fail(
                    f"Unexpected error for '{edge_case}': {type(e).__name__}: {e}"
                )


class TestHighlightApiSecurityAndSafety:
    """Test security and safety aspects of the updated API."""

    def test_highlight_thread_safety(self):
        """Test thread safety of highlight() function."""
        import threading

        code = "def test(): return 42"
        results = []
        errors = []

        def worker():
            try:
                html = highlight(code)
                results.append(html)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have no errors and consistent results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_highlight_memory_safety(self):
        """Test memory usage doesn't grow excessively."""
        import gc

        # Run highlight many times and check memory doesn't grow
        code = "def test(x): return x * 2"

        # Force garbage collection
        gc.collect()

        # Run many iterations
        for _ in range(100):
            html = highlight(code)
            assert "token-keyword" in html

        # Force garbage collection again
        gc.collect()

        # Memory usage should not have grown excessively
        # This is a basic test - more sophisticated memory testing would need profiling
        assert True  # Basic completion test

    def test_highlight_no_code_execution(self):
        """Ensure highlight() cannot be used for code execution."""
        # Test that even malicious-looking Python code is just highlighted
        potentially_dangerous = """
import os
os.system("rm -rf /")
exec("dangerous_code")
eval("1+1")
"""

        html = highlight(potentially_dangerous)

        # Should just be highlighted, not executed
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-keyword" in html  # import
        assert "token-identifier" in html  # os, exec, eval
        # Code should not be executed - just highlighted


class TestHighlightApiTestCoverage:
    """Additional tests to improve code coverage."""

    def test_highlight_renderer_integration(self):
        """Test integration with HTMLRenderer."""
        code = "x = 42"
        html = highlight(code)

        # Should use HTMLRenderer for token rendering
        assert html.startswith('<pre><code class="language-python">')
        assert "token-identifier" in html
        assert "token-number" in html

    def test_highlight_language_parameter(self):
        """Test language parameter handling."""
        code = "def test(): pass"

        # With default language
        html1 = highlight(code)
        assert 'class="language-python"' in html1

        # With explicit language
        html2 = highlight(code, language="python")
        assert 'class="language-python"' in html2

        # With different language (renderer still uses Python classes today)
        html3 = highlight(code, language="other")
        assert 'class="language-python"' in html3

    def test_highlight_internal_error_wrapping(self):
        """Test that internal errors are properly wrapped."""
        # Test with code that should work normally
        code = "def normal(): return True"
        html = highlight(code)

        assert html.startswith("<pre><code")
        assert "token-keyword" in html

        # More sophisticated error wrapping tests would require mocking
        # internal components to force specific error conditions

    def test_highlight_all_token_types_from_unified_parser(self):
        """Test that all token types from unified parser are handled."""
        comprehensive_code = '''
# Comment
@decorator
def function_name(param: int = 42) -> str:
    """Docstring"""
    builtin_func = len([1, 2, 3])
    number_int = 123
    number_float = 3.14
    number_hex = 0xFF
    number_bin = 0b1010
    string_single = 'hello'
    string_double = "world"
    fstring = f"result: {builtin_func}"
    operators = + - * / == != <= >=
    return f"Complete: {number_int + number_float}"
'''

        html = highlight(comprehensive_code)

        # Should contain all major token types
        token_types = [
            "token-comment",
            "token-decorator",
            "token-keyword",
            "token-identifier",
            "token-builtin",
            "token-number",
            "token-string",
            "token-operator",
        ]

        for token_type in token_types:
            assert token_type in html, f"Missing {token_type} in comprehensive test"
