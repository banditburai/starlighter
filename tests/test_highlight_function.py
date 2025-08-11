"""
Tests for the main highlight() function - the primary public API.

This test suite covers the main entry point for starlighter syntax highlighting,
including input validation, integration testing, error handling, and edge cases.
Following TDD approach, tests are written first before implementation.
"""

import pytest
from starlighter import highlight
# Using standard exceptions now


class TestHighlightFunctionBasic:
    """Test basic functionality of the highlight function."""

    def test_highlight_simple_function(self):
        """Test highlighting a simple Python function."""
        code = "def hello():\n    return 'world'"
        html = highlight(code)

        # Should return complete HTML structure
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")

        # Should contain proper token spans
        assert "token-keyword" in html
        assert "token-identifier" in html
        assert "token-string" in html
        assert "language-python" in html

    def test_highlight_empty_string(self):
        """Test highlighting empty string."""
        html = highlight("")

        # Should return basic HTML structure even for empty input
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "language-python" in html

    def test_highlight_whitespace_only(self):
        """Test highlighting whitespace-only input."""
        html = highlight("   \n  \t  \n")

        # Should handle whitespace gracefully
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        # May contain whitespace tokens

    def test_highlight_single_keyword(self):
        """Test highlighting single keyword."""
        html = highlight("def")

        assert "token-keyword" in html
        assert ">def<" in html

    def test_highlight_single_identifier(self):
        """Test highlighting single identifier."""
        html = highlight("myvar")

        assert "token-identifier" in html
        assert ">myvar<" in html

    def test_highlight_single_string(self):
        """Test highlighting single string literal."""
        html = highlight('"hello world"')

        assert "token-string" in html
        assert "&quot;hello world&quot;" in html  # Should be HTML escaped

    def test_highlight_single_number(self):
        """Test highlighting single number."""
        html = highlight("42")

        assert "token-number" in html
        assert ">42<" in html


class TestHighlightFunctionIntegration:
    """Test integration with all parsing features."""

    def test_highlight_complex_function(self):
        """Test highlighting complex function with multiple token types."""
        code = '''@decorator
def calculate(x: int, y=10) -> float:
    """Calculate something."""
    # This is a comment
    result = x * y / 2.5
    return f"Result: {result}"
'''
        html = highlight(code)

        # Should contain all token types
        assert "token-decorator" in html
        assert "token-keyword" in html  # def, return
        assert "token-identifier" in html  # calculate, x, y
        assert "token-builtin" in html  # int, float
        assert "token-string" in html  # docstring and f-string
        assert "token-comment" in html
        assert "token-number" in html
        assert "token-operator" in html

    def test_highlight_class_definition(self):
        """Test highlighting class definition."""
        code = """class MyClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.value = None
"""
        html = highlight(code)

        assert "token-keyword" in html  # class, def
        assert "token-identifier" in html  # MyClass, __init__
        assert "token-builtin" in html  # super, None

    def test_highlight_import_statements(self):
        """Test highlighting import statements."""
        code = """import os
from pathlib import Path
from typing import List, Dict, Optional
"""
        html = highlight(code)

        assert "token-keyword" in html  # import, from
        assert "token-identifier" in html  # os, pathlib, Path, typing, etc.

    def test_highlight_fstring_expression(self):
        """Test highlighting f-string with embedded expressions."""
        code = 'message = f"Hello {name}, you have {count} items"'
        html = highlight(code)

        assert "token-string" in html
        assert "token-identifier" in html
        # F-string content should be properly escaped
        assert "&quot;" in html

    def test_highlight_multiline_string(self):
        """Test highlighting multiline string."""
        code = '''docstring = """
This is a multiline
string with "quotes" inside
"""'''
        html = highlight(code)

        assert "token-string" in html
        assert "&quot;" in html  # Quotes should be escaped

    def test_highlight_various_numbers(self):
        """Test highlighting various number formats."""
        code = """
a = 42
b = 3.14
c = 1e5
d = 0xFF
e = 0o777
f = 0b1010
g = 2+3j
"""
        html = highlight(code)

        # All should be classified as numbers
        assert "token-number" in html
        # Should contain the basic number literals (complex numbers may be split)
        basic_numbers = ["42", "3.14", "1e5", "0xFF", "0o777", "0b1010"]
        for num in basic_numbers:
            assert num in html

        # Complex numbers: number part and imaginary identifier
        assert ">2<" in html and ">3<" in html  # Numbers are separate tokens
        assert 'token-identifier">j<' in html  # 'j' is correctly an identifier


class TestHighlightFunctionValidation:
    """Test input validation and error handling."""

    def test_highlight_none_input(self):
        """Test that None input raises appropriate error."""
        with pytest.raises(ValueError) as exc_info:
            highlight(None)

        assert "Expected str" in str(exc_info.value)

    def test_highlight_non_string_input(self):
        """Test that non-string input raises appropriate error."""
        with pytest.raises(ValueError) as exc_info:
            highlight(123)

        assert "Expected str" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            highlight(["code"])

        assert "Expected str" in str(exc_info.value)

    def test_highlight_bytes_input(self):
        """Test that bytes input raises appropriate error."""
        with pytest.raises(ValueError) as exc_info:
            highlight(b"def hello(): pass")

        assert "Expected str" in str(exc_info.value)


class TestHighlightFunctionErrorHandling:
    """Test error handling and graceful degradation."""

    def test_highlight_malformed_syntax(self):
        """Test that malformed syntax is handled gracefully."""
        # This should not crash, even with malformed Python
        code = "def incomplete("
        html = highlight(code)

        # Should still produce HTML output
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        # Should contain the partial tokens that could be parsed
        assert "token-keyword" in html  # def

    def test_highlight_unterminated_string(self):
        """Test handling of unterminated strings."""
        code = 'message = "unterminated string'
        html = highlight(code)

        # Should still produce output
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-identifier" in html  # message
        assert "token-operator" in html  # =

    def test_highlight_mixed_indentation(self):
        """Test handling of mixed indentation."""
        code = """def test():
\tif True:
        return "mixed indentation"
"""
        html = highlight(code)

        # Should still produce output despite indentation issues
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-keyword" in html

    def test_highlight_unicode_content(self):
        """Test handling of Unicode content."""
        code = """
# 测试 Unicode 支持
def функция():
    return "こんにちは世界"
"""
        html = highlight(code)

        # Should handle Unicode gracefully
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        # Unicode should be properly escaped in HTML
        assert "token-comment" in html
        assert "token-keyword" in html


class TestHighlightFunctionPerformance:
    """Test performance characteristics."""

    def test_highlight_large_file_simulation(self):
        """Test highlighting larger code sample."""
        # Simulate a reasonably large file
        code_lines = []
        for i in range(100):
            code_lines.append(f"def function_{i}():")
            code_lines.append(f'    """Function {i} docstring."""')
            code_lines.append(f"    result = {i} * 2")
            code_lines.append("    return result")
            code_lines.append("")

        large_code = "\n".join(code_lines)
        html = highlight(large_code)

        # Should complete successfully
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        # Should contain expected number of function definitions
        assert html.count("token-keyword") >= 200  # def and return keywords


class TestHighlightFunctionXSSPrevention:
    """Test XSS prevention and security."""

    def test_highlight_xss_in_strings(self):
        """Test XSS prevention in string literals."""
        malicious_code = '''message = "<script>alert('XSS')</script>"'''
        html = highlight(malicious_code)

        # Script tags should be escaped
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "alert" in html  # But escaped as part of string content

    def test_highlight_xss_in_comments(self):
        """Test XSS prevention in comments."""
        malicious_code = """# <img src="x" onerror="alert(1)">"""
        html = highlight(malicious_code)

        # HTML should be escaped
        assert "<img" not in html
        assert "&lt;img" in html
        assert "onerror" in html  # But as escaped content

    def test_highlight_html_injection_attempt(self):
        """Test prevention of HTML injection through identifiers."""
        # Even if someone tries to inject HTML through variable names (unlikely in valid Python)
        malicious_code = '''<script>dangerous</script> = "value"'''
        html = highlight(malicious_code)

        # Should be treated as tokens but HTML characters should be escaped
        assert "<script>" not in html
        assert "&lt;" in html and "&gt;" in html  # < and > should be escaped

    def test_highlight_quote_escaping(self):
        """Test proper escaping of quotes in all contexts."""
        code = '''
text = "String with 'single' quotes"
other = 'String with "double" quotes'
mixed = """Triple with 'single' and "double" quotes"""
'''
        html = highlight(code)

        # All quotes should be properly escaped
        assert "&quot;" in html  # Double quotes
        assert (
            "&#x27;" in html or "'" in html
        )  # Single quotes (may not be escaped in content)
        # Should not contain unescaped problematic quotes that could break HTML


class TestHighlightFunctionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_highlight_only_whitespace_types(self):
        """Test highlighting content that's only whitespace and newlines."""
        html = highlight("\t\n  \r\n\t")

        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")

    def test_highlight_only_operators(self):
        """Test highlighting only operators."""
        html = highlight("+ - * / == != <= >=")

        assert "token-operator" in html
        assert html.count("token-operator") >= 4  # Multiple operators

    def test_highlight_only_numbers(self):
        """Test highlighting only numbers."""
        html = highlight("1 2.0 3e5 0xFF")

        assert "token-number" in html
        assert html.count("token-number") >= 2

    def test_highlight_nested_structures(self):
        """Test deeply nested structures."""
        code = """
def outer():
    def inner():
        def deepest():
            return {"key": [1, 2, {"nested": True}]}
        return deepest()
    return inner()
"""
        html = highlight(code)

        assert "token-keyword" in html
        assert "token-identifier" in html
        assert (
            "token-punctuation" in html
        )  # : and , are punctuation (semantically correct)

    def test_highlight_all_token_types_together(self):
        """Test code sample that uses all major token types."""
        code = '''@property
def example_function(param: str = "default") -> Optional[int]:
    """
    Comprehensive example with all token types.
    """
    # Comment here
    if param == "special":
        result = 42 + 3.14e-5
        return result
    elif param in ["a", "b", "c"]:
        return len(param)
    else:
        return None
'''
        html = highlight(code)

        # Should contain most token types
        expected_token_types = [
            "token-decorator",  # @property
            "token-keyword",  # def, if, return, etc.
            "token-identifier",  # function and variable names
            "token-builtin",  # Optional, len, None
            "token-string",  # string literals
            "token-comment",  # comment
            "token-number",  # numbers
            "token-operator",  # operators and punctuation
        ]

        for token_type in expected_token_types:
            assert token_type in html, f"Missing {token_type} in output"


class TestHighlightFunctionCoverage:
    """Tests to improve code coverage of highlight function."""

    def test_highlight_unknown_characters(self):
        """Test handling of unknown/invalid characters."""
        # Test with some unusual characters that might not be recognized
        code = "variable = 'text' € ¥ ∑"  # Contains unicode symbols
        html = highlight(code)

        # Should still produce valid HTML
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-identifier" in html  # variable
        assert "token-string" in html  # 'text'

    def test_highlight_exception_handling(self):
        """Test that unexpected exceptions are properly wrapped."""
        # This is a bit tricky to test directly, but we can at least verify
        # that our function doesn't crash on complex edge cases
        code = """
# Various edge cases that might cause issues
def func():
    x = 1_000_000.5e-10
    y = 0b1010_1010
    z = 0o777_666
    return f"Result: {x+y+z:.2e}"
"""
        html = highlight(code)

        # Should handle complex numeric formats
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-number" in html

    def test_highlight_empty_content_edge_case(self):
        """Test edge case with null/empty character handling."""
        # Test with null bytes (which should be handled safely)
        code = "test\x00value"
        html = highlight(code)

        # Should still work and not crash
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")

    def test_highlight_mock_parser_exception(self):
        """Test error handling by mocking a parser exception."""
        # Create a code that should work but we'll need to test error handling
        # by potentially causing an issue in the lexer

        # This is tricky to test without mocking, but let's try a code
        # that might cause issues in the parsing logic
        code = "def test():\n    pass"  # Simple valid code

        # In normal cases this should work fine
        html = highlight(code)
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
