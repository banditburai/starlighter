"""
Consolidated parser tests - testing BEHAVIOR not implementation.

This module tests the HTML output behavior of the parser, ensuring correct
syntax highlighting without testing internal implementation details.

Tests focus on:
- Input code → Expected HTML output
- Correct token classification in HTML
- Edge cases and error handling
- Performance requirements

Consolidated from:
- test_parser.py (main parser tests)
- test_parser_strings_basic.py (basic string tests)
- test_parser_strings_advanced.py (advanced string tests)
- test_parser_starhtml.py (StarHTML-specific parser tests)
"""

import pytest
import time
from starlighter import highlight


class TestBasicSyntax:
    """Test basic Python syntax highlighting."""

    def test_empty_input(self):
        """Test highlighting empty code."""
        html = highlight("")
        assert html == '<pre><code class="language-python"></code></pre>'

    def test_simple_identifiers(self):
        """Test highlighting of simple identifiers."""
        html = highlight("hello world _var var123 _123")

        # Check that identifiers are highlighted
        assert 'token-identifier">hello</span>' in html
        assert 'token-identifier">world</span>' in html
        assert 'token-identifier">_var</span>' in html
        assert 'token-identifier">var123</span>' in html
        assert 'token-identifier">_123</span>' in html

    def test_python_keywords(self):
        """Test recognition of Python keywords."""
        html = highlight("def class if else for while return")

        # All should be marked as keywords
        assert 'token-keyword">def</span>' in html
        assert 'token-keyword">class</span>' in html
        assert 'token-keyword">if</span>' in html
        assert 'token-keyword">else</span>' in html
        assert 'token-keyword">for</span>' in html
        assert 'token-keyword">while</span>' in html
        assert 'token-keyword">return</span>' in html

    def test_builtin_functions(self):
        """Test recognition of Python builtins."""
        html = highlight("print len range str int list dict")

        # All should be marked as builtins
        assert 'token-builtin">print</span>' in html
        assert 'token-builtin">len</span>' in html
        assert 'token-builtin">range</span>' in html
        assert 'token-builtin">str</span>' in html
        assert 'token-builtin">int</span>' in html
        assert 'token-builtin">list</span>' in html
        assert 'token-builtin">dict</span>' in html

    def test_number_literals(self):
        """Test highlighting of number literals."""
        html = highlight("42 3.14 0x1F 0b1010 0o755 1e10 2.5e-3")

        # All should be marked as numbers
        assert 'token-number">42</span>' in html
        assert 'token-number">3.14</span>' in html
        assert 'token-number">0x1F</span>' in html
        assert 'token-number">0b1010</span>' in html
        assert 'token-number">0o755</span>' in html
        assert 'token-number">1e10</span>' in html
        assert 'token-number">2.5e-3</span>' in html

    def test_operators(self):
        """Test highlighting of operators."""
        html = highlight("a + b - c * d / e == f != g <= h >= i")

        # Check operators are highlighted
        assert 'token-operator">+</span>' in html
        assert 'token-operator">-</span>' in html
        assert 'token-operator">*</span>' in html
        assert 'token-operator">/</span>' in html
        assert 'token-operator">==</span>' in html
        assert 'token-operator">!=</span>' in html
        assert 'token-operator">&lt;=</span>' in html
        assert 'token-operator">&gt;=</span>' in html

    def test_comments(self):
        """Test highlighting of comments."""
        code = """
# This is a comment
x = 5  # Inline comment
# Another comment
"""
        html = highlight(code)

        # All comments should be highlighted
        assert 'token-comment"># This is a comment</span>' in html
        assert 'token-comment"># Inline comment</span>' in html
        assert 'token-comment"># Another comment</span>' in html

    def test_decorators(self):
        """Test highlighting of decorators."""
        code = """
@property
@functools.lru_cache
@app.route("/")
def func():
    pass
"""
        html = highlight(code)

        # Decorators should be highlighted (note: only the @ part is marked as decorator)
        assert 'token-decorator">@property</span>' in html
        assert (
            'token-decorator">@functools</span>' in html
        )  # Dot notation handled separately
        assert 'token-decorator">@app</span>' in html  # Dot notation handled separately


class TestStringHandling:
    """Test all varieties of string literal handling."""

    def test_basic_string_quotes(self):
        """Test single and double quoted strings."""
        html = highlight("'hello'\n\"world\"")
        assert "token-string" in html
        assert "&#x27;hello&#x27;" in html
        assert "&quot;world&quot;" in html

    def test_triple_quoted_strings(self):
        """Test triple quoted and multiline strings."""
        code = '"""hello\nworld"""'
        html = highlight(code)
        assert "token-string" in html
        assert "&quot;&quot;&quot;hello" in html

    def test_escape_sequences(self):
        """Test that escape sequences are properly escaped in HTML."""
        html = highlight(r"'It\'s \\ ok'")
        assert "token-string" in html
        # Our escape uses numeric form
        assert "&#x27;" in html or "&#39;" in html or "&apos;" in html

    def test_string_literals_comprehensive(self):
        """Test highlighting of all string literal types."""
        code = '''
"single line"
'single quotes'
"""triple
quoted"""
f"formatted {var}"
r"raw\\string"
'''
        html = highlight(code)

        # Check all string types are highlighted
        assert 'token-string">&quot;single line&quot;</span>' in html
        assert 'token-string">&#x27;single quotes&#x27;</span>' in html
        assert 'token-string">&quot;&quot;&quot;triple' in html
        assert 'token-string">f&quot;formatted {var}&quot;</span>' in html
        assert 'token-string">r&quot;raw\\string&quot;</span>' in html

    def test_advanced_string_types(self):
        """Test raw, byte, and f-string variations."""
        code = "\n".join(
            [
                r"r'path\\to'",
                r"b'bytes'",
                r"f'val {x}'",
                r"rf'raw {x}\n'",
            ]
        )
        html = highlight(code)
        assert "token-string" in html

        # Check specific prefixes are preserved
        assert "r&#x27;path" in html
        assert "b&#x27;bytes" in html
        assert "f&#x27;val" in html
        assert "rf&#x27;raw" in html

    def test_fstring_expressions(self):
        """Test f-strings with expressions are handled."""
        code = "f'Hello {name} {len(items)}'"
        html = highlight(code)
        assert "token-string" in html
        assert "f&#x27;Hello {name} {len(items)}&#x27;" in html


class TestStarHTMLSupport:
    """Test StarHTML element recognition."""

    def test_starhtml_elements(self):
        """Test recognition of StarHTML elements."""
        html = highlight("Div Button Input Form H1 H2 Table")

        # All should be marked as StarHTML elements
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Form</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">H2</span>' in html
        assert 'token-starhtml-element">Table</span>' in html

    def test_starhtml_in_context(self):
        """Test StarHTML elements in realistic context."""
        code = """
def create_ui():
    return Div(
        H1("Title"),
        Button("Click me", onclick="handler()"),
        Input(type="text", placeholder="Enter text")
    )
"""
        html = highlight(code)

        # StarHTML elements should be highlighted
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-starhtml-element">Input</span>' in html

    def test_starhtml_with_brackets(self):
        """Test StarHTML elements with bracket notation."""
        code = "Div[H1('Title'), Button('Click')]"
        html = highlight(code)
        # Should recognize elements even with bracket notation
        assert (
            'token-starhtml-element">Div</span>' in html
            or 'token-identifier">Div</span>' in html
        )
        assert (
            'token-starhtml-element">H1</span>' in html
            or 'token-identifier">H1</span>' in html
        )
        assert (
            'token-starhtml-element">Button</span>' in html
            or 'token-identifier">Button</span>' in html
        )


class TestDataStarSupport:
    """Test DataStar attribute recognition."""

    def test_datastar_attributes(self):
        """Test recognition of DataStar attributes."""
        html = highlight("data_show data_bind data_on_click data_class")

        # All should be marked as DataStar attributes
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-datastar-attr">data_class</span>' in html

    def test_datastar_in_context(self):
        """Test DataStar attributes in realistic context."""
        code = """
element = {
    "data_show": "user.isVisible",
    "data_bind": "user.name",
    "data_on_click": "handleClick()",
    "data_class": "{'active': isActive}"
}
"""
        html = highlight(code)

        # DataStar attributes should be highlighted even in dict keys
        assert 'token-string">&quot;data_show&quot;</span>' in html
        assert 'token-string">&quot;data_bind&quot;</span>' in html

        # When used as identifiers
        code2 = "data_show = True; data_bind = 'value'"
        html2 = highlight(code2)
        assert 'token-datastar-attr">data_show</span>' in html2
        assert 'token-datastar-attr">data_bind</span>' in html2


class TestComplexCode:
    """Test highlighting of complex, realistic code."""

    def test_class_definition(self):
        """Test highlighting of a complete class."""
        code = '''
class DataProcessor:
    """Process data with various transformations."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = []
    
    @property
    def is_ready(self) -> bool:
        return len(self.data) > 0
    
    def process(self, item: str) -> dict:
        # Process the item
        result = {"value": item, "timestamp": 12345}
        self.data.append(result)
        return result
'''
        html = highlight(code)

        # Check various elements are highlighted correctly
        assert 'token-keyword">class</span>' in html
        assert 'token-identifier">DataProcessor</span>' in html
        assert 'token-string">&quot;&quot;&quot;Process data' in html
        assert 'token-keyword">def</span>' in html
        assert 'token-identifier">__init__</span>' in html
        assert 'token-decorator">@property</span>' in html
        assert 'token-comment"># Process the item</span>' in html
        assert 'token-builtin">len</span>' in html
        assert 'token-number">0</span>' in html
        assert 'token-number">12345</span>' in html

    def test_mixed_starhtml_datastar(self):
        """Test code with both StarHTML and DataStar."""
        code = """
def create_component():
    data_show = "isVisible"
    data_bind = "user.name"
    
    return Div(
        H1("User Profile"),
        Input(data_bind=data_bind),
        Button("Save", data_on_click="save()")
    )
"""
        html = highlight(code)

        # Check both are highlighted
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Button</span>' in html


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unterminated_string(self):
        """Test handling of unterminated strings."""
        html = highlight('"unterminated string')

        # Should still produce valid HTML
        assert "<pre><code" in html
        assert "</code></pre>" in html
        # String should be highlighted (even if unterminated)
        assert "token-string" in html

    def test_nested_quotes(self):
        """Test handling of nested quotes."""
        code = '''text = "She said \\"Hello\\" to me"'''
        html = highlight(code)

        # Should handle escaped quotes correctly
        assert "token-string" in html
        assert "&quot;She said \\&quot;Hello\\&quot; to me&quot;" in html

    def test_very_long_line(self):
        """Test handling of very long lines."""
        # Create a line with 1000 characters
        long_line = "x = " + "a + " * 250 + "b"
        html = highlight(long_line)

        # Should handle without issues
        assert "<pre><code" in html
        assert "</code></pre>" in html
        assert 'token-identifier">x</span>' in html
        assert 'token-operator">+</span>' in html

    def test_unicode_characters(self):
        """Test handling of Unicode characters."""
        code = 'message = "Hello 世界 🌍"'
        html = highlight(code)

        # Should preserve Unicode characters
        assert "世界" in html
        assert "🌍" in html
        assert "token-string" in html

    def test_html_injection_prevention(self):
        """Test that HTML in code is properly escaped."""
        code = """
text = "<script>alert('xss')</script>"
html = "<img src=x onerror=alert(1)>"
"""
        html = highlight(code)

        # HTML should be escaped
        assert "&lt;script&gt;" in html
        assert "&lt;img" in html
        assert "alert" in html  # The text should be there
        assert "<script>" not in html  # But not as actual HTML
        assert "<img" not in html


class TestPerformance:
    """Test performance requirements."""

    def test_small_file_performance(self):
        """Test performance on small files (<100 lines)."""
        code = "\n".join(["x = 1" for _ in range(50)])

        # Warm up
        for _ in range(3):
            highlight(code)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            highlight(code)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        # P99 should be < 5ms for small files (allow 50% variance in CI)
        import os

        threshold = 7.5 if os.getenv("CI") else 5.0  # 50% more lenient in CI
        p99 = sorted(times)[9]
        assert p99 < threshold, f"P99 was {p99:.2f}ms, expected < {threshold}ms"

    def test_medium_file_performance(self):
        """Test performance on medium files (~500 lines)."""
        code = "\n".join([f"def func_{i}():\n    return {i} * 2" for i in range(250)])

        # Warm up
        for _ in range(3):
            highlight(code)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            highlight(code)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        # P99 should be < 10ms for medium files (allow 50% variance in CI)
        import os

        threshold = 15.0 if os.getenv("CI") else 10.0  # 50% more lenient in CI
        p99 = sorted(times)[9]
        assert p99 < threshold, f"P99 was {p99:.2f}ms, expected < {threshold}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
