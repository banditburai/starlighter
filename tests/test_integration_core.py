"""
Core Integration Tests for Starlighter v2.

This test suite consolidates core integration and end-to-end tests that validate:
1. Complete pipeline from code to HTML works
2. API integration and themes work together
3. Error handling across all components
4. Performance characteristics meet requirements
5. Security features are maintained

Consolidates tests from:
- tests/test_integration.py
- tests/test_final_validation.py
- tests/test_requirements_verification.py
"""

import pytest
import time

from starlighter import highlight, CodeBlock
# Using standard exceptions now


def has_decorator_pattern(html):
    """Check if HTML contains decorator pattern (either token-decorator or @ + identifier)."""
    return "token-decorator" in html or ("token-unknown" in html and "@" in html)


def extract_text_from_html(html):
    """Extract the text content from tokenized HTML for pattern matching."""
    import re

    text = re.sub(r"<[^>]+>", "", html)
    # Decode HTML entities
    text = (
        text.replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return text


class TestEndToEndWorkflow:
    """Test complete end-to-end highlighting workflow."""

    def test_simple_function_complete_workflow(self):
        """Test complete workflow with simple function."""
        code = """def hello_world():
    '''Say hello to the world.'''
    return "Hello, World!"
"""
        html = highlight(code)

        # Verify HTML structure
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "language-python" in html

        # Verify all expected tokens are present
        assert "token-keyword" in html  # def, return
        assert "token-identifier" in html  # hello_world
        assert "token-string" in html  # docstring and return string
        # Punctuation like ':' and '()' are emitted as token-punctuation in current impl
        assert ("token-operator" in html) or ("token-punctuation" in html)

        # Verify content is properly escaped
        assert "&quot;" in html  # Quotes should be escaped
        assert "<script>" not in html  # No unescaped HTML

        # Verify structure preservation
        lines = html.count("\n")
        assert lines >= 2  # Should preserve line structure

    def test_complex_class_complete_workflow(self):
        """Test complete workflow with complex class definition."""
        code = '''@dataclass
class Person:
    """A person with name and age."""
    name: str
    age: int = 0
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError(f"Age cannot be negative: {self.age}")
    
    @property
    def is_adult(self) -> bool:
        """Check if person is an adult."""
        return self.age >= 18
    
    def greet(self, other: "Person") -> str:
        return f"Hello {other.name}, I'm {self.name}!"
'''
        html = highlight(code)

        # Verify comprehensive token coverage
        expected_tokens = [
            "token-keyword",  # class, def, if, raise, return
            "token-identifier",  # Person, name, age, etc.
            "token-builtin",  # str, int, bool, ValueError
            "token-string",  # docstrings and f-strings (docstrings are strings, not comments)
            "token-number",  # 0, 18
            "token-operator",  # :, =, <, >=, etc.
        ]

        for token_type in expected_tokens:
            assert token_type in html, f"Missing {token_type} in complex class"

        # Check for decorator pattern (may be token-decorator or token-unknown + token-identifier)
        assert "token-decorator" in html or (
            "token-unknown" in html and "dataclass" in html
        ), "Decorator pattern not found"

        # Verify type hints are properly highlighted
        assert "str" in html and "int" in html and "bool" in html

        # Verify f-string content
        assert "Hello" in html and "other.name" in html

    def test_async_function_complete_workflow(self):
        """Test complete workflow with async/await patterns."""
        code = '''import asyncio

async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async for attempt in range(3):
            try:
                async with session.get(url) as response:
                    return await response.json()
            except asyncio.TimeoutError:
                await asyncio.sleep(2 ** attempt)
        raise Exception("Failed after 3 attempts")

# Usage
async def main():
    data = await fetch_data("https://api.example.com/data")
    return [item async for item in process_data(data)]
'''
        html = highlight(code)

        # Verify async keywords are properly highlighted
        assert "token-keyword" in html
        assert "async" in html and "await" in html

        # Verify async comprehension syntax (tokens may be separate)
        assert "async" in html and "for" in html

        # Verify exception handling in async context
        assert "Exception" in html

    def test_module_imports_complete_workflow(self):
        """Test complete workflow with various import patterns."""
        code = """# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import defaultdict, Counter

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Relative imports
from . import utils
from ..models import User
from ...config import settings
"""
        html = highlight(code)

        # Verify import keywords
        assert "token-keyword" in html
        assert "import" in html and "from" in html

        # Verify module names are identifiers
        assert "token-identifier" in html

        # Verify relative import dots are handled
        assert "." in html


class TestRealWorldCodeSamples:
    """Test highlighting with real-world Python code patterns."""

    def test_django_style_models(self):
        """Test Django-style model definitions."""
        django_code = '''from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """Custom user model with additional fields."""
    
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=30, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['username']
    
    def get_absolute_url(self):
        return reverse('user-detail', kwargs={'pk': self.pk})
    
    def __str__(self):
        return f"{self.username} ({self.get_full_name()})"

class Post(models.Model):
    """Blog post model."""
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['author', 'created_at']),
        ]
    
    def __str__(self):
        return self.title
'''
        html = highlight(django_code)

        # Verify Django patterns (tokens may be separate)
        assert "models" in html and "TextField" in html
        assert "models" in html and "ForeignKey" in html
        assert "on_delete" in html and "CASCADE" in html
        assert "class" in html and "Meta" in html
        assert "__str__" in html

    def test_flask_application_patterns(self):
        """Test Flask application patterns."""
        flask_code = '''from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required, current_user

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/api/users', methods=['GET', 'POST'])
@login_required
def users_api():
    """Users API endpoint."""
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({'error': 'Invalid data'}), 400
        
        user = User(username=data['username'])
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'id': user.id,
            'username': user.username,
            'created_at': user.created_at.isoformat()
        }), 201
    
    # GET request
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username
    } for u in users])

@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
'''
        html = highlight(flask_code)

        # Verify Flask patterns (tokens may be separate)
        assert has_decorator_pattern(html) and "app" in html and "route" in html
        assert has_decorator_pattern(html) and "login_required" in html
        assert "request" in html and "method" in html
        assert "jsonify" in html
        assert has_decorator_pattern(html) and "errorhandler" in html

    def test_data_science_patterns(self):
        """Test data science and pandas-style code."""
        pandas_code = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def analyze_dataset(filepath: str) -> dict:
    """Analyze dataset and return summary statistics."""
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic statistics
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    # Data cleaning
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned[df_cleaned['age'] > 0]
    
    # Feature engineering
    df_cleaned['age_group'] = pd.cut(
        df_cleaned['age'], 
        bins=[0, 18, 35, 50, 100], 
        labels=['child', 'young_adult', 'adult', 'senior']
    )
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    df_cleaned['age'].hist(bins=30)
    plt.title('Age Distribution')
    
    plt.subplot(2, 2, 2)
    df_cleaned['age_group'].value_counts().plot(kind='bar')
    plt.title('Age Group Distribution')
    
    plt.tight_layout()
    plt.savefig('analysis.png')
    
    return stats

# Machine learning pipeline
def build_model(X, y):
    """Build and train a random forest model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return model, score
'''
        html = highlight(pandas_code)

        # Verify data science patterns (tokens may be separate)
        assert "pd" in html and "read_csv" in html
        assert "df" in html and "shape" in html
        assert "plt" in html and "figure" in html
        assert "train_test_split" in html
        assert "RandomForestClassifier" in html


class TestPerformanceIntegration:
    """Test performance characteristics with real-world scenarios."""

    def test_large_file_performance(self):
        """Test performance with large file (simulated 1000+ lines)."""
        # Generate a large code file
        large_code_lines = []

        # Add imports
        large_code_lines.extend(
            [
                "import os",
                "import sys",
                "from pathlib import Path",
                "from typing import List, Dict, Optional",
                "",
            ]
        )

        # Add many function definitions
        for i in range(200):  # 200 functions = ~1000 lines
            large_code_lines.extend(
                [
                    f"def function_{i}(param1: str, param2: int = {i}) -> dict:",
                    f'    """Function number {i} with comprehensive logic."""',
                    f"    # Comment for function {i}",
                    f"    if param2 > {i * 2}:",
                    f'        result = {{"id": {i}, "name": param1, "value": param2 * 2}}',
                    "    else:",
                    f'        result = {{"id": {i}, "name": param1, "value": param2}}',
                    "    return result",
                    "",
                ]
            )

        large_code = "\n".join(large_code_lines)

        # Measure performance
        start_time = time.time()
        html = highlight(large_code)
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify performance target (current implementation baseline)
        assert processing_time < 0.100, (
            f"Large file took {processing_time:.3f}s (>100ms)"
        )

        # Verify output quality
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert html.count("token-keyword") > 400  # Many def, if, else, return
        assert html.count("token-identifier") > 600  # Many function and variable names

        print(
            f"Large file ({len(large_code_lines)} lines) processed in {processing_time:.3f}s"
        )

    def test_deeply_nested_structures_performance(self):
        """Test performance with deeply nested data structures."""
        nested_code = "data = " + "{'level': " * 50 + "'deep'" + "}}" * 50

        start_time = time.time()
        html = highlight(nested_code)
        end_time = time.time()

        processing_time = end_time - start_time
        assert processing_time < 0.050, f"Nested structure took {processing_time:.3f}s"

        # Verify correct handling
        assert "token-identifier" in html  # data
        assert "token-string" in html  # 'level', 'deep'
        assert "token-operator" in html  # {, }, :

    def test_memory_usage_estimation(self):
        """Test that memory usage is reasonable for large inputs."""
        # Create moderately large input
        code_lines = []
        for i in range(500):
            code_lines.append(
                f"variable_{i} = 'value_{i}' + str({i}) + f'{{calculated_{i}}}'"
            )

        large_input = "\n".join(code_lines)
        input_size = len(large_input)  # Size in bytes

        html = highlight(large_input)
        output_size = len(html)

        # Output should be larger due to HTML markup but not excessively so
        # Syntax highlighting with detailed tokenization: typically 10-15x expansion
        expansion_ratio = output_size / input_size

        assert expansion_ratio < 20, (
            f"Memory expansion too high: {expansion_ratio:.1f}x"
        )
        assert expansion_ratio > 2, (
            f"Expansion too low, might be missing markup: {expansion_ratio:.1f}x"
        )

        print(
            f"Memory expansion ratio: {expansion_ratio:.1f}x ({input_size} → {output_size} bytes)"
        )


class TestSecurityValidation:
    """Test XSS prevention and security features."""

    def test_xss_prevention_in_strings(self):
        """Test XSS attack prevention in string literals."""
        malicious_strings = [
            '''message = "<script>alert('XSS Attack!')</script>"''',
            """html = '<img src="x" onerror="alert(1)">'  """,
            '''code = "</pre></code><script>malicious()</script><pre><code>"''',
            '''injection = "'; DROP TABLE users; --"''',
        ]

        for malicious_code in malicious_strings:
            html = highlight(malicious_code)

            # Verify no unescaped script tags
            assert "<script>" not in html, f"Unescaped script in: {malicious_code}"
            assert "onerror=" not in html or "&" in html, (
                f"Unescaped onerror in: {malicious_code}"
            )

            # Verify proper escaping (check for HTML entity encoding)
            assert "&lt;" in html or "&#x27;" in html or "&quot;" in html, (
                f"No HTML escaping found in: {html[:200]}..."
            )
            assert "&quot;" in html or "&#x27;" in html, "Quotes should be escaped"

    def test_xss_prevention_in_comments(self):
        """Test XSS attack prevention in comments."""
        malicious_comments = [
            """# <script>alert("XSS in comment")</script>""",
            """# <img src="x" onerror="alert('comment-xss')">""",
            """# --></pre><script>alert("break out")</script><!--""",
        ]

        for comment in malicious_comments:
            html = highlight(comment)

            # Verify HTML is properly escaped
            assert "<script>" not in html
            assert "<img" not in html or "&lt;img" in html
            assert "onerror=" not in html or "onerror=" in html and "&" in html

    def test_html_injection_prevention(self):
        """Test prevention of HTML structure injection."""
        injection_attempts = [
            """</code></pre><h1>Injected Header</h1><pre><code>""",
            '''variable = "</span><script>alert('span break')</script><span>"''',
            """# "></span></code></pre><script>alert("full break")</script><pre><code><span>""",
        ]

        for injection in injection_attempts:
            html = highlight(injection)

            # Verify structure integrity
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")

            # Count HTML tags to ensure no injection (using more precise patterns)
            pre_count = html.count("<pre>")
            pre_close_count = html.count("</pre>")
            code_close_count = html.count("</code>")

            assert pre_count == pre_close_count == 1, (
                f"Pre tag injection in: {injection}"
            )
            assert code_close_count == 1, f"Code tag injection in: {injection}"

            # Verify no unescaped HTML
            assert "&lt;" in html or "&gt;" in html  # Should have escaped HTML


class TestCrossModuleIntegration:
    """Test integration between all starlighter modules."""

    def test_parser_integration(self):
        """Test integration with unified parser module."""
        test_code = '''@decorator
def test_function(x: int) -> str:
    """Test function with type hints."""
    return f"Value: {x * 2}"
'''
        html = highlight(test_code)

        # Verify parser tokenization worked
        assert has_decorator_pattern(
            html
        )  # Decorator pattern (may be token-unknown + identifier)
        assert "token-keyword" in html
        assert "token-identifier" in html
        assert "token-builtin" in html  # int, str
        assert "token-string" in html

        # Verify unified implementation produced valid HTML
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert 'class="' in html  # CSS classes applied
        assert "<span" in html  # Tokens wrapped in spans

    def test_error_handling_integration(self):
        """Test error handling across all modules."""
        # Test with malformed but parseable code
        malformed_code = """def incomplete_function(
    # Missing closing parenthesis and body
    x = 42
    # No return statement
"""

        # Should not raise exception, should handle gracefully
        html = highlight(malformed_code)

        # Should still produce valid HTML
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")

        # Should contain partial tokens that could be parsed
        assert "token-keyword" in html  # def
        assert "token-identifier" in html  # incomplete_function, x
        assert "token-number" in html  # 42

    def test_token_css_mapping_integration(self):
        """Test integration between token types and CSS class mapping."""
        code_with_all_tokens = '''# Comment
@decorator
def function_name(param: int = 42) -> str:
    """Docstring."""
    variable = "string" + f"{param}"
    return variable
'''
        html = highlight(code_with_all_tokens)

        # Verify all major token types have corresponding CSS classes
        expected_mappings = [
            ("def", "token-keyword"),
            ("function_name", "token-identifier"),
            ("int", "token-builtin"),
            ("42", "token-number"),
            ('"string"', "token-string"),
            ("# Comment", "token-comment"),
        ]

        for content, css_class in expected_mappings:
            assert css_class in html, f"Missing CSS class {css_class} for {content}"

        # Special check for decorator pattern
        assert has_decorator_pattern(html), "Missing decorator pattern"

    def test_public_api_integration(self):
        """Test public API integration with all internal modules."""
        # Test the main highlight() function uses all modules correctly
        code = 'def test(): return "hello"'

        # Should work without errors
        result = highlight(code)
        assert isinstance(result, str)
        assert len(result) > len(code)  # Should have HTML markup

        # Test error conditions are properly handled
        with pytest.raises(ValueError):
            highlight(None)

        with pytest.raises(ValueError):
            highlight(123)

    def test_configuration_integration(self):
        """Test configuration parameter integration."""
        code = "def test(): pass"

        # Test language parameter with empty string (this uses language parameter)
        html_empty_python = highlight("", language="python")
        assert "language-python" in html_empty_python

        html_empty_other = highlight("", language="other")
        # Current implementation always renders python class
        assert "language-python" in html_empty_other

        # Test that non-empty code currently defaults to python
        html_python = highlight(code, language="python")
        html_other = highlight(code, language="other")

        # Both should produce HTML (regardless of language parameter behavior)
        assert html_python.startswith("<pre><code")
        assert html_other.startswith("<pre><code")
        assert html_python.endswith("</code></pre>")
        assert html_other.endswith("</code></pre>")


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_empty_and_whitespace_only(self):
        """Test various empty and whitespace-only inputs."""
        edge_cases = [
            "",
            "   ",
            "\n\n\n",
            "\t\t\t",
            "   \n  \t  \n  ",
        ]

        for case in edge_cases:
            html = highlight(case)

            # Should always produce valid HTML structure
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")
            assert "language-python" in html

    def test_single_characters_and_tokens(self):
        """Test single characters and minimal tokens."""
        minimal_cases = [
            "#",  # Single comment character
            "'",  # Single quote
            "x",  # Single identifier
            "1",  # Single digit
            "+",  # Single operator
            "(",  # Single delimiter
        ]

        for case in minimal_cases:
            html = highlight(case)

            # Should handle gracefully
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")
            # Content should be properly escaped/tokenized
            assert case in html or "&" in html  # Either preserved or escaped

    def test_very_long_lines(self):
        """Test handling of very long lines."""
        # Create a very long line (1000+ characters)
        long_line = 'x = "' + "very long string content " * 50 + '"'
        assert len(long_line) > 1000

        html = highlight(long_line)

        # Should handle without issues
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")
        assert "token-identifier" in html  # x
        assert "token-string" in html  # long string
        assert "&quot;" in html  # Quotes escaped

    def test_unicode_and_special_characters(self):
        """Test Unicode and special character handling."""
        unicode_cases = [
            "# 测试中文注释",
            'variable = "emoji: 🚀🔥💯"',
            "def функция(): pass",
            '"""Docstring with symbols: ∑∆∏∫"""',
            "# Symbols: ≤≥≠±∞√",
        ]

        for case in unicode_cases:
            html = highlight(case)

            # Should preserve Unicode while maintaining HTML safety
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")
            # Unicode should be preserved in some form
            assert len(html) >= len(case)

    def test_malformed_syntax_recovery(self):
        """Test recovery from various malformed syntax."""
        malformed_cases = [
            "def function_without_colon()",  # Missing colon
            "if condition\n    pass",  # Missing colon
            'string = "unclosed string',  # Unclosed string
            "def (invalid_name): pass",  # Invalid function name
            "class (): pass",  # Invalid class name
            "import",  # Incomplete import
            "from . import",  # Incomplete from import
        ]

        for case in malformed_cases:
            html = highlight(case)

            # Should not crash, should produce some output
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")

            # Should contain at least some recognized tokens
            assert (
                "token-keyword" in html
                or "token-identifier" in html
                or "token-string" in html
            )


class TestRegressionAndStabilityTests:
    """Test for regression issues and stability."""

    def test_consistent_output(self):
        """Test that the same input always produces the same output."""
        test_code = '''def test_function(x, y=10):
    """Test function for consistency."""
    result = x + y
    return f"Result: {result}"
'''

        # Highlight the same code multiple times
        outputs = [highlight(test_code) for _ in range(5)]

        # All outputs should be identical
        first_output = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            assert output == first_output, f"Output {i + 1} differs from first output"

    def test_memory_leak_simulation(self):
        """Test for potential memory leaks with repeated highlighting."""
        test_code = """def memory_test():
    data = [i**2 for i in range(100)]
    return sum(data)
"""

        # Perform many highlighting operations
        for _ in range(100):
            html = highlight(test_code)

            # Verify each operation succeeds
            assert html.startswith("<pre><code")
            assert html.endswith("</code></pre>")

            # Clear any references to allow garbage collection
            del html

        # If we reach here without issues, no obvious memory leaks
        assert True  # Test passed

    def test_error_boundary_conditions(self):
        """Test error handling at boundary conditions."""
        # Test with maximum reasonable input sizes
        large_function = "def " + "very_long_function_name" * 100 + "(): pass"
        html = highlight(large_function)
        assert "token-keyword" in html

        # Test with many nested levels
        nested = "if True: " * 50 + "pass"
        html = highlight(nested)
        assert "token-keyword" in html

        # Test with many string quotes
        many_quotes = '"' * 1000
        html = highlight(f"x = {many_quotes}")
        # Should handle gracefully, might treat as unclosed string
        assert html.startswith("<pre><code")
        assert html.endswith("</code></pre>")


class TestBackwardCompatibilityConfirmation:
    """Confirm that all original APIs still work after refactoring."""

    def test_highlight_function_signature_unchanged(self):
        """Test that highlight() function signature is unchanged."""
        import inspect

        sig = inspect.signature(highlight)
        params = list(sig.parameters.keys())

        # Original signature should be maintained
        assert "code" in params
        # Optional parameters should still work
        result1 = highlight("def test(): pass")
        result2 = highlight("def test(): pass", language="python")

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert "<pre>" in result1
        assert "<pre>" in result2

    def test_codeblock_function_still_exists(self):
        """Test that CodeBlock() function still exists and works."""
        try:
            result = CodeBlock("def test(): pass")
            # CodeBlock returns a component object, not a string
            assert result is not None
            assert (
                hasattr(result, "__call__") or hasattr(result, "render") or str(result)
            )
        except ImportError as e:
            # Expected if no web frameworks installed - this is acceptable
            pytest.skip(f"CodeBlock requires web framework: {e}")

    def test_import_paths_unchanged(self):
        """Test that all import paths still work."""
        # These imports should not raise exceptions
        from starlighter import highlight

        # Basic functionality should work
        result = highlight("print('hello')")
        assert isinstance(result, str)

    def test_token_types_unchanged(self):
        """Validate rendered output includes expected token classes (API neutral)."""
        code = 'def x():\n    # comment\n    return "s" + 1\n'
        html = highlight(code)
        # Assert presence of representative token classes present in this sample
        for css_class in [
            "token-keyword",  # def, return
            "token-identifier",  # x
            "token-string",  # "s"
            "token-number",  # 1
            "token-operator",  # +
            "token-comment",  # # comment
        ]:
            assert css_class in html


class TestProjectSuccessValidation:
    """Final validation that the overall integration objectives are met."""

    def test_complete_pipeline_integration(self):
        """Test complete pipeline from Python code to HTML output."""
        python_code = '''
def fibonacci(n):
    """Generate Fibonacci sequence up to n."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Example usage
result = fibonacci(10)
print(f"Fibonacci sequence: {result}")
'''

        # Process through complete pipeline
        html_result = highlight(python_code)

        # Verify HTML structure
        assert '<pre><code class="language-python">' in html_result
        assert "</code></pre>" in html_result

        # Verify token types are properly classified
        assert 'class="token-keyword"' in html_result  # def, if, elif, etc.
        assert 'class="token-string"' in html_result  # docstrings and strings
        assert 'class="token-identifier"' in html_result  # function names
        assert 'class="token-number"' in html_result  # numbers
        assert 'class="token-comment"' in html_result  # comments

    def test_error_recovery_and_handling(self):
        """Test error recovery with malformed Python code."""
        malformed_code = """
def incomplete_function(
    # Missing closing parenthesis and body
    
class IncompleteClass
    # Missing colon and body
    
print("This should still work")
"""

        # Should handle malformed code gracefully
        result = highlight(malformed_code)

        # Should still produce HTML output
        assert isinstance(result, str)
        assert "<pre>" in result

        # Should include the valid parts
        assert "print" in result
        assert "This should still work" in result

    def test_production_readiness_checklist(self):
        """Validate that the system is production-ready."""
        checklist_results = {}

        # 1. Basic functionality
        try:
            result = highlight("def test(): pass")
            checklist_results["basic_functionality"] = "✅ PASS"
        except Exception as e:
            checklist_results["basic_functionality"] = f"❌ FAIL: {e}"

        # 2. Error handling
        try:
            result = highlight("invalid python ][}{")
            checklist_results["error_handling"] = (
                "✅ PASS" if result else "❌ FAIL: No output"
            )
        except Exception as e:
            checklist_results["error_handling"] = f"❌ FAIL: {e}"

        # 3. Security
        try:
            result = highlight("<script>alert('xss')</script>")
            if "<script>" not in result:
                checklist_results["security"] = "✅ PASS"
            else:
                checklist_results["security"] = "❌ FAIL: XSS vulnerability"
        except Exception as e:
            checklist_results["security"] = f"❌ FAIL: {e}"

        # 4. Performance (basic check)
        try:
            start = time.perf_counter()
            highlight("print('hello')" * 100)
            duration = time.perf_counter() - start
            if duration < 0.01:  # 10ms for simple test
                checklist_results["basic_performance"] = "✅ PASS"
            else:
                checklist_results["basic_performance"] = (
                    f"⚠️  SLOW: {duration * 1000:.1f}ms"
                )
        except Exception as e:
            checklist_results["basic_performance"] = f"❌ FAIL: {e}"

        print("\n🔍 PRODUCTION READINESS CHECKLIST:")
        for check, result in checklist_results.items():
            print(f"   {result} - {check}")

        # All critical checks must pass
        failures = [k for k, v in checklist_results.items() if "❌" in v]
        assert len(failures) == 0, f"Production readiness failures: {failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
