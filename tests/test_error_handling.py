"""
Unit tests for simplified starlighter error handling system.

Tests the streamlined error classes for input validation and parsing errors.
"""

import pytest
from starlighter import highlight


class TestSimplifiedErrorSystem:
    """Test the simplified error class hierarchy."""

    def test_input_error_creation(self):
        """Test ValueError creation for invalid input."""
        error = ValueError("Invalid input")
        assert str(error) == "Invalid input"
        assert isinstance(error, ValueError)

    def test_parse_error_creation(self):
        """Test RuntimeError creation for parse failures."""
        error = RuntimeError("Parse failed")
        assert str(error) == "Parse failed"
        assert isinstance(error, Exception)

    def test_render_error_creation(self):
        """Test RuntimeError creation for render failures."""
        error = RuntimeError("Render failed")
        assert str(error) == "Render failed"
        assert isinstance(error, Exception)


class TestInputValidation:
    """Test input validation errors in the highlight function."""

    def test_none_input_raises_input_error(self):
        """Test that None input raises InputError."""
        with pytest.raises(ValueError) as exc_info:
            highlight(None)
        assert "Expected string, got None" in str(exc_info.value)

    def test_wrong_type_input_raises_input_error(self):
        """Test that wrong type input raises InputError."""
        with pytest.raises(ValueError) as exc_info:
            highlight(123)
        assert "Expected string, got int" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            highlight([])
        assert "Expected string, got list" in str(exc_info.value)

    def test_empty_string_works(self):
        """Test that empty string works without error."""
        result = highlight("")
        assert result == '<pre><code class="language-python"></code></pre>'

    def test_valid_string_works(self):
        """Test that valid string works without error."""
        result = highlight("def test(): pass")
        assert '<span class="token-keyword">def</span>' in result


if __name__ == "__main__":
    pytest.main([__file__])
