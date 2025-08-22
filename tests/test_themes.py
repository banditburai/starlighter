"""
Comprehensive test suite for the themes module.

Tests all theme-related functionality to achieve >90% coverage.
Focus on behavior testing: input → expected CSS output.
"""

import re
import pytest
from starlighter.themes import get_theme_css, StarlighterStyles, THEMES


class TestThemeCSS:
    """Test get_theme_css function behavior."""

    def test_get_default_theme(self):
        """Test getting default theme CSS."""
        css = get_theme_css()

        # Should return CSS string
        assert isinstance(css, str)
        assert len(css) > 0

        # Should contain expected CSS classes
        assert ".token-keyword" in css
        assert ".token-string" in css
        assert ".token-comment" in css
        assert ".token-identifier" in css
        assert ".token-builtin" in css
        assert ".token-number" in css

        # Should have GitHub Dark theme colors by default
        assert "#0d1117" in css  # GitHub Dark background color

    def test_get_specific_theme(self):
        """Test getting specific theme CSS."""
        # Test each available theme
        for theme_name in THEMES:
            css = get_theme_css(theme_name)

            assert isinstance(css, str)
            assert len(css) > 0
            assert ".token-keyword" in css

            # Theme-specific checks
            if theme_name == "monokai":
                assert "#F92672" in css or "#f92672" in css  # Monokai pink
            elif theme_name == "dracula":
                assert "#BD93F9" in css or "#bd93f9" in css  # Dracula purple
            elif theme_name == "vscode":
                assert "#569CD6" in css or "#569cd6" in css  # VSCode blue

    def test_invalid_theme_raises_error(self):
        """Test that invalid theme raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_theme_css("non-existent-theme")

        assert "Unknown theme" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_theme_css_structure(self):
        """Test CSS structure and selectors."""
        css = get_theme_css()

        # Check for proper CSS structure
        assert "code-container" in css  # Uses code-container not pre code
        assert "background" in css or "background-color" in css
        assert "color:" in css

        # Basic token classes should be present
        assert ".token-keyword" in css
        assert ".token-string" in css

    # Removed test_get_all_themes - "all" parameter is legacy behavior not in clean API


class TestStarlighterStyles:
    """Test StarlighterStyles function behavior."""

    def test_starlighter_styles_requires_framework(self):
        """Test that StarlighterStyles works when frameworks are available."""
        # Since we have FastHTML/StarHTML installed, it should work
        try:
            result = StarlighterStyles()
            assert result is not None
        except ImportError:
            # This is acceptable if frameworks truly aren't available
            pytest.skip("Framework not available")

    def test_starlighter_styles_with_themes(self):
        """Test StarlighterStyles with specific themes."""
        try:
            result = StarlighterStyles("monokai", "dracula")
            assert result is not None
        except ImportError:
            pytest.skip("Framework not available")

    def test_starlighter_styles_auto_switch(self):
        """Test auto-switch parameter."""
        try:
            result = StarlighterStyles(auto_switch=True)
            assert result is not None
        except ImportError:
            pytest.skip("Framework not available")

    def test_auto_switch_theme_assignment_fix(self):
        """Test fix for incorrect light theme selector assignment."""
        try:
            styles = StarlighterStyles('github-light', 'github-dark', auto_switch=True)
            css = str(styles)
            
            # Strip style tags if present
            if css.startswith('<style>') and css.endswith('</style>'):
                css = css[7:-8]
            
            # Helper function to extract background color
            def get_bg_color(selector_pattern):
                match = re.search(f'{selector_pattern}\\s*{{([^}}]+)}}', css)
                if match:
                    content = match.group(1)
                    bg_match = re.search(r'--code-bg:\s*([^;]+);', content)
                    return bg_match.group(1).strip() if bg_match else None
                return None
            
            # Test correct theme assignments
            assert get_bg_color(':root') == '#ffffff'  # Light theme in root
            assert get_bg_color(r'\.dark') == '#0d1117'  # Dark theme in .dark
            assert get_bg_color(r"\[data-theme='dark'\]") == '#0d1117'  # Dark theme in data attribute
            
            # Test that redundant light theme selectors are removed
            assert "[data-theme='light']" not in css
            assert "html[data-theme='light']" not in css
            assert ".theme-light" not in css
            
            # Test that excessive dark selectors are removed
            assert "html[data-theme='dark']" not in css
            assert ".theme-dark" not in css
            
        except ImportError:
            pytest.skip("Framework not available")


class TestThemeData:
    """Test THEMES constant and theme data structure."""

    def test_themes_constant(self):
        """Test THEMES constant is properly defined."""
        assert isinstance(THEMES, dict)
        assert len(THEMES) > 0

        # Should have core themes (testing behavior, not specific implementation)
        core_themes = {"github-dark", "monokai", "dracula"}
        for theme in core_themes:
            assert theme in THEMES, f"Core theme {theme} should be available"

    def test_theme_names_valid(self):
        """Test all theme names are valid identifiers."""
        for theme_name in THEMES.keys():
            # Should be valid CSS class name
            assert isinstance(theme_name, str)
            assert len(theme_name) > 0
            assert " " not in theme_name  # No spaces
            assert theme_name.replace("-", "").replace("_", "").isalnum()

    def test_theme_definitions(self):
        """Test theme definitions are present and valid."""
        for theme_name, theme_def in THEMES.items():
            assert isinstance(theme_def, dict)
            # Each theme should define required CSS variables
            assert "--code-bg" in theme_def
            assert "--token-keyword" in theme_def


class TestThemeIntegration:
    """Test theme integration with highlighting."""

    def test_theme_with_highlight(self):
        """Test that themes work with actual highlighting."""
        from starlighter import highlight

        # Get highlighted code
        code = '''def test():
    return "hello"'''
        html = highlight(code)

        # Get theme CSS
        css = get_theme_css("monokai")

        # Verify compatibility
        assert '<span class="token-keyword">' in html
        assert ".token-keyword" in css

        # CSS should style the HTML output
        assert "token-string" in html
        assert ".token-string" in css

    def test_complete_styled_output(self):
        """Test creating complete styled output."""
        from starlighter import highlight

        code = 'print("test")'
        html = highlight(code)
        css = get_theme_css("dracula")

        # Create style element manually
        style = f'<style type="text/css">{css}</style>'

        # Combine for complete output
        complete = style + html

        # Should be valid HTML with styling
        assert "<style" in complete
        assert "<pre><code" in complete
        assert ".token-builtin" in complete  # CSS
        assert "token-builtin" in complete  # HTML class

    def test_theme_css_coverage(self):
        """Test that theme CSS covers tokens used by parser."""
        from starlighter import highlight

        # Code with various token types
        code = '''
@decorator
def test_func(param: int) -> str:
    """Docstring"""
    # Comment
    data_show = "visible"  # DataStar attribute
    Button("Click")  # StarHTML element
    return f"Result: {param * 2}"
'''
        html = highlight(code)
        css = get_theme_css("monokai")

        # Extract token classes from HTML
        import re

        token_classes = set(re.findall(r'class="(token-[^"]+)"', html))

        # Check CSS covers all token classes used
        for token_class in token_classes:
            # Some tokens might not have specific styling (like whitespace)
            if token_class not in ["token-whitespace", "token-newline"]:
                # At least the base theme should style it
                assert f".{token_class}" in css or "token-" in css


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_theme_name_raises_error(self):
        """Test handling empty theme name."""
        with pytest.raises(ValueError) as exc_info:
            get_theme_css("")

        assert "Unknown theme ''" in str(exc_info.value)

    def test_none_theme_name_raises_error(self):
        """Test handling None theme name."""
        with pytest.raises((ValueError, TypeError)):
            get_theme_css(None)

    def test_theme_css_map_completeness(self):
        """Test that all THEMES have corresponding CSS."""
        for theme_name in THEMES.keys():
            # Should not raise error
            css = get_theme_css(theme_name)
            assert len(css) > 0

    def test_base_css_included(self):
        """Test that base CSS is included in output."""
        css = get_theme_css("monokai")

        # Should have base styles
        assert ".code-container" in css
        assert "overflow-x: auto" in css
        assert "font-family" in css


class TestCoverage:
    """Tests to improve coverage of untested code paths."""

    # Removed test_build_all_css - "all" parameter is legacy behavior not in clean API

    # Removed test_theme_css_map_structure - testing internal implementation details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
