"""Server-side Python syntax highlighting. Fast, safe, zero deps."""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("starlighter")
except (ImportError, Exception):
    __version__ = "0.1.0"


from .parser import PythonLexer
from .themes import StarlighterStyles, get_theme_css, THEMES


def highlight(
    code: str,
    language: str | None = "python",
    elements: set[str] | None = None,
    attrs: set[str] | None = None,
    signals: set[str] | None = None,
) -> str:
    """Return HTML for Python code. Only Python; language is a CSS hint.

    Args:
        code: Python source code to highlight
        language: CSS class hint (default: "python")
        elements: Custom HTML element names (e.g., {"MyComponent", "CustomDiv"})
        attrs: Custom attribute names (e.g., {"custom_attr", "my_data_bind"})
        signals: Signal variable names (e.g., {"count", "message"})

    Example:
        >>> highlight("Div(cls='test')", elements={"Div"})
        >>> highlight("Button(on_click=count)", attrs={"on_click"}, signals={"count"})
    """
    if code is None:
        raise ValueError("Expected string, got None")
    if not isinstance(code, str):
        raise ValueError(f"Expected string, got {type(code).__name__}")
    if not code:
        return '<pre><code class="language-python"></code></pre>'
    try:
        return PythonLexer(
            code, elements=elements, attrs=attrs, signals=signals
        ).highlight_streaming()
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Unexpected error during highlighting: {str(e)}")


def CodeBlock(code: str, theme: str = "github-dark", **kwargs):
    """Return a FastHTML/StarHTML component with highlighted code."""
    highlighted_html = highlight(code)

    theme_class = f"theme-{theme}" if theme != "github-dark" else ""
    full_class = f"code-container {theme_class}".strip()

    def _as_component(html: str, cls: str, **kw):
        try:
            from starhtml.html import ft_datastar

            try:
                from starhtml import NotStr
            except ImportError:
                from fastcore.xml import NotStr
            return ft_datastar("div", NotStr(html), cls=cls, **kw)
        except ImportError:
            try:
                from fasthtml.common import ft_hx, NotStr
            except ImportError:
                raise ImportError("CodeBlock requires FastHTML or StarHTML.")
            return ft_hx("div", NotStr(html), cls=cls, **kw)

    return _as_component(highlighted_html, full_class, **kwargs)


__all__ = [
    "__version__",
    "highlight",
    "CodeBlock",
    "StarlighterStyles",
    "get_theme_css",
    "THEMES",
]
