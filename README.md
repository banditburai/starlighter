# Starlighter

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![PyPI Version](https://img.shields.io/pypi/v/starlighter.svg)](https://pypi.org/project/starlighter/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-red.svg)]()

Server-side Python syntax highlighter with zero dependencies. Built for [StarHTML](https://github.com/banditburai/starHTML) with Datastar-aware tokenization, and compatible with FastHTML.

## Features

- **Fast**: ~1.5ms P99 for 200-line files, ~3ms for 500-line files
- **XSS-safe**: All output is entity-encoded
- **Zero dependencies**: Pure Python standard library
- **17 built-in themes**: GitHub, VS Code, Monokai, Catppuccin, Dracula, Nord, Solarized, One Dark, Xcode (dark + light variants)
- **StarHTML-aware**: Distinct token classes for HTML elements, Datastar attributes, signals, and CSS class/style strings
- **FastHTML compatible**: Works with both StarHTML and FastHTML projects

## Installation

```bash
# Using uv (recommended)
uv add starlighter

# Using pip
pip install starlighter
```

## Usage

### Basic highlighting

```python
from starlighter import highlight

code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))'''

html = highlight(code)
```

### StarHTML-aware highlighting

The lexer recognizes StarHTML elements, Datastar attributes, and signal references, giving each a distinct token class:

```python
from starlighter import highlight

code = '''Button(
    "Increment",
    data_on_click=count.add(1),
    cls="px-4 py-2 bg-blue-500"
)'''

html = highlight(code)
# Button  -> token-starhtml-element
# data_on_click -> token-datastar-attr
# cls string    -> token-css-class
```

You can also register custom elements, attributes, and signal names:

```python
html = highlight(
    code,
    elements={"MyWidget", "AppShell"},
    attrs={"custom_bind"},
    signals={"count", "message"},
)
```

### With StarHTML

```python
from starhtml import *
from starlighter import CodeBlock, StarlighterStyles

app, rt = star_app(
    hdrs=(
        StarlighterStyles("github-dark", "monokai", "dracula"),
    )
)

@rt("/")
def home():
    sample = '''def greet(name: str) -> str:
    return f"Hello, {name}!"'''

    return Div(
        H1("Code Preview"),
        CodeBlock(sample, theme="github-dark"),
        cls="container"
    )

serve()
```

### With FastHTML

```python
from fasthtml.common import *
from starlighter import CodeBlock, StarlighterStyles

app, rt = fast_app(
    pico=False,
    hdrs=(StarlighterStyles("vscode"),)
)

@rt("/")
def get():
    return Div(
        CodeBlock('print("hello")', theme="vscode"),
        cls="container"
    )

serve()
```

## Themes

17 themes are available:

| Dark | Light |
|------|-------|
| `github-dark` (default) | `github-light` |
| `vscode-dark` / `vscode` | `vscode-light` |
| `monokai` | `catppuccin-latte` |
| `catppuccin-mocha` | `nord-light` |
| `dracula` | `solarized-light` |
| `one-dark` | `xcode-light` |
| `nord-dark` | |
| `solarized-dark` | |
| `xcode-dark` | |

```python
from starlighter import get_theme_css

# Get raw CSS for a theme
css = get_theme_css("monokai")

# Use StarlighterStyles to load multiple themes at once
StarlighterStyles("github-dark", "monokai", "dracula")

# Auto-switch between themes based on prefers-color-scheme
StarlighterStyles("github-dark", "github-light", auto_switch=True)
```

## CSS Token Classes

| Class | Tokens |
|-------|--------|
| `.token-keyword` | `def`, `class`, `if`, `for`, `return`, ... |
| `.token-string` | `"text"`, `'text'`, `f"..."` |
| `.token-comment` | `# ...`, docstrings |
| `.token-number` | `42`, `3.14`, `0xFF` |
| `.token-operator` | `+`, `-`, `==`, `!=` |
| `.token-identifier` | Variable and function names |
| `.token-builtin` | `print`, `len`, `str`, `range`, ... |
| `.token-decorator` | `@app.route`, `@property` |
| `.token-punctuation` | `()`, `[]`, `{}`, `,`, `:` |
| `.token-starhtml-element` | `Div`, `Button`, `Input`, ... |
| `.token-datastar-attr` | `data_on_click`, `data_bind`, ... |
| `.token-signal` | Signal variable references |
| `.token-css-class` | Strings passed to `cls=`, `data_class=` |
| `.token-css-style` | Strings passed to `style=`, `data_style=` |

## API

```python
from starlighter import highlight, CodeBlock, StarlighterStyles, get_theme_css

highlight(
    code: str,
    language: str | None = "python",
    elements: set[str] | None = None,
    attrs: set[str] | None = None,
    signals: set[str] | None = None,
) -> str

CodeBlock(code: str, theme: str = "github-dark", **kwargs) -> Component

StarlighterStyles(*themes: str, auto_switch: bool = False, **kwargs) -> Style

get_theme_css(theme: str = "github-dark") -> str
```

## Security

All input is HTML-entity-encoded before output. Script injection via code input is not possible:

```python
code = 'return "</pre><script>alert(1)</script><pre>"'
html = highlight(code)
# Scripts are escaped: &lt;script&gt;alert(1)&lt;/script&gt;
```

## Development

```bash
git clone https://github.com/banditburai/starlighter.git
cd starlighter
uv sync --all-extras
uv run pytest tests/ -v --cov=starlighter
uv run ruff check && uv run ruff format --check
```

## Links

- [StarHTML](https://github.com/banditburai/starHTML)
- [PyPI](https://pypi.org/project/starlighter/)
- [GitHub](https://github.com/banditburai/starlighter)
