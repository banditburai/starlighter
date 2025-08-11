"""Built-in CSS themes for Starlighter; concise APIs and minimal comments."""

BASE_CSS = """
/* Base syntax highlighting styles */
.code-container {
    border-radius: 8px;
    padding: 20px;
    overflow-x: auto;
    border: 1px solid #4a5568;
    margin: 0;
    min-width: 0;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'SF Mono', monospace;
    font-size: 14px;
    line-height: 1.5;
}

.code-container pre {
    margin: 0;
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
    white-space: pre;
    overflow-x: auto;
    min-width: 0;
}

.code-container code {
    font-family: inherit;
    background: none;
    padding: 0;
    display: block;
    white-space: pre;
    overflow-x: auto;
}

/* Scrollbar styling */
.code-container::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

.code-container::-webkit-scrollbar-track {
    background: #2d3748;
}

.code-container::-webkit-scrollbar-thumb {
    background: #4a5568;
    border-radius: 4px;
}

.code-container::-webkit-scrollbar-thumb:hover {
    background: #718096;
}
"""

# VS Code Dark+ Theme
VSCODE_DARK_CSS = """
/* VS Code Dark+ Theme */
.theme-vscode .code-container,
.code-container.theme-vscode {
    background: #1e1e1e;
    color: #d4d4d4;
}

.theme-vscode .token-keyword,
.code-container.theme-vscode .token-keyword { color: #569cd6; }

.theme-vscode .token-string,
.code-container.theme-vscode .token-string { color: #ce9178; }

.theme-vscode .token-comment,
.code-container.theme-vscode .token-comment { color: #6a9955; font-style: italic; }

.theme-vscode .token-number,
.code-container.theme-vscode .token-number { color: #b5cea8; }

.theme-vscode .token-operator,
.code-container.theme-vscode .token-operator { color: #d4d4d4; }

.theme-vscode .token-identifier,
.code-container.theme-vscode .token-identifier { color: #9cdcfe; }

.theme-vscode .token-builtin,
.code-container.theme-vscode .token-builtin { color: #4ec9b0; }

.theme-vscode .token-decorator,
.code-container.theme-vscode .token-decorator { color: #dcdcaa; }

.theme-vscode .token-punctuation,
.code-container.theme-vscode .token-punctuation { color: #d4d4d4; }
"""

VSCODE_LIGHT_CSS = """
/* VS Code Light+ Theme */
.theme-light .code-container,
.code-container.theme-light {
    background: #ffffff !important;
    color: #333;
    border-color: #e1e8ed;
}

.theme-light .token-keyword,
.code-container.theme-light .token-keyword { color: #0000ff; }

.theme-light .token-string,
.code-container.theme-light .token-string { color: #a31515; }

.theme-light .token-comment,
.code-container.theme-light .token-comment { color: #008000; font-style: italic; }

.theme-light .token-number,
.code-container.theme-light .token-number { color: #098658; }

.theme-light .token-operator,
.code-container.theme-light .token-operator { color: #000000; }

.theme-light .token-identifier,
.code-container.theme-light .token-identifier { color: #001080; }

.theme-light .token-builtin,
.code-container.theme-light .token-builtin { color: #267f99; }

.theme-light .token-decorator,
.code-container.theme-light .token-decorator { color: #795e26; }

.theme-light .token-punctuation,
.code-container.theme-light .token-punctuation { color: #000000; }
"""

MONOKAI_CSS = """
/* Monokai Theme */
.theme-monokai .code-container,
.code-container.theme-monokai {
    background: #272822;
    color: #f8f8f2;
}

.theme-monokai .token-keyword,
.code-container.theme-monokai .token-keyword { color: #f92672; }

.theme-monokai .token-string,
.code-container.theme-monokai .token-string { color: #e6db74; }

.theme-monokai .token-comment,
.code-container.theme-monokai .token-comment { color: #75715e; font-style: italic; }

.theme-monokai .token-number,
.code-container.theme-monokai .token-number { color: #ae81ff; }

.theme-monokai .token-operator,
.code-container.theme-monokai .token-operator { color: #f8f8f2; }

.theme-monokai .token-identifier,
.code-container.theme-monokai .token-identifier { color: #a6e22e; }

.theme-monokai .token-builtin,
.code-container.theme-monokai .token-builtin { color: #66d9ef; }

.theme-monokai .token-decorator,
.code-container.theme-monokai .token-decorator { color: #f92672; }

.theme-monokai .token-punctuation,
.code-container.theme-monokai .token-punctuation { color: #f8f8f2; }
"""

# GitHub Dark Theme (default)
GITHUB_DARK_CSS = """
/* GitHub Dark Theme (default) */
.code-container {
    background: #0d1117;
    color: #c9d1d9;
}

.token-keyword { color: #ff7b72; }
.token-string { color: #a5d6ff; }
.token-comment { color: #8b949e; font-style: italic; }
.token-number { color: #79c0ff; }
.token-operator { color: #c9d1d9; }
.token-identifier { color: #d2a8ff; }
.token-builtin { color: #ffa657; }
.token-decorator { color: #d2a8ff; }
.token-punctuation { color: #c9d1d9; }
"""

DRACULA_CSS = """
/* Dracula Theme */
.theme-dracula .code-container,
.code-container.theme-dracula {
    background: #282a36;
    color: #f8f8f2;
}

.theme-dracula .token-keyword,
.code-container.theme-dracula .token-keyword { color: #ff79c6; }

.theme-dracula .token-string,
.code-container.theme-dracula .token-string { color: #f1fa8c; }

.theme-dracula .token-comment,
.code-container.theme-dracula .token-comment { color: #6272a4; font-style: italic; }

.theme-dracula .token-number,
.code-container.theme-dracula .token-number { color: #bd93f9; }

.theme-dracula .token-operator,
.code-container.theme-dracula .token-operator { color: #f8f8f2; }

.theme-dracula .token-identifier,
.code-container.theme-dracula .token-identifier { color: #50fa7b; }

.theme-dracula .token-builtin,
.code-container.theme-dracula .token-builtin { color: #8be9fd; }

.theme-dracula .token-decorator,
.code-container.theme-dracula .token-decorator { color: #ff79c6; }

.theme-dracula .token-punctuation,
.code-container.theme-dracula .token-punctuation { color: #f8f8f2; }
"""

CATPPUCCIN_CSS = """
/* Catppuccin Mocha Theme */
.theme-catppuccin .code-container,
.code-container.theme-catppuccin {
    background: #1e1e2e;
    color: #cdd6f4;
}

.theme-catppuccin .token-keyword,
.code-container.theme-catppuccin .token-keyword { color: #cba6f7; }

.theme-catppuccin .token-string,
.code-container.theme-catppuccin .token-string { color: #a6e3a1; }

.theme-catppuccin .token-comment,
.code-container.theme-catppuccin .token-comment { color: #6c7086; font-style: italic; }

.theme-catppuccin .token-number,
.code-container.theme-catppuccin .token-number { color: #fab387; }

.theme-catppuccin .token-operator,
.code-container.theme-catppuccin .token-operator { color: #89dceb; }

.theme-catppuccin .token-identifier,
.code-container.theme-catppuccin .token-identifier { color: #cdd6f4; }

.theme-catppuccin .token-builtin,
.code-container.theme-catppuccin .token-builtin { color: #f9e2af; }

.theme-catppuccin .token-decorator,
.code-container.theme-catppuccin .token-decorator { color: #f5c2e7; }

.theme-catppuccin .token-punctuation,
.code-container.theme-catppuccin .token-punctuation { color: #cdd6f4; }
"""

THEME_CSS_MAP = {
    "vscode": VSCODE_DARK_CSS,
    "light": VSCODE_LIGHT_CSS,
    "monokai": MONOKAI_CSS,
    "github-dark": GITHUB_DARK_CSS,
    "dracula": DRACULA_CSS,
    "catppuccin": CATPPUCCIN_CSS,
}


def _build_all_css() -> str:
    # Include BASE first so later theme blocks can override as needed
    return "\n\n".join([BASE_CSS] + list(THEME_CSS_MAP.values()))


ALL_THEMES_CSS = _build_all_css()

THEMES = {
    "vscode": "VS Code Dark+",
    "light": "VS Code Light+",
    "monokai": "Monokai",
    "github-dark": "GitHub Dark",
    "dracula": "Dracula",
    "catppuccin": "Catppuccin Mocha",
}


def get_theme_css(theme: str = "all") -> str:
    """Return CSS for a theme; 'all' returns all built-ins."""
    if theme == "all":
        return _build_all_css()
    try:
        return BASE_CSS + "\n\n" + THEME_CSS_MAP[theme]
    except KeyError:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {list(THEME_CSS_MAP.keys())}"
        )


def _style_element(css_content: str, **kwargs):
    """Create a <style> element for StarHTML/FastHTML"""
    try:
        from starhtml.tags import Style
    except ImportError:
        try:
            from fasthtml.common import Style
        except ImportError:
            raise ImportError(
                "StarlighterStyles requires FastHTML or StarHTML. Use get_theme_css()."
            )
    return Style(css_content, **kwargs)


def StarlighterStyles(*themes, auto_switch: bool = False, **kwargs):
    """Style element with base + requested themes. Optional dark/light auto-switch."""
    if not themes:
        themes = ("github-dark",)

    css_parts = [BASE_CSS]

    # Add requested themes
    for theme in themes:
        if theme in THEME_CSS_MAP:
            css_parts.append(THEME_CSS_MAP[theme])

    # Auto-switch: system preference and explicit data-theme hooks
    if auto_switch:
        dark_theme = themes[0] if themes else "vscode"
        light_theme = "light" if "light" in themes else None

        if dark_theme in THEME_CSS_MAP:
            dark_css = THEME_CSS_MAP[dark_theme]
            css_parts.append(
                f"""@media (prefers-color-scheme: dark) {{{dark_css}}}
[data-theme="dark"] .code-container {{{dark_css}}}"""
            )
        if light_theme and light_theme in THEME_CSS_MAP:
            light_css = THEME_CSS_MAP[light_theme]
            css_parts.append(
                f"""@media (prefers-color-scheme: light) {{{light_css}}}
[data-theme="light"] .code-container {{{light_css}}}"""
            )

    return _style_element("\n".join(css_parts), **kwargs)


__all__ = [
    "BASE_CSS",
    "VSCODE_DARK_CSS",
    "VSCODE_LIGHT_CSS",
    "MONOKAI_CSS",
    "GITHUB_DARK_CSS",
    "DRACULA_CSS",
    "CATPPUCCIN_CSS",
    "THEME_CSS_MAP",
    "THEMES",
    "get_theme_css",
    "StarlighterStyles",
]
