"""Built-in CSS themes for Starlighter; concise APIs and minimal comments."""

# Modern CSS Variables-based theming
BASE_CSS_WITH_VARS = """
/* Base syntax highlighting styles with CSS variables */
.code-container {
    background: var(--code-bg);
    color: var(--code-color);
    border: 1px solid var(--code-border);
    border-radius: 8px;
    padding: 20px;
    overflow-x: auto;
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

/* Token colors using CSS variables */
.token-keyword { color: var(--token-keyword); }
.token-string { color: var(--token-string); }
.token-comment { color: var(--token-comment); font-style: italic; }
.token-number { color: var(--token-number); }
.token-operator { color: var(--token-operator); }
.token-identifier { color: var(--token-identifier); }
.token-builtin { color: var(--token-builtin); }
.token-decorator { color: var(--token-decorator); }
.token-punctuation { color: var(--token-punctuation); }

/* Scrollbar styling */
.code-container::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

.code-container::-webkit-scrollbar-track {
    background: var(--scrollbar-track);
}

.code-container::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 4px;
}

.code-container::-webkit-scrollbar-thumb:hover {
    background: var(--scrollbar-thumb-hover);
}
"""

# Legacy BASE_CSS for backward compatibility
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

# CSS Variables theme definitions
THEME_VARS = {
    "github-dark": {
        "--code-bg": "#0d1117",
        "--code-color": "#c9d1d9",
        "--code-border": "#30363d",
        "--token-keyword": "#ff7b72",
        "--token-string": "#a5d6ff",
        "--token-comment": "#8b949e",
        "--token-number": "#79c0ff",
        "--token-operator": "#c9d1d9",
        "--token-identifier": "#d2a8ff",
        "--token-builtin": "#ffa657",
        "--token-decorator": "#d2a8ff",
        "--token-punctuation": "#c9d1d9",
        "--scrollbar-track": "#0d1117",
        "--scrollbar-thumb": "#30363d",
        "--scrollbar-thumb-hover": "#484f58",
    },
    "light": {
        "--code-bg": "#ffffff",
        "--code-color": "#333333",
        "--code-border": "#e1e8ed",
        "--token-keyword": "#0000ff",
        "--token-string": "#a31515",
        "--token-comment": "#008000",
        "--token-number": "#098658",
        "--token-operator": "#000000",
        "--token-identifier": "#001080",
        "--token-builtin": "#267f99",
        "--token-decorator": "#795e26",
        "--token-punctuation": "#000000",
        "--scrollbar-track": "#f6f8fa",
        "--scrollbar-thumb": "#d1d5da",
        "--scrollbar-thumb-hover": "#a8b1bb",
    },
    "vscode": {
        "--code-bg": "#1e1e1e",
        "--code-color": "#d4d4d4",
        "--code-border": "#4a5568",
        "--token-keyword": "#569cd6",
        "--token-string": "#ce9178",
        "--token-comment": "#6a9955",
        "--token-number": "#b5cea8",
        "--token-operator": "#d4d4d4",
        "--token-identifier": "#9cdcfe",
        "--token-builtin": "#4ec9b0",
        "--token-decorator": "#dcdcaa",
        "--token-punctuation": "#d4d4d4",
        "--scrollbar-track": "#1e1e1e",
        "--scrollbar-thumb": "#4a5568",
        "--scrollbar-thumb-hover": "#718096",
    },
    "monokai": {
        "--code-bg": "#272822",
        "--code-color": "#f8f8f2",
        "--code-border": "#4a5568",
        "--token-keyword": "#f92672",
        "--token-string": "#e6db74",
        "--token-comment": "#75715e",
        "--token-number": "#ae81ff",
        "--token-operator": "#f8f8f2",
        "--token-identifier": "#a6e22e",
        "--token-builtin": "#66d9ef",
        "--token-decorator": "#f92672",
        "--token-punctuation": "#f8f8f2",
        "--scrollbar-track": "#272822",
        "--scrollbar-thumb": "#4a5568",
        "--scrollbar-thumb-hover": "#718096",
    },
    "dracula": {
        "--code-bg": "#282a36",
        "--code-color": "#f8f8f2",
        "--code-border": "#44475a",
        "--token-keyword": "#ff79c6",
        "--token-string": "#f1fa8c",
        "--token-comment": "#6272a4",
        "--token-number": "#bd93f9",
        "--token-operator": "#f8f8f2",
        "--token-identifier": "#50fa7b",
        "--token-builtin": "#8be9fd",
        "--token-decorator": "#ff79c6",
        "--token-punctuation": "#f8f8f2",
        "--scrollbar-track": "#282a36",
        "--scrollbar-thumb": "#44475a",
        "--scrollbar-thumb-hover": "#6272a4",
    },
    "catppuccin": {
        "--code-bg": "#1e1e2e",
        "--code-color": "#cdd6f4",
        "--code-border": "#45475a",
        "--token-keyword": "#cba6f7",
        "--token-string": "#a6e3a1",
        "--token-comment": "#6c7086",
        "--token-number": "#fab387",
        "--token-operator": "#cdd6f4",
        "--token-identifier": "#89dceb",
        "--token-builtin": "#f9e2af",
        "--token-decorator": "#f5c2e7",
        "--token-punctuation": "#cdd6f4",
        "--scrollbar-track": "#1e1e2e",
        "--scrollbar-thumb": "#45475a",
        "--scrollbar-thumb-hover": "#6c7086",
    },
}


def _generate_css_vars_string(vars_dict):
    """Generate CSS variable declarations from a dictionary."""
    return "\n    ".join(f"{key}: {value};" for key, value in vars_dict.items())


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


def StarlighterStyles(
    *themes, auto_switch: bool = False, use_vars: bool = True, **kwargs
):
    """
    Style element with base + requested themes.

    Args:
        *themes: Theme names to include. Defaults to ("github-dark",)
        auto_switch: Enable automatic theme switching based on system preference
        use_vars: Use CSS variables for theming (recommended for auto_switch)
        **kwargs: Additional arguments passed to Style element
    """
    if not themes:
        themes = ("github-dark",)

    # Use CSS variables approach when auto_switch is enabled or use_vars is True
    if auto_switch or use_vars:
        css_parts = [BASE_CSS_WITH_VARS]

        # Determine default theme and optional light theme for auto-switching
        default_theme = themes[0] if themes else "github-dark"
        light_theme = "light" if "light" in themes else None

        # Set default theme variables (dark theme by default)
        if default_theme in THEME_VARS:
            css_parts.append(f"""
/* Default theme: {default_theme} */
:root {{
    {_generate_css_vars_string(THEME_VARS[default_theme])}
}}""")

        # Add auto-switch support if requested
        if auto_switch and light_theme and light_theme in THEME_VARS:
            # Light theme for system preference
            css_parts.append(f"""
/* Auto-switch to light theme based on system preference */
@media (prefers-color-scheme: light) {{
    :root {{
        {_generate_css_vars_string(THEME_VARS[light_theme])}
    }}
}}""")

            # Manual theme overrides via data-theme attribute
            css_parts.append(f"""
/* Manual theme override: dark */
[data-theme="dark"] {{
    {_generate_css_vars_string(THEME_VARS[default_theme])}
}}

/* Manual theme override: light */
[data-theme="light"] {{
    {_generate_css_vars_string(THEME_VARS[light_theme])}
}}""")

        # Add theme-specific class overrides for backward compatibility
        for theme in themes:
            if theme in THEME_VARS:
                css_parts.append(f"""
/* Theme class override: {theme} */
.theme-{theme} {{
    {_generate_css_vars_string(THEME_VARS[theme])}
}}""")

    else:
        # Legacy approach: use class-based themes
        css_parts = [BASE_CSS]

        # Add requested themes
        for theme in themes:
            if theme in THEME_CSS_MAP:
                css_parts.append(THEME_CSS_MAP[theme])

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
