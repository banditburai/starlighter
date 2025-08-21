"""CSS themes for Starlighter using CSS variables."""

BASE_CSS = """
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

.token-keyword { color: var(--token-keyword); }
.token-string { color: var(--token-string); }
.token-comment { color: var(--token-comment); font-style: italic; }
.token-number { color: var(--token-number); }
.token-operator { color: var(--token-operator); }
.token-identifier { color: var(--token-identifier); }
.token-builtin { color: var(--token-builtin); }
.token-decorator { color: var(--token-decorator); }
.token-punctuation { color: var(--token-punctuation); }

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

THEMES = {
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


def css_vars(theme_dict):
    """Convert theme dictionary to CSS variable declarations."""
    return "\n    ".join(f"{k}: {v};" for k, v in theme_dict.items())


def get_theme_css(theme="github-dark"):
    """Get CSS with theme variables set."""
    if theme not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {list(THEMES.keys())}"
        )
    return f"{BASE_CSS}\n\n:root {{\n    {css_vars(THEMES[theme])}\n}}"


def StarlighterStyles(*themes, auto_switch=False, **kwargs):
    """Generate a Style element with theme CSS."""
    themes = themes or ("github-dark",)
    css_parts = [BASE_CSS]

    # Default theme
    default_theme = themes[0]
    if default_theme in THEMES:
        css_parts.append(f":root {{\n    {css_vars(THEMES[default_theme])}\n}}")

    # Auto-switching between dark and light
    if auto_switch and "light" in themes:
        css_parts.append(f"""
@media (prefers-color-scheme: light) {{
    :root {{
        {css_vars(THEMES["light"])}
    }}
}}

/* Manual overrides via data-theme attribute */
[data-theme="dark"] {{
    {css_vars(THEMES[default_theme])}
}}

[data-theme="light"] {{
    {css_vars(THEMES["light"])}
}}

/* Manual overrides via class (Tailwind-style) */
.dark {{
    {css_vars(THEMES[default_theme])}
}}

.light {{
    {css_vars(THEMES["light"])}
}}

/* Higher specificity for nested class patterns */
:root.dark {{
    {css_vars(THEMES[default_theme])}
}}

:root.light {{
    {css_vars(THEMES["light"])}
}}

html.dark {{
    {css_vars(THEMES[default_theme])}
}}

html.light {{
    {css_vars(THEMES["light"])}
}}""")

    # Additional theme classes for manual switching
    for theme in themes:
        if theme in THEMES and theme != default_theme:
            css_parts.append(f".theme-{theme} {{\n    {css_vars(THEMES[theme])}\n}}")

    try:
        from starhtml.tags import Style
    except ImportError:
        try:
            from fasthtml.common import Style
        except ImportError:
            return "\n".join(css_parts)

    return Style("\n".join(css_parts), **kwargs)


__all__ = ["BASE_CSS", "THEMES", "get_theme_css", "StarlighterStyles"]
