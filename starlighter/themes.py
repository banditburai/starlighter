"""CSS themes for Starlighter."""

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
.token-css-class { color: var(--token-css-class); font-weight: 500; }
.token-css-style { color: var(--token-css-style); font-style: italic; }
.token-comment { color: var(--token-comment); font-style: italic; }
.token-number { color: var(--token-number); }
.token-operator { color: var(--token-operator); }
.token-identifier { color: var(--token-identifier); }
.token-builtin { color: var(--token-builtin); }
.token-decorator { color: var(--token-decorator); }
.token-punctuation { color: var(--token-punctuation); }
.token-starhtml-element { color: var(--token-starhtml-element); font-weight: bold; }
.token-datastar-attr { color: var(--token-datastar-attr); }
.token-signal { color: var(--token-signal); font-weight: bold; }

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
        # Standard tokens - original GitHub dark colors
        "--token-keyword": "#ff7b72",
        "--token-string": "#a5d6ff",
        "--token-comment": "#8b949e",
        "--token-number": "#79c0ff",
        "--token-operator": "#c9d1d9",
        "--token-identifier": "#d2a8ff",
        "--token-builtin": "#ffa657",
        "--token-decorator": "#d2a8ff",
        "--token-punctuation": "#c9d1d9",
        # StarHTML tokens - Catppuccin Mocha pastel palette
        "--token-starhtml-element": "#a6e3a1",  # green
        "--token-datastar-attr": "#cba6f7",  # mauve
        "--token-css-class": "#94e2d5",  # teal
        "--token-css-style": "#f38ba8",  # pink
        "--token-signal": "#fab387",  # peach
        "--scrollbar-track": "#0d1117",
        "--scrollbar-thumb": "#30363d",
        "--scrollbar-thumb-hover": "#484f58",
    },
    "github-light": {
        "--code-bg": "#ffffff",
        "--code-color": "#24292e",
        "--code-border": "#e1e4e8",
        # Standard tokens - original GitHub light colors
        "--token-keyword": "#d73a49",
        "--token-string": "#032f62",
        "--token-comment": "#6e7781",
        "--token-number": "#005cc5",
        "--token-operator": "#24292e",
        "--token-identifier": "#6f42c1",
        "--token-builtin": "#e36209",
        "--token-decorator": "#6f42c1",
        "--token-punctuation": "#24292e",
        # StarHTML tokens - Catppuccin Latte pastel palette
        "--token-starhtml-element": "#40a02b",  # green
        "--token-datastar-attr": "#8839ef",  # mauve
        "--token-css-class": "#179299",  # teal
        "--token-css-style": "#d20f39",  # red
        "--token-signal": "#fe640b",  # peach
        "--scrollbar-track": "#ffffff",
        "--scrollbar-thumb": "#e1e4e8",
        "--scrollbar-thumb-hover": "#c8e1ff",
    },
    "default": {
        "--code-bg": "#ffffff",
        "--code-color": "#333333",
        "--code-border": "#e1e8ed",
        "--token-keyword": "#0000ff",
        "--token-string": "#a31515",
        "--token-css-class": "#e36209",
        "--token-css-style": "#9a5215",
        "--token-comment": "#008000",
        "--token-number": "#098658",
        "--token-operator": "#000000",
        "--token-identifier": "#001080",
        "--token-builtin": "#267f99",
        "--token-decorator": "#795e26",
        "--token-punctuation": "#000000",
        "--token-starhtml-element": "#098658",
        "--token-datastar-attr": "#af00db",
        "--token-signal": "#af00db",
        "--scrollbar-track": "#f6f8fa",
        "--scrollbar-thumb": "#d1d5da",
        "--scrollbar-thumb-hover": "#a8b1bb",
    },
    "vscode-dark": {
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
        "--token-starhtml-element": "#4ec9b0",
        "--token-datastar-attr": "#9cdcfe",
        "--token-signal": "#dcdcaa",
        "--scrollbar-track": "#1e1e1e",
        "--scrollbar-thumb": "#4a5568",
        "--scrollbar-thumb-hover": "#718096",
    },
    "vscode-light": {
        "--code-bg": "#ffffff",
        "--code-color": "#000000",
        "--code-border": "#e5e5e5",
        "--token-keyword": "#0000ff",
        "--token-string": "#a31515",
        "--token-comment": "#008000",
        "--token-number": "#098658",
        "--token-operator": "#000000",
        "--token-identifier": "#001080",
        "--token-builtin": "#267f99",
        "--token-decorator": "#795e26",
        "--token-punctuation": "#000000",
        "--token-starhtml-element": "#098658",
        "--token-datastar-attr": "#001080",
        "--token-signal": "#af00db",
        "--scrollbar-track": "#ffffff",
        "--scrollbar-thumb": "#e5e5e5",
        "--scrollbar-thumb-hover": "#c8c8c8",
    },
    "vscode": None,  # Alias for vscode-dark, set after dict creation
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
        "--token-starhtml-element": "#a6e22e",
        "--token-datastar-attr": "#66d9ef",
        "--token-signal": "#e6db74",
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
        "--token-starhtml-element": "#50fa7b",
        "--token-datastar-attr": "#8be9fd",
        "--token-signal": "#ffb86c",
        "--scrollbar-track": "#282a36",
        "--scrollbar-thumb": "#44475a",
        "--scrollbar-thumb-hover": "#6272a4",
    },
    "one-dark": {
        "--code-bg": "#282c34",
        "--code-color": "#abb2bf",
        "--code-border": "#3e4451",
        "--token-keyword": "#c678dd",
        "--token-string": "#98c379",
        "--token-comment": "#5c6370",
        "--token-number": "#d19a66",
        "--token-operator": "#abb2bf",
        "--token-identifier": "#61afef",
        "--token-builtin": "#e06c75",
        "--token-decorator": "#c678dd",
        "--token-punctuation": "#abb2bf",
        "--token-starhtml-element": "#98c379",
        "--token-datastar-attr": "#61afef",
        "--token-signal": "#e5c07b",
        "--scrollbar-track": "#282c34",
        "--scrollbar-thumb": "#3e4451",
        "--scrollbar-thumb-hover": "#5c6370",
    },
    "xcode-dark": {
        "--code-bg": "#1f1f24",
        "--code-color": "#ffffff",
        "--code-border": "#3a3a3f",
        "--token-keyword": "#ff7ab2",
        "--token-string": "#ff8170",
        "--token-comment": "#6c7986",
        "--token-number": "#d0bf69",
        "--token-operator": "#ffffff",
        "--token-identifier": "#6bdfff",
        "--token-builtin": "#4eb0cc",
        "--token-decorator": "#ff7ab2",
        "--token-punctuation": "#ffffff",
        "--token-starhtml-element": "#5cefa2",
        "--token-datastar-attr": "#6bdfff",
        "--token-signal": "#ffd666",
        "--scrollbar-track": "#1f1f24",
        "--scrollbar-thumb": "#3a3a3f",
        "--scrollbar-thumb-hover": "#5a5a5f",
    },
    "xcode-light": {
        "--code-bg": "#ffffff",
        "--code-color": "#000000",
        "--code-border": "#e5e5e5",
        "--token-keyword": "#ad3da4",
        "--token-string": "#d12f1b",
        "--token-comment": "#5d6c79",
        "--token-number": "#272ad8",
        "--token-operator": "#000000",
        "--token-identifier": "#3f6e75",
        "--token-builtin": "#23575c",
        "--token-decorator": "#ad3da4",
        "--token-punctuation": "#000000",
        "--token-starhtml-element": "#0a7a3a",
        "--token-datastar-attr": "#3f6e75",
        "--token-signal": "#d12f1b",
        "--scrollbar-track": "#ffffff",
        "--scrollbar-thumb": "#e5e5e5",
        "--scrollbar-thumb-hover": "#c8c8c8",
    },
    "catppuccin-mocha": {
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
        "--token-starhtml-element": "#a6e3a1",
        "--token-datastar-attr": "#89dceb",
        "--token-signal": "#f9e2af",
        "--scrollbar-track": "#1e1e2e",
        "--scrollbar-thumb": "#45475a",
        "--scrollbar-thumb-hover": "#6c7086",
    },
    "catppuccin-latte": {
        "--code-bg": "#eff1f5",
        "--code-color": "#4c4f69",
        "--code-border": "#bcc0cc",
        "--token-keyword": "#8839ef",
        "--token-string": "#40a02b",
        "--token-comment": "#6c6f85",
        "--token-number": "#fe640b",
        "--token-operator": "#4c4f69",
        "--token-identifier": "#179299",
        "--token-builtin": "#df8e1d",
        "--token-decorator": "#ea76cb",
        "--token-punctuation": "#4c4f69",
        "--token-starhtml-element": "#40a02b",
        "--token-datastar-attr": "#179299",
        "--token-signal": "#df8e1d",
        "--scrollbar-track": "#eff1f5",
        "--scrollbar-thumb": "#bcc0cc",
        "--scrollbar-thumb-hover": "#9ca0b0",
    },
    "nord-dark": {
        "--code-bg": "#2e3440",
        "--code-color": "#d8dee9",
        "--code-border": "#3b4252",
        "--token-keyword": "#81a1c1",
        "--token-string": "#a3be8c",
        "--token-comment": "#616e88",
        "--token-number": "#b48ead",
        "--token-operator": "#d8dee9",
        "--token-identifier": "#88c0d0",
        "--token-builtin": "#d08770",
        "--token-decorator": "#5e81ac",
        "--token-punctuation": "#d8dee9",
        "--token-starhtml-element": "#a3be8c",
        "--token-datastar-attr": "#88c0d0",
        "--token-signal": "#ebcb8b",
        "--scrollbar-track": "#2e3440",
        "--scrollbar-thumb": "#3b4252",
        "--scrollbar-thumb-hover": "#434c5e",
    },
    "nord-light": {
        "--code-bg": "#eceff4",
        "--code-color": "#2e3440",
        "--code-border": "#d8dee9",
        "--token-keyword": "#5e81ac",
        "--token-string": "#a3be8c",
        "--token-comment": "#a0a8cd",
        "--token-number": "#b48ead",
        "--token-operator": "#2e3440",
        "--token-identifier": "#88c0d0",
        "--token-builtin": "#d08770",
        "--token-decorator": "#81a1c1",
        "--token-punctuation": "#2e3440",
        "--token-starhtml-element": "#8fbcbb",
        "--token-datastar-attr": "#88c0d0",
        "--token-signal": "#d08770",
        "--scrollbar-track": "#eceff4",
        "--scrollbar-thumb": "#d8dee9",
        "--scrollbar-thumb-hover": "#c8d2e3",
    },
    "solarized-dark": {
        "--code-bg": "#002b36",
        "--code-color": "#839496",
        "--code-border": "#073642",
        "--token-keyword": "#268bd2",
        "--token-string": "#2aa198",
        "--token-comment": "#586e75",
        "--token-number": "#d33682",
        "--token-operator": "#839496",
        "--token-identifier": "#b58900",
        "--token-builtin": "#cb4b16",
        "--token-decorator": "#6c71c4",
        "--token-punctuation": "#839496",
        "--token-starhtml-element": "#859900",
        "--token-datastar-attr": "#2aa198",
        "--token-signal": "#b58900",
        "--scrollbar-track": "#002b36",
        "--scrollbar-thumb": "#073642",
        "--scrollbar-thumb-hover": "#586e75",
    },
    "solarized-light": {
        "--code-bg": "#fdf6e3",
        "--code-color": "#657b83",
        "--code-border": "#eee8d5",
        "--token-keyword": "#268bd2",
        "--token-string": "#2aa198",
        "--token-comment": "#93a1a1",
        "--token-number": "#d33682",
        "--token-operator": "#657b83",
        "--token-identifier": "#b58900",
        "--token-builtin": "#cb4b16",
        "--token-decorator": "#6c71c4",
        "--token-punctuation": "#657b83",
        "--token-starhtml-element": "#859900",
        "--token-datastar-attr": "#2aa198",
        "--token-signal": "#b58900",
        "--scrollbar-track": "#fdf6e3",
        "--scrollbar-thumb": "#eee8d5",
        "--scrollbar-thumb-hover": "#d7d0b8",
    },
}
THEMES["vscode"] = THEMES["vscode-dark"]

DARK_SELECTORS = (".dark", "[data-theme='dark']")


def _css_vars(theme_dict: dict[str, str]) -> str:
    return "\n    ".join(f"{k}: {v};" for k, v in theme_dict.items())


def _wrap_vars(selector: str, theme: str) -> str:
    return f"{selector} {{\n    {_css_vars(THEMES[theme])}\n}}"


def get_theme_css(theme="github-dark"):
    if theme not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {list(THEMES.keys())}"
        )
    return f"{BASE_CSS}\n\n{_wrap_vars(':root', theme)}"


def StarlighterStyles(*themes, auto_switch=False, **kwargs):
    themes = themes or ("github-dark",)

    for theme in themes:
        if theme not in THEMES:
            raise ValueError(
                f"Unknown theme '{theme}'. Available themes: {list(THEMES.keys())}"
            )

    css = [BASE_CSS, _wrap_vars(":root", themes[0])]

    for theme in themes[1:]:
        css.append(_wrap_vars(f".theme-{theme}", theme))

    if auto_switch and len(themes) > 1:
        css.extend(_wrap_vars(sel, themes[1]) for sel in DARK_SELECTORS)

    try:
        from starhtml.tags import Style
        from starhtml import NotStr
    except ImportError:
        try:
            from fasthtml.common import Style, NotStr
        except ImportError:
            return "\n\n".join(css)

    return Style(NotStr("\n\n".join(css)), **kwargs)


__all__ = ["BASE_CSS", "THEMES", "get_theme_css", "StarlighterStyles"]
