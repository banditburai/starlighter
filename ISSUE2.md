# Bug Report: StarlighterStyles auto_switch Still Broken in v0.1.2

## Summary

The `auto_switch` parameter in `StarlighterStyles` (v0.1.2) generates incorrect CSS that prevents theme switching from working. The issue is that theme CSS with class selectors is being placed directly inside media queries, which doesn't match any elements.

## Bug Location

**File:** `starlighter/themes.py`  
**Function:** `StarlighterStyles`  
**Lines:** 340-344 (and similar pattern for dark theme)

## Current Buggy Code (v0.1.2)

```python
# Lines 340-344
css_parts.append(f"""
/* Light theme for system preference */
@media (prefers-color-scheme: light) {{
    {light_css}
}}
```

Where `light_css` contains:
```css
/* VS Code Light+ Theme */
.theme-light .code-container,
.code-container.theme-light {
    background: #ffffff !important;
    color: #333;
    border-color: #e1e8ed;
}

.theme-light .token-keyword,
.code-container.theme-light .token-keyword { color: #0000ff; }
/* ... more rules with .theme-light selectors ... */
```

## The Problem

When this CSS is placed inside a media query, the resulting CSS looks like:

```css
@media (prefers-color-scheme: light) {
    .theme-light .code-container,
    .code-container.theme-light {
        background: #ffffff !important;
        /* ... */
    }
    .theme-light .token-keyword,
    .code-container.theme-light .token-keyword { color: #0000ff; }
}
```

**This doesn't work because:**
1. The media query applies based on system preference
2. But the CSS rules inside still require a `.theme-light` class to be present
3. No JavaScript is adding/removing `.theme-light` class based on system preference
4. Therefore, the styles never apply

## Expected Behavior

When `auto_switch=True` is used, the CSS should:
1. Apply dark theme by default (without requiring classes)
2. Apply light theme when `prefers-color-scheme: light` (without requiring classes)
3. Support manual override via `data-theme` attributes

## Root Cause

The theme CSS definitions (`VSCODE_LIGHT_CSS`, etc.) are designed to work with explicit class selectors (`.theme-light`), which is fine for manual theme switching. However, when these are placed inside media queries for automatic switching, the class requirement prevents them from working.

## Proposed Solution

### Option 1: Transform CSS When Using auto_switch

Create a helper function to strip class requirements when placing theme CSS inside media queries:

```python
def _transform_theme_css_for_media_query(theme_css: str, theme_name: str) -> str:
    """
    Transform theme CSS to work inside media queries by removing class requirements.
    
    Example:
        Input:  ".theme-light .code-container { background: #fff; }"
        Output: ".code-container { background: #fff; }"
    """
    import re
    
    # Remove .theme-{name} class requirements
    transformed = re.sub(
        rf'\.theme-{theme_name}\s+\.code-container,?\s*',
        '.code-container',
        theme_css
    )
    transformed = re.sub(
        rf'\.code-container\.theme-{theme_name}',
        '.code-container',
        transformed
    )
    transformed = re.sub(
        rf'\.theme-{theme_name}\s+\.token-',
        '.token-',
        transformed
    )
    transformed = re.sub(
        rf'\.code-container\.theme-{theme_name}\s+\.token-',
        '.code-container .token-',
        transformed
    )
    
    return transformed

# Then in StarlighterStyles:
if auto_switch:
    # ...
    if light_theme and light_theme in THEME_CSS_MAP:
        light_css = THEME_CSS_MAP[light_theme]
        # Transform for media query use
        light_css_for_media = _transform_theme_css_for_media_query(light_css, 'light')
        css_parts.append(f"""
/* Light theme for system preference */
@media (prefers-color-scheme: light) {{
    {light_css_for_media}
}}""")
```

### Option 2: Maintain Separate CSS Definitions

Create separate CSS definitions specifically for media query use:

```python
# Add to themes.py
VSCODE_LIGHT_CSS_NO_CLASS = """
/* VS Code Light+ Theme (no class selectors) */
.code-container {
    background: #ffffff !important;
    color: #333;
    border-color: #e1e8ed;
}

.token-keyword { color: #0000ff; }
.token-string { color: #a31515; }
.token-comment { color: #008000; font-style: italic; }
.token-number { color: #098658; }
.token-operator { color: #000000; }
.token-identifier { color: #001080; }
.token-builtin { color: #267f99; }
.token-decorator { color: #795e26; }
.token-punctuation { color: #000000; }
"""

THEME_CSS_MAP_NO_CLASS = {
    "light": VSCODE_LIGHT_CSS_NO_CLASS,
    # ... other themes without class selectors
}

# Use in StarlighterStyles:
if auto_switch:
    if light_theme in THEME_CSS_MAP_NO_CLASS:
        light_css_no_class = THEME_CSS_MAP_NO_CLASS[light_theme]
        css_parts.append(f"""
@media (prefers-color-scheme: light) {{
    {light_css_no_class}
}}""")
```

### Option 3: Use CSS Variables (Most Robust)

Define themes using CSS variables and switch variable values based on media queries:

```python
BASE_CSS_WITH_VARS = """
.code-container {
    background: var(--code-bg);
    color: var(--code-color);
    border-color: var(--code-border);
    /* ... */
}

.token-keyword { color: var(--token-keyword); }
.token-string { color: var(--token-string); }
/* ... more token rules ... */

/* Dark theme (default) */
:root {
    --code-bg: #0d1117;
    --code-color: #c9d1d9;
    --code-border: #4a5568;
    --token-keyword: #ff7b72;
    --token-string: #a5d6ff;
    /* ... */
}

/* Light theme via media query */
@media (prefers-color-scheme: light) {
    :root {
        --code-bg: #ffffff;
        --code-color: #333;
        --code-border: #e1e8ed;
        --token-keyword: #0000ff;
        --token-string: #a31515;
        /* ... */
    }
}

/* Manual theme override */
[data-theme="light"] {
    --code-bg: #ffffff;
    --code-color: #333;
    /* ... */
}
"""
```

## Testing Requirements

After implementing the fix, verify:

1. **Default behavior**: Code blocks use dark theme by default
2. **System preference**: Code blocks switch to light theme when `prefers-color-scheme: light`
3. **Manual override**: Code blocks respect `data-theme` attribute on parent elements
4. **No class dependency**: Themes work without requiring `.theme-light` or `.theme-dark` classes
5. **All tokens styled**: Verify all token types have appropriate colors in both themes

## Impact

This bug completely breaks the `auto_switch` feature, forcing users to implement workarounds with manual CSS. This defeats the purpose of having a clean, reusable syntax highlighting library.

## Recommendation

I recommend **Option 3 (CSS Variables)** as the most robust solution because:
1. It's the standard modern CSS approach for theming
2. It avoids string manipulation and regex transformations
3. It's easier to maintain and extend with new themes
4. It naturally supports both automatic and manual theme switching
5. It results in smaller CSS output (no duplication of rules)

The current implementation mixes two incompatible patterns (class-based themes and media queries), which is why it fails. A proper solution needs to choose one consistent approach.