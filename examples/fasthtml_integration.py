from fasthtml.common import *
from starlighter import CodeBlock, StarlighterStyles

# Create FastHTML app with Starlighter styles
app, rt = fast_app(
    pico=False,
    hdrs=(
        # Include all themes for the interactive editor demo
        StarlighterStyles(
            "vscode", "light", "monokai", "catppuccin", "github-dark", "dracula"
        ),
        Style("""
            /* Additional app-specific styles */
            body {
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f5f5f5;
                color: #333;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }

            header {
                background: white;
                border-bottom: 1px solid #e1e8ed;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }

            header h1 {
                margin: 0;
                font-size: 2rem;
                color: #1a1a1a;
            }

            header p {
                margin: 0.5rem 0 0 0;
                color: #666;
            }

            section {
                background: white;
                border-radius: 8px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            section h2 {
                margin-top: 0;
                color: #1a1a1a;
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }

            section h3 {
                margin-top: 0;
                color: #333;
                font-size: 1.2rem;
                margin-bottom: 1rem;
            }

            section p {
                color: #4a5568;
                line-height: 1.6;
                margin-bottom: 1rem;
            }

            footer {
                text-align: center;
                padding: 2rem;
                color: #666;
                font-size: 0.9rem;
            }

            a {
                color: #0066cc;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }

            /* Interactive editor styles */
            .editor-container {
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                gap: 20px;
                margin: 20px 0;
            }

            .editor-pane {
                display: flex;
                flex-direction: column;
                min-width: 0;
                height: 100%;
            }

            .editor-pane h3 {
                margin-bottom: 10px;
                flex-shrink: 0;
            }

            .editor-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                flex-shrink: 0;
            }

            .editor-input {
                background: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #4a5568;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace;
                font-size: 14px;
                line-height: 1.6;
                min-height: 400px;
                resize: vertical;
                width: 100%;
                box-sizing: border-box;
                tab-size: 4;
                white-space: pre;
                overflow-x: auto;
                flex: 1;
            }

            .editor-output {
                border: 1px solid #4a5568;
                border-radius: 8px;
                min-height: 400px;
                overflow: auto;
                position: relative;
            }

            /* Override code-container styles for editor output */
            .editor-output .code-container {
                border: none;
                border-radius: 8px;
                min-height: 400px;
                margin: 0;
            }

            .performance-stats {
                background: #e6f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 4px;
                padding: 12px;
                margin: 16px 0;
                font-family: monospace;
                font-size: 13px;
            }

            /* Theme selector styles */
            .theme-selector {
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 10px;
            }

            .theme-selector label {
                font-weight: 600;
                color: #333;
            }

            .theme-selector select {
                padding: 8px 12px;
                border-radius: 6px;
                border: 1px solid #e1e8ed;
                background: white;
                font-size: 14px;
                cursor: pointer;
            }

            .theme-selector select:hover {
                border-color: #0066cc;
            }

            .stats-explanation {
                background: #f0f7ff;
                border-left: 4px solid #0066cc;
                padding: 12px;
                margin-top: 10px;
                font-size: 13px;
                color: #333;
            }

            .stats-explanation p {
                color: #333;
                margin: 8px 0;
            }

            .stats-explanation strong {
                color: #0066cc;
                font-weight: 600;
            }

            @media (max-width: 968px) {
                .editor-container {
                    grid-template-columns: 1fr;
                    grid-template-rows: auto auto;
                }

                .editor-pane {
                    width: 100%;
                }

                .editor-input {
                    min-height: 300px;
                    max-width: 100%;
                }

                .editor-output {
                    min-height: 300px;
                    max-width: 100%;
                }
            }

            @media (max-width: 768px) {
                .container {
                    padding: 1rem;
                }

                section {
                    padding: 1rem;
                }

                .editor-input {
                    min-height: 250px;
                    font-size: 13px;
                }

                .editor-output {
                    min-height: 250px;
                }

                .code-container {
                    padding: 12px;
                    font-size: 13px;
                }

                .code-container pre {
                    font-size: 13px;
                }
            }
        """),
    ),
)

# Sample Python code for demonstrations
SAMPLE_CODES = {
    "hello_world": """from fasthtml.common import *

app, rt = fast_app()

@rt("/")
def get():
    return Titled("FastHTML Example",
        P("Hello from FastHTML!"),
        Button("Click me",
               hx_get="/clicked",
               hx_swap="outerHTML")
    )

@rt("/clicked")
def get():
    return P("Button was clicked!",
             style="color: green;")

serve()""",
    "class_example": '''from dataclasses import dataclass
from typing import Optional
from fasthtml.common import *

@dataclass
class TodoItem:
    """A todo item with ID, text, and completion status."""
    id: int
    text: str
    done: bool = False
    created_at: Optional[str] = None

    def toggle(self) -> None:
        """Toggle the completion status."""
        self.done = not self.done

    def to_ft(self) -> FT:
        """Render as FastHTML component with HTMX."""
        style = "text-decoration: line-through;" if self.done else ""
        return Div(
            Input(
                type="checkbox",
                checked=self.done,
                hx_post=f"/toggle/{self.id}",
                hx_target=f"#todo-{self.id}",
                hx_swap="outerHTML"
            ),
            Span(self.text, style=style),
            cls="todo-item",
            id=f"todo-{self.id}"
        )''',
    "async_example": '''import asyncio
from typing import List

async def fetch_data(url: str) -> dict:
    """Simulate async data fetching."""
    print(f"Fetching data from {url}")
    await asyncio.sleep(1)
    return {"url": url, "data": "sample data"}

async def process_urls(urls: List[str]) -> List[dict]:
    """Process multiple URLs concurrently."""
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Usage
urls = [
    "https://api1.example.com",
    "https://api2.example.com"
]
results = asyncio.run(process_urls(urls))
print(f"Processed {len(results)} URLs")''',
    "pattern_matching": '''def analyze_data(data):
    """Pattern matching example."""
    match data:
        case {'type': 'user',
              'name': str(name),
              'age': int(age)} if age >= 18:
            return f"Adult user: {name}"

        case {'type': 'user',
              'name': str(name),
              'age': int(age)}:
            return f"Minor user: {name}"

        case {'type': 'product',
              'price': float(price)} if price > 0:
            return f"Product costs ${price:.2f}"

        case list() if len(data) > 0:
            return f"List with {len(data)} items"

        case str(text) if text.startswith('ERROR'):
            return f"Error message: {text}"

        case _:
            return "Unknown data type"

# Test cases
test_data = [
    {'type': 'user', 'name': 'Alice', 'age': 25},
    {'type': 'product', 'price': 29.99},
    ['item1', 'item2', 'item3'],
    'ERROR: Something went wrong'
]

for item in test_data:
    print(analyze_data(item))''',
}


@rt("/")
def get():
    """Home page with basic syntax highlighting examples."""
    return Div(
        Header(
            Div(
                H1("🌟 Starlighter + FastHTML Integration"),
                P("Server-side Python syntax highlighting for FastHTML applications"),
                cls="container",
            )
        ),
        Div(
            Section(
                H2("Basic Example"),
                P("Here's a simple function highlighted with Starlighter:"),
                CodeBlock(SAMPLE_CODES["hello_world"]),
            ),
            Section(
                H2("Class Definition with Decorators"),
                P("More complex example with dataclass and properties:"),
                CodeBlock(SAMPLE_CODES["class_example"]),
            ),
            Section(
                H2("Async/Await Example"),
                P("Modern Python with async programming:"),
                CodeBlock(SAMPLE_CODES["async_example"]),
            ),
            Section(
                H2("Pattern Matching (Python 3.10+)"),
                P("Structural pattern matching with match/case:"),
                CodeBlock(SAMPLE_CODES["pattern_matching"]),
            ),
            Section(
                H2("Why Starlighter?"),
                Div(
                    P(
                        "Starlighter provides:",
                        style="color: #2d3748; font-weight: 500;",
                    ),
                    Ul(
                        Li("🚀 Server-side rendering - no JavaScript required"),
                        Li("🎨 FastHTML/StarHTML-aware syntax highlighting"),
                        Li("⚡ Zero dependencies - pure Python"),
                        Li("🔧 Customizable themes via CSS"),
                        Li("📊 Sub-10ms performance for typical files"),
                        style="color: #4a5568; line-height: 1.8;",
                    ),
                    P(
                        "Try the ",
                        A("interactive code editor", href="/interactive"),
                        " to see real-time highlighting in action, or explore ",
                        A("different themes", href="/demo/themes"),
                        ".",
                    ),
                ),
            ),
            cls="container",
        ),
        Footer(P("Powered by Starlighter - Zero-dependency Python syntax highlighter")),
    )


@rt("/interactive")
def get():
    """Interactive code editor page."""
    return Div(
        Header(
            Div(
                H1("🔧 Interactive Code Editor"),
                P("Type Python code and see it highlighted in real-time"),
                P(A("← Back to examples", href="/")),
                cls="container",
            )
        ),
        Div(
            Section(
                Div(
                    Div(
                        H3("Input (Python Code)"),
                        Form(
                            Textarea(
                                SAMPLE_CODES["hello_world"],
                                name="code",
                                id="code-input",
                                cls="editor-input",
                                placeholder="Enter Python code here...",
                                spellcheck="false",
                                autocomplete="off",
                                autocorrect="off",
                                autocapitalize="off",
                            ),
                            hx_post="/api/highlight",
                            hx_target="#code-output",
                            hx_trigger="input changed delay:500ms from:textarea",
                            hx_swap="outerHTML",
                        ),
                        cls="editor-pane",
                    ),
                    Div(
                        Div(
                            H3("Output (Highlighted HTML)"),
                            Div(
                                Label("Theme: ", for_="theme-select"),
                                Form(
                                    Select(
                                        Option(
                                            "VS Code Dark+",
                                            value="vscode",
                                            selected=True,
                                        ),
                                        Option("VS Code Light+", value="light"),
                                        Option("Monokai", value="monokai"),
                                        Option("Catppuccin Mocha", value="catppuccin"),
                                        Option("GitHub Dark", value="github-dark"),
                                        Option("Dracula", value="dracula"),
                                        name="theme",
                                        id="theme-select",
                                    ),
                                    Hidden(
                                        name="code", value="", id="theme-code-input"
                                    ),
                                    hx_get="/api/theme",
                                    hx_target="#code-output",
                                    hx_trigger="change from:select",
                                    hx_swap="outerHTML",
                                    style="display: inline;",
                                ),
                                cls="theme-selector",
                            ),
                            cls="editor-header",
                        ),
                        Div(
                            P(
                                "Enter some Python code to see it highlighted",
                                style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
                            ),
                            id="code-output",
                            cls="editor-output",
                            hx_get="/api/highlight-initial",
                            hx_trigger="load",
                            hx_swap="outerHTML",
                        ),
                        cls="editor-pane",
                    ),
                    cls="editor-container",
                )
            ),
            Div(id="performance-info", cls="performance-stats"),
            Div(
                Details(
                    Summary(
                        "What do these metrics mean?",
                        style="color: #2d3748; font-weight: 500; cursor: pointer;",
                    ),
                    Div(
                        P(
                            Strong("Server time:"),
                            " Time taken by Starlighter's Python highlight() function to parse and convert your code to HTML on the server.",
                        ),
                        P(
                            Strong("Client time:"),
                            " Total round-trip time including network request, server processing, response, and browser DOM update.",
                        ),
                        P(
                            Strong("Network latency:"),
                            " The difference between client and server time represents network overhead and browser rendering.",
                        ),
                        P(
                            "For optimal performance, server time should be under 10ms for typical files."
                        ),
                        cls="stats-explanation",
                    ),
                )
            ),
            Div(id="error-info"),
            cls="container",
        ),
        Script("""
            // Minimal JavaScript - sync code between forms
            document.addEventListener('DOMContentLoaded', function() {
                const codeInput = document.getElementById('code-input');
                const themeCodeInput = document.getElementById('theme-code-input');

                // Sync code to theme form
                function syncCode() {
                    if (themeCodeInput) themeCodeInput.value = codeInput.value;
                }

                // Sync on input
                codeInput.addEventListener('input', syncCode);

                // Initial sync
                syncCode();
            });
        """),
    )


@rt("/api/highlight", methods=["POST"])
async def post(code: str = ""):
    """HTMX endpoint for code highlighting."""
    import time

    try:
        if not code.strip():
            return Div(
                P(
                    "Enter some Python code to see it highlighted",
                    style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
                ),
                id="code-output",
                cls="editor-output",
            )

        # Time the highlighting operation
        start_time = time.perf_counter()

        try:
            # Use CodeBlock for consistent rendering
            highlighted_elem = CodeBlock(code)
            # Extract just the inner HTML content
            highlighted_html = to_xml(highlighted_elem)

        except (ValueError, RuntimeError) as e:
            return Div(
                P(
                    f"Highlighting failed: {str(e)}",
                    style="color: #c33; text-align: center; margin: 50px;",
                ),
                id="code-output",
                cls="editor-output",
            )

        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        # Return highlighted HTML wrapped in editor output div
        return Div(
            NotStr(highlighted_html),
            id="code-output",
            cls="editor-output",
            **{
                "hx-trigger": "load",
                "hx-get": f"/api/performance?server_time={server_time_ms}&lines={len(code.split('\n'))}&chars={len(code)}&html_len={len(highlighted_html)}",
                "hx-target": "#performance-info",
                "hx-swap": "innerHTML",
            },
        )

    except Exception as e:
        return Div(
            P(
                f"Server error: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            id="code-output",
            cls="editor-output",
        )


@rt("/api/highlight-initial")
def get():
    """HTMX endpoint for initial code highlighting on page load."""
    import time

    try:
        # Use the default sample code for initial highlighting
        code = SAMPLE_CODES["hello_world"]

        start_time = time.perf_counter()
        highlighted_elem = CodeBlock(code)
        highlighted_html = to_xml(highlighted_elem)
        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        # Return highlighted HTML with performance data trigger
        return Div(
            NotStr(highlighted_html),
            id="code-output",
            cls="editor-output",
            **{
                "hx-trigger": "load",
                "hx-get": f"/api/performance?server_time={server_time_ms}&lines={len(code.split('\n'))}&chars={len(code)}&html_len={len(highlighted_html)}",
                "hx-target": "#performance-info",
                "hx-swap": "innerHTML",
            },
        )

    except Exception as e:
        return Div(
            P(
                f"Initial highlighting failed: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            id="code-output",
            cls="editor-output",
        )


@rt("/api/theme")
def get(code: str = "", theme: str = "vscode"):
    """HTMX endpoint for theme switching."""
    import time

    if not code.strip():
        return Div(
            P(
                "Enter some Python code to see it highlighted",
                style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
            ),
            id="code-output",
            cls="editor-output",
        )

    try:
        start_time = time.perf_counter()
        # Use CodeBlock with specified theme
        highlighted_elem = CodeBlock(code, theme=theme)
        highlighted_html = to_xml(highlighted_elem)
        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        return Div(
            NotStr(highlighted_html),
            id="code-output",
            cls="editor-output",
            **{
                "hx-trigger": "load",
                "hx-get": f"/api/performance?server_time={server_time_ms}&lines={len(code.split('\n'))}&chars={len(code)}&html_len={len(highlighted_html)}",
                "hx-target": "#performance-info",
                "hx-swap": "innerHTML",
            },
        )

    except Exception as e:
        return Div(
            P(
                f"Highlighting failed: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            id="code-output",
            cls="editor-output",
        )


@rt("/api/performance")
def get(server_time: float, lines: int, chars: int, html_len: int):
    """HTMX endpoint for performance stats."""
    return Div(
        NotStr(f"""
            📊 <strong>Performance Metrics</strong><br>
            <div style="margin-top: 8px;">
                <strong>Server Processing:</strong><br>
                • Python highlighting: {server_time:.2f}ms<br>
                • HTMX round-trip: ~{server_time + 5:.0f}ms<br>
            </div>
            <div style="margin-top: 8px;">
                <strong>Input/Output:</strong><br>
                • Input: {lines} lines, {chars:,} chars<br>
                • Output HTML: {html_len:,} chars<br>
                • Expansion ratio: {html_len / chars if chars > 0 else 0:.1f}x
            </div>
        """)
    )


@rt("/demo/themes")
def get():
    """Demonstrate different color themes."""
    sample_code = SAMPLE_CODES["class_example"]

    return Div(
        Header(
            Div(
                H1("🎨 Theme Demo"),
                P(A("← Back to examples", href="/")),
                P("The same code with different CSS themes:"),
                cls="container",
            )
        ),
        Div(
            Section(
                H2("VS Code Dark+ (Default)"), CodeBlock(sample_code, theme="vscode")
            ),
            Section(H2("VS Code Light+"), CodeBlock(sample_code, theme="light")),
            Section(H2("Monokai"), CodeBlock(sample_code, theme="monokai")),
            Section(H2("Catppuccin Mocha"), CodeBlock(sample_code, theme="catppuccin")),
            Section(H2("GitHub Dark"), CodeBlock(sample_code, theme="github-dark")),
            Section(H2("Dracula"), CodeBlock(sample_code, theme="dracula")),
            Section(
                H2("Custom Styling Example"),
                P("You can customize the appearance by overriding CSS classes:"),
                Style("""
                    .custom-theme {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: 2px solid #764ba2;
                    }
                    .custom-theme pre {
                        color: #fff;
                    }
                    .custom-theme .token-keyword {
                        color: #ffd700;
                        font-weight: bold;
                    }
                    .custom-theme .token-string {
                        color: #98fb98;
                    }
                    .custom-theme .token-comment {
                        color: #e0e0e0;
                        font-style: italic;
                    }
                    .custom-theme .token-number {
                        color: #ff69b4;
                    }
                    .custom-theme .token-identifier {
                        color: #87ceeb;
                    }
                """),
                Div(CodeBlock(sample_code), cls="custom-theme"),
            ),
            cls="container",
        ),
    )


if __name__ == "__main__":
    serve(port=5001)
