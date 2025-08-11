import time

from fastcore.xml import to_xml
from starhtml import *
from starlighter import CodeBlock, StarlighterStyles
from starlighter import InputError, ParseError, RenderError

# Create StarHTML app with Starlighter styles
app, rt = star_app(
    hdrs=(
        # Include all themes for the interactive editor demo
        StarlighterStyles(
            "github-dark", "light", "monokai", "catppuccin", "vscode", "dracula"
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

            /* Notification styles for Datastar */
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #48bb78;
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                animation: slideIn 0.3s ease-out;
                z-index: 1000;
            }

            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
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
    )
)

# Sample Python code for demonstrations
SAMPLE_CODES = {
    "hello_world": """from starhtml import *

app, rt = star_app()

@rt("/")
def get():
    return Titled("StarHTML Example",
        P("Hello from StarHTML with Datastar!"),
        Button("Click me",
               ds_on_click("@post('/clicked')"))
    )

@rt("/clicked", methods=["POST"])
def post():
    return P("Button was clicked!",
             style="color: green;")

serve()""",
    "datastar_example": """from starhtml import *

app, rt = star_app()

@rt("/")
def get():
    return Titled("Datastar Reactivity",
        Div(
            Input(
                type="text",
                placeholder="Type something...",
                ds_bind("message")
            ),
            P("You typed: ",
              Span(ds_text("$message"))),
            Button("Clear",
                   ds_on_click("$message=''")),
            ds_signals(message="")
        )
    )

serve()""",
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
def home():
    """Home page with basic syntax highlighting examples."""
    return Div(
        Header(
            Div(
                H1("🌟 Starlighter + StarHTML Integration"),
                P("Server-side Python syntax highlighting with Datastar reactivity"),
                cls="container",
            )
        ),
        Div(
            Section(
                H2("Basic Example"),
                P("Here's a simple StarHTML app with Datastar:"),
                CodeBlock(SAMPLE_CODES["hello_world"]),
            ),
            Section(
                H2("Datastar Reactivity"),
                P("Example showcasing reactive data binding:"),
                CodeBlock(SAMPLE_CODES["datastar_example"]),
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
                H2("Why Starlighter + StarHTML?"),
                Div(
                    P(
                        "The perfect combination provides:",
                        style="color: #2d3748; font-weight: 500;",
                    ),
                    Ul(
                        Li("🚀 Server-side rendering with Datastar reactivity"),
                        Li(
                            "🎨 Beautiful syntax highlighting without JavaScript libraries"
                        ),
                        Li("⚡ Zero dependencies for highlighting"),
                        Li("🔧 Reactive UI updates with Datastar"),
                        Li("📊 Sub-10ms highlighting performance"),
                        style="color: #4a5568; line-height: 1.8;",
                    ),
                    P(
                        "Try the ",
                        A("interactive code editor", href="/interactive"),
                        " to see real-time highlighting with Datastar, or explore ",
                        A("different themes", href="/demo/themes"),
                        ".",
                    ),
                ),
            ),
            cls="container",
        ),
        Footer(P("Powered by Starlighter + StarHTML with Datastar")),
    )


@rt("/interactive")
def get():
    """Interactive code editor page with Datastar."""
    return Div(
        Header(
            Div(
                H1("🔧 Interactive Code Editor"),
                P("Type Python code and see it highlighted in real-time with Datastar"),
                P(A("← Back to examples", href="/")),
                cls="container",
            )
        ),
        Div(
            Section(
                Div(
                    Div(
                        H3("Input (Python Code)"),
                        Textarea(
                            SAMPLE_CODES["hello_world"],
                            ds_bind("code"),
                            ds_on_input("@post('/api/highlight')", debounce="500ms"),
                            id="code-input",
                            cls="editor-input",
                            placeholder="Enter Python code here...",
                            spellcheck="false",
                            autocomplete="off",
                            autocorrect="off",
                            autocapitalize="off",
                        ),
                        cls="editor-pane",
                    ),
                    Div(
                        Div(
                            H3("Output (Highlighted HTML)"),
                            Div(
                                Label("Theme: ", for_="theme-select"),
                                Select(
                                    Option(
                                        "GitHub Dark",
                                        value="github-dark",
                                        selected=True,
                                    ),
                                    Option("VS Code Light+", value="light"),
                                    Option("VS Code Dark+", value="vscode"),
                                    Option("Monokai", value="monokai"),
                                    Option("Catppuccin Mocha", value="catppuccin"),
                                    Option("Dracula", value="dracula"),
                                    ds_bind("theme"),
                                    ds_on_change("@post('/api/theme')"),
                                    id="theme-select",
                                ),
                                cls="theme-selector",
                            ),
                            cls="editor-header",
                        ),
                        Div(
                            P(
                                "Loading code highlighting...",
                                style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
                            ),
                            ds_on_load("@get('/api/highlight-initial')"),
                            id="code-output",
                            cls="editor-output",
                        ),
                        cls="editor-pane",
                    ),
                    ds_signals(code=SAMPLE_CODES["hello_world"], theme="github-dark"),
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
                            Strong("Network time:"),
                            " Total round-trip time including request, server processing, and response.",
                        ),
                        P(
                            Strong("Datastar update:"),
                            " Time for Datastar to update the DOM with new content.",
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
    )


@rt("/api/highlight")
@sse
def post(req, code: str = "", theme: str = "github-dark"):
    """Datastar SSE endpoint for code highlighting."""
    try:
        if not code.strip():
            yield elements(
                P(
                    "Enter some Python code to see it highlighted",
                    style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
                ),
                "#code-output",
                "inner",
            )
            return

        # Time the highlighting operation
        start_time = time.perf_counter()

        try:
            # Use CodeBlock for consistent rendering with specified theme
            highlighted_elem = CodeBlock(code, theme=theme)
            # Extract just the inner HTML content
            highlighted_html = to_xml(highlighted_elem)

        except (InputError, ParseError, RenderError) as e:
            yield elements(
                P(
                    f"Highlighting failed: {str(e)}",
                    style="color: #c33; text-align: center; margin: 50px;",
                ),
                "#code-output",
                "inner",
            )
            return

        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        # Update highlighted code output
        yield elements(NotStr(highlighted_html), "#code-output", "inner")

        # Update performance stats
        performance_html = f"""
            📊 <strong>Performance Metrics</strong><br>
            <div style="margin-top: 8px;">
                <strong>Server Processing:</strong><br>
                • Python highlighting: {server_time_ms:.2f}ms<br>
                • Datastar update: ~5ms<br>
            </div>
            <div style="margin-top: 8px;">
                <strong>Input/Output:</strong><br>
                • Input: {len(code.split("\n"))} lines, {len(code):,} chars<br>
                • Output HTML: {len(highlighted_html):,} chars<br>
                • Expansion ratio: {len(highlighted_html) / len(code) if code else 0:.1f}x
            </div>
        """

        yield elements(NotStr(performance_html), "#performance-info", "inner")

    except Exception as e:
        yield elements(
            P(
                f"Server error: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            "#code-output",
            "inner",
        )


@rt("/api/highlight-initial")
@sse
def get(req):
    """Datastar SSE endpoint for initial code highlighting on page load."""
    try:
        # Use the default sample code for initial highlighting
        code = SAMPLE_CODES["hello_world"]
        theme = "github-dark"  # Default theme

        start_time = time.perf_counter()
        highlighted_elem = CodeBlock(code, theme=theme)
        highlighted_html = to_xml(highlighted_elem)
        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        # Update the code output
        yield elements(NotStr(highlighted_html), "#code-output", "inner")

        # Update performance stats
        performance_html = f"""
            📊 <strong>Performance Metrics</strong><br>
            <div style="margin-top: 8px;">
                <strong>Server Processing:</strong><br>
                • Python highlighting: {server_time_ms:.2f}ms<br>
                • Initial load: instant<br>
            </div>
            <div style="margin-top: 8px;">
                <strong>Input/Output:</strong><br>
                • Input: {len(code.split("\n"))} lines, {len(code):,} chars<br>
                • Output HTML: {len(highlighted_html):,} chars<br>
                • Expansion ratio: {len(highlighted_html) / len(code) if code else 0:.1f}x
            </div>
        """

        yield elements(NotStr(performance_html), "#performance-info", "inner")

    except Exception as e:
        yield elements(
            P(
                f"Initial highlighting failed: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            "#code-output",
            "inner",
        )


@rt("/api/theme")
@sse
def post(req, code: str = "", theme: str = "github-dark"):
    """Datastar SSE endpoint for theme switching."""
    try:
        if not code.strip():
            yield elements(
                P(
                    "Enter some Python code to see it highlighted",
                    style="color: #808080; text-align: center; margin: 50px; font-style: italic;",
                ),
                "#code-output",
                "inner",
            )
            return

        start_time = time.perf_counter()
        # Use CodeBlock with specified theme
        highlighted_elem = CodeBlock(code, theme=theme)
        highlighted_html = to_xml(highlighted_elem)
        end_time = time.perf_counter()
        server_time_ms = (end_time - start_time) * 1000

        # Update highlighted code output
        yield elements(NotStr(highlighted_html), "#code-output", "inner")

        # Update performance stats
        performance_html = f"""
            📊 <strong>Performance Metrics</strong><br>
            <div style="margin-top: 8px;">
                <strong>Server Processing:</strong><br>
                • Python highlighting: {server_time_ms:.2f}ms<br>
                • Theme switch: instant<br>
            </div>
            <div style="margin-top: 8px;">
                <strong>Input/Output:</strong><br>
                • Input: {len(code.split("\n"))} lines, {len(code):,} chars<br>
                • Output HTML: {len(highlighted_html):,} chars<br>
                • Expansion ratio: {len(highlighted_html) / len(code) if code else 0:.1f}x
            </div>
        """

        yield elements(NotStr(performance_html), "#performance-info", "inner")

    except Exception as e:
        yield elements(
            P(
                f"Highlighting failed: {str(e)}",
                style="color: #c33; text-align: center; margin: 50px;",
            ),
            "#code-output",
            "inner",
        )


@rt("/demo/themes")
def get():
    """Demonstrate different color themes."""
    sample_code = SAMPLE_CODES["datastar_example"]

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
            Section(H2("GitHub Dark (Default)"), CodeBlock(sample_code)),
            Section(H2("VS Code Light+"), CodeBlock(sample_code, theme="light")),
            Section(H2("VS Code Dark+"), CodeBlock(sample_code, theme="vscode")),
            Section(H2("Monokai"), CodeBlock(sample_code, theme="monokai")),
            Section(H2("Catppuccin Mocha"), CodeBlock(sample_code, theme="catppuccin")),
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
