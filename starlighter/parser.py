"""High-performance streaming lexer; simple and fast by design."""

# fmt: off
PYTHON_KEYWORDS = frozenset({
    "if", "elif", "else", "for", "while", "break", "continue", "pass",
    "def", "class", "return", "yield", "lambda",
    "try", "except", "finally", "raise", "assert", "with", "as",
    "import", "from", "and", "or", "not", "in", "is",
    "True", "False", "None", "del", "global", "nonlocal",
    "async", "await", "match", "case",
})

PYTHON_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset",
    "getattr", "globals", "hasattr", "hash", "help", "hex",
    "id", "input", "int", "isinstance", "issubclass", "iter",
    "len", "list", "locals", "map", "max", "memoryview", "min", "next",
    "object", "oct", "open", "ord", "pow", "print", "property",
    "range", "repr", "reversed", "round",
    "set", "setattr", "slice", "sorted", "staticmethod",
    "str", "sum", "super", "tuple", "type", "vars", "zip",
    "Exception", "BaseException", "ArithmeticError", "LookupError",
    "ValueError", "TypeError", "IndexError", "KeyError",
    "AttributeError", "NameError", "ImportError", "RuntimeError",
    "NotImplementedError", "StopIteration", "FileNotFoundError",
    "__import__", "__name__", "__file__", "__doc__",
    "__package__", "__loader__", "__spec__", "__builtins__", "__cached__",
    "Ellipsis", "NotImplemented",
})

STARHTML_ELEMENTS = frozenset({
    "Div", "Span", "P", "A", "Img", "Br", "Hr",
    "H1", "H2", "H3", "H4", "H5", "H6",
    "Form", "Input", "Button", "Select", "Option", "Textarea", "Label", "Fieldset", "Legend",
    "Ul", "Ol", "Li", "Dl", "Dt", "Dd",
    "Table", "Thead", "Tbody", "Tfoot", "Tr", "Th", "Td", "Caption", "Colgroup", "Col",
    "Header", "Footer", "Nav", "Main", "Section", "Article", "Aside",
    "Figure", "Figcaption", "Details", "Summary",
    "Mark", "Time", "Progress", "Meter",
    "Audio", "Video", "Source", "Track", "Canvas", "Svg",
    "Dialog", "Menu", "Menuitem",
    "Strong", "Em", "B", "I", "U", "S", "Small", "Sub", "Sup",
    "Code", "Kbd", "Samp", "Var", "Pre", "Blockquote", "Cite", "Q",
    "Iframe", "Embed", "Object", "Param",
    "Html", "Head", "Title", "Meta", "Link", "Style", "Script", "Noscript", "Base",
    "Ruby", "Rt", "Rp", "Template", "Slot",
})

OP_2CHARS = frozenset({
    "==", "!=", "<=", ">=", "//", "**", "<<", ">>",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "//=", "**=", "<<=", ">>=", "->", ":=",
})
# fmt: on

_ESCAPE_CHARS = frozenset("<>&\"'")
HTML_ESCAPE_TABLE = str.maketrans(
    {"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#x27;"}
)


def _escape_html(text: str) -> str:
    if not text or _ESCAPE_CHARS.isdisjoint(text):
        return text
    return text.translate(HTML_ESCAPE_TABLE)


def _find_triple_end(code: str, start_pos: int, quote_char: str) -> int:
    """Return position just after closing triple quote, or len(code) if not found."""
    idx = code.find(quote_char * 3, start_pos)
    return (idx + 3) if idx != -1 else len(code)


# fmt: off
DATASTAR_LOOKUP = frozenset([
    # Core binding
    "data_bind", "data_text", "data_show", "data_class", "data_attr_class", "data_style", "data_attr",
    # Signals
    "data_signals", "data_computed",
    # Event handlers
    "data_on", "data_on_click", "data_on_input", "data_on_keydown", "data_on_keyup",
    "data_on_submit", "data_on_scroll", "data_on_load",
    "data_on_mouseover", "data_on_mouseout", "data_on_mouseenter", "data_on_mouseleave",
    "data_on_focus", "data_on_blur", "data_on_change",
    "data_on_intersect", "data_on_interval", "data_on_raf", "data_on_resize",
    "data_on_signal_patch", "data_on_signal_patch_filter",
    # Plugins
    "data_effect", "data_persist", "data_indicator", "data_ref",
    # Advanced
    "data_ignore", "data_ignore_morph", "data_preserve_attr", "data_animate",
    "data_custom_validity", "data_query_string", "data_replace_url",
    "data_scroll_into_view", "data_view_transition", "data_json_signals", "data_init",
    # Slot attributes
    "data_slot",
])

CSS_CLASS_ATTRS = frozenset({"cls", "class", "data_class", "data_attr_class"})
CSS_STYLE_ATTRS = frozenset({"style", "data_style"})

DATASTAR_METHODS = frozenset([
    "set", "add", "sub", "mul", "div", "mod", "toggle", "toggle_in",        # Signal mutations
    "if_", "then", "default", "one_of",                                      # Conditionals
    "lower", "upper", "strip", "contains",                                   # String methods
    "round", "abs", "min", "max", "clamp",                                   # Number methods
    "append", "prepend", "pop", "remove", "join", "slice",                   # Array methods
    "with_", "js", "f", "match", "switch",                                   # Other / helpers
])
# fmt: on

# StarHTML-specific identifiers treated as builtins for highlighting
STARHTML_BUILTINS = (
    PYTHON_BUILTINS
    | frozenset(
        {"Signal", "cls", "href", "src", "alt", "type", "name", "value", "placeholder"}
    )
    | DATASTAR_METHODS
)


class ParseError(Exception):
    """Raised for catastrophic parse failures (should be rare)."""

    def __init__(self, message: str, line: int = 0, column: int = 0, position: int = 0):
        self.message = message
        self.line = line
        self.column = column
        self.position = position
        super().__init__(f"Parse error at line {line}, column {column}: {message}")


class PythonLexer:
    """Streaming lexer for server-side highlighting."""

    def __init__(
        self,
        code: str,
        elements: set[str] | None = None,
        attrs: set[str] | None = None,
        signals: set[str] | None = None,
    ):
        self.code = code
        self.length = len(code)
        self.elements = (elements or set()) | STARHTML_ELEMENTS
        self.signals = signals or set()
        self.attrs = (attrs or set()) | DATASTAR_LOOKUP

    def highlight_streaming(self) -> str:
        code = self.code
        length = self.length

        if not code:
            return '<pre><code class="language-python"></code></pre>'

        result = ['<pre><code class="language-python">']
        result_append = result.append

        pos = 0
        prev_identifier = None

        python_keywords = PYTHON_KEYWORDS
        python_builtins = STARHTML_BUILTINS
        starhtml_elements = self.elements
        signal_names = self.signals
        datastar_lookup = self.attrs
        css_class_attrs = CSS_CLASS_ATTRS
        css_style_attrs = CSS_STYLE_ATTRS

        while pos < length:
            char = code[pos]

            if char in " \t":
                start = pos
                while pos < length and code[pos] in " \t":
                    pos += 1
                result_append(code[start:pos])
                continue

            elif char == "\n":
                pos += 1
                result_append("\n")
                continue

            elif char == "#":
                start = pos
                pos += 1
                while pos < length and code[pos] != "\n":
                    pos += 1
                result_append(
                    '<span class="token-comment">'
                    + _escape_html(code[start:pos])
                    + "</span>"
                )
                continue

            elif char in "\"'":
                start = pos
                quote_char = char
                pos += 1

                is_triple = (
                    pos + 1 < length
                    and code[pos] == quote_char
                    and code[pos + 1] == quote_char
                )
                if is_triple:
                    pos = _find_triple_end(code, pos + 2, quote_char)
                else:
                    while pos < length:
                        if code[pos] == "\\" and pos + 1 < length:
                            pos += 2
                        elif code[pos] == quote_char:
                            pos += 1
                            break
                        elif code[pos] == "\n":
                            break  # Unterminated string
                        else:
                            pos += 1

                string_text = code[start:pos]

                if prev_identifier in css_class_attrs:
                    token_class = "token-css-class"
                elif prev_identifier in css_style_attrs:
                    token_class = "token-css-style"
                else:
                    token_class = "token-string"

                result_append(
                    '<span class="'
                    + token_class
                    + '">'
                    + _escape_html(string_text)
                    + "</span>"
                )
                continue

            elif char == "$":
                start = pos
                pos += 1
                while pos < length:
                    c = code[pos]
                    if (
                        c == "_"
                        or ("A" <= c <= "Z")
                        or ("a" <= c <= "z")
                        or ("0" <= c <= "9")
                    ):
                        pos += 1
                    else:
                        break
                result_append(
                    '<span class="token-signal">' + code[start:pos] + "</span>"
                )
                continue

            elif char == "_" or "A" <= char <= "Z" or "a" <= char <= "z":
                start = pos
                pos += 1

                while pos < length:
                    c = code[pos]
                    if (
                        c == "_"
                        or ("A" <= c <= "Z")
                        or ("a" <= c <= "z")
                        or ("0" <= c <= "9")
                    ):
                        pos += 1
                    else:
                        break

                identifier = code[start:pos]

                if (
                    pos < length
                    and code[pos] in "\"'"
                    and identifier.lower()
                    in ("f", "r", "b", "u", "fr", "rf", "br", "rb")
                ):
                    string_start = start
                    quote_char = code[pos]
                    pos += 1

                    is_triple = (
                        pos + 1 < length
                        and code[pos] == quote_char
                        and code[pos + 1] == quote_char
                    )
                    if is_triple:
                        pos = _find_triple_end(code, pos + 2, quote_char)
                    else:
                        while pos < length:
                            if code[pos] == "\\" and pos + 1 < length:
                                pos += 2
                            elif code[pos] == quote_char:
                                pos += 1
                                break
                            elif code[pos] == "\n":
                                break
                            else:
                                pos += 1

                    full_string = code[string_start:pos]

                    if prev_identifier in css_class_attrs:
                        token_class = "token-css-class"
                    elif prev_identifier in css_style_attrs:
                        token_class = "token-css-style"
                    else:
                        token_class = "token-string"

                    result_append(
                        '<span class="'
                        + token_class
                        + '">'
                        + _escape_html(full_string)
                        + "</span>"
                    )
                    continue

                if identifier in python_keywords:
                    result_append(
                        '<span class="token-keyword">' + identifier + "</span>"
                    )
                elif identifier in python_builtins:
                    result_append(
                        '<span class="token-builtin">' + identifier + "</span>"
                    )
                elif identifier in starhtml_elements:
                    result_append(
                        '<span class="token-starhtml-element">' + identifier + "</span>"
                    )
                elif identifier in datastar_lookup:
                    result_append(
                        '<span class="token-datastar-attr">' + identifier + "</span>"
                    )
                elif identifier in signal_names:
                    result_append(
                        '<span class="token-signal">' + identifier + "</span>"
                    )
                else:
                    result_append(
                        '<span class="token-identifier">' + identifier + "</span>"
                    )
                prev_identifier = identifier
                continue

            elif ("0" <= char <= "9") or (
                char == "." and pos + 1 < length and "0" <= code[pos + 1] <= "9"
            ):
                start = pos

                if char == "0" and pos + 1 < length:
                    second = code[pos + 1]
                    if second in "xX":
                        pos += 2
                        while pos < length and (
                            ("0" <= code[pos] <= "9")
                            or ("a" <= code[pos] <= "f")
                            or ("A" <= code[pos] <= "F")
                        ):
                            pos += 1
                    elif second in "bB":
                        pos += 2
                        while pos < length and code[pos] in "01":
                            pos += 1
                    elif second in "oO":
                        pos += 2
                        while pos < length and "0" <= code[pos] <= "7":
                            pos += 1
                    else:
                        pos += 1
                        while pos < length and "0" <= code[pos] <= "9":
                            pos += 1
                else:
                    while pos < length and "0" <= code[pos] <= "9":
                        pos += 1

                if (
                    pos < length
                    and code[pos] == "."
                    and pos + 1 < length
                    and "0" <= code[pos + 1] <= "9"
                ):
                    pos += 1
                    while pos < length and "0" <= code[pos] <= "9":
                        pos += 1

                if pos < length and code[pos] in "eE":
                    exp_pos = pos + 1
                    if exp_pos < length and code[exp_pos] in "+-":
                        exp_pos += 1
                    if exp_pos < length and "0" <= code[exp_pos] <= "9":
                        pos = exp_pos
                        while pos < length and "0" <= code[pos] <= "9":
                            pos += 1

                # Numbers are always safe ASCII, no escaping needed
                result_append(
                    '<span class="token-number">' + code[start:pos] + "</span>"
                )
                continue

            elif char in "+-*/%=<>!&|^~":
                start = pos
                pos += 1

                if pos < length:
                    two_char = code[start : pos + 1]
                    if two_char in OP_2CHARS:
                        pos += 1

                operator = code[start:pos]
                # < and > need HTML escaping
                if "<" in operator or ">" in operator:
                    result_append(
                        '<span class="token-operator">'
                        + _escape_html(operator)
                        + "</span>"
                    )
                else:
                    result_append(
                        '<span class="token-operator">' + operator + "</span>"
                    )
                continue

            elif char in "()[]{}.,:;":
                pos += 1
                result_append('<span class="token-punctuation">' + char + "</span>")
                continue

            elif char == "@":
                start = pos
                pos += 1
                while pos < length:
                    c = code[pos]
                    if (
                        c == "_"
                        or ("A" <= c <= "Z")
                        or ("a" <= c <= "z")
                        or ("0" <= c <= "9")
                    ):
                        pos += 1
                    else:
                        break

                result_append(
                    '<span class="token-decorator">' + code[start:pos] + "</span>"
                )
                continue

            else:
                pos += 1
                if char in "<>&\"'":
                    result_append(_escape_html(char))
                else:
                    result_append(char)

        result_append("</code></pre>")
        return "".join(result)
