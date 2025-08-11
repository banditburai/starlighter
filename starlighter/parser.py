"""High-performance streaming lexer; simple and fast by design."""

from typing import Optional


PYTHON_KEYWORDS = frozenset(
    {
        "if",
        "elif",
        "else",
        "for",
        "while",
        "break",
        "continue",
        "pass",
        "def",
        "class",
        "return",
        "yield",
        "lambda",
        "try",
        "except",
        "finally",
        "raise",
        "assert",
        "with",
        "as",
        "import",
        "from",
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "None",
        "del",
        "global",
        "nonlocal",
        "async",
        "await",
        "match",
        "case",
    }
)

PYTHON_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "breakpoint",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "compile",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "eval",
        "exec",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "help",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        "Exception",
        "BaseException",
        "ArithmeticError",
        "LookupError",
        "ValueError",
        "TypeError",
        "IndexError",
        "KeyError",
        "AttributeError",
        "NameError",
        "ImportError",
        "RuntimeError",
        "NotImplementedError",
        "StopIteration",
        "FileNotFoundError",
        "__import__",
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__cached__",
        "Ellipsis",
        "NotImplemented",
    }
)

STARHTML_ELEMENTS = frozenset(
    {
        "Div",
        "Span",
        "P",
        "A",
        "Img",
        "Br",
        "Hr",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "Form",
        "Input",
        "Button",
        "Select",
        "Option",
        "Textarea",
        "Label",
        "Fieldset",
        "Legend",
        "Ul",
        "Ol",
        "Li",
        "Dl",
        "Dt",
        "Dd",
        "Table",
        "Thead",
        "Tbody",
        "Tfoot",
        "Tr",
        "Th",
        "Td",
        "Caption",
        "Colgroup",
        "Col",
        "Header",
        "Footer",
        "Nav",
        "Main",
        "Section",
        "Article",
        "Aside",
        "Figure",
        "Figcaption",
        "Details",
        "Summary",
        "Mark",
        "Time",
        "Progress",
        "Meter",
        "Audio",
        "Video",
        "Source",
        "Track",
        "Canvas",
        "Svg",
        "Dialog",
        "Menu",
        "Menuitem",
        "Strong",
        "Em",
        "B",
        "I",
        "U",
        "S",
        "Small",
        "Sub",
        "Sup",
        "Code",
        "Kbd",
        "Samp",
        "Var",
        "Pre",
        "Blockquote",
        "Cite",
        "Q",
        "Iframe",
        "Embed",
        "Object",
        "Param",
        "Html",
        "Head",
        "Title",
        "Meta",
        "Link",
        "Style",
        "Script",
        "Noscript",
        "Base",
        "Ruby",
        "Rt",
        "Rp",
        "Template",
        "Slot",
    }
)


# Minimal operator tuples to keep checks centralized
OP_2CHARS = frozenset(
    {
        "==",
        "!=",
        "<=",
        ">=",
        "//",
        "**",
        "<<",
        ">>",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "//=",
        "**=",
        "<<=",
        ">>=",
        "->",
        ":=",
    }
)

# Module-level HTML escaping for reuse
HTML_ESCAPE_TABLE = str.maketrans(
    {"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#x27;"}
)


def _escape_html(text: str) -> str:
    if not text or (
        ("<" not in text)
        and (">" not in text)
        and ("&" not in text)
        and ('"' not in text)
        and ("'" not in text)
    ):
        return text
    return text.translate(HTML_ESCAPE_TABLE)


def _find_triple_end(code: str, start_pos: int, quote_char: str) -> int:
    """Return position just after closing triple quote or len(code) if not found."""
    idx = code.find(quote_char * 3, start_pos)
    return (idx + 3) if idx != -1 else len(code)


DATASTAR_LOOKUP = frozenset(
    [
        "data_bind",
        "data_show",
        "data_class",
        "data_style",
        "data_attr",
        "data_text",
        "data_signals",
        "data_effect",
        "data_computed",
        "data_persist",
        "data_on",
        "data_on_click",
        "data_on_input",
        "data_on_keydown",
        "data_on_keyup",
        "data_on_submit",
        "data_on_scroll",
        "data_on_load",
        "data_on_mouseover",
        "data_on_mouseout",
        "data_on_mouseenter",
        "data_on_mouseleave",
        "data_on_focus",
        "data_on_blur",
        "data_on_change",
        "data_on_intersect",
        "data_on_interval",
        "data_on_signal_patch",
        "data_on_signal_patch_filter",
        "data_indicator",
        "data_ignore",
        "data_ignore_morph",
        "data_preserve_attr",
        "data_ref",
        "data_json_signals",
        "data_animate",
        "data_custom_validity",
        "data_on_raf",
        "data_on_resize",
        "data_query_string",
        "data_replace_url",
        "data_scroll_into_view",
        "data_view_transition",
    ]
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
    """Streaming lexer for server-side highlighting (fast, simple)."""

    def __init__(self, code: str):
        self.code = code
        self.length = len(code)
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = code[0] if code else None
        self.errors = []

    def advance(self) -> None:
        """Advance one character; updates line/column on newlines."""
        if self.position < self.length - 1:
            if self.current_char == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1

            self.position += 1
            self.current_char = self.code[self.position]
        else:
            self.position = self.length
            self.current_char = None

    def peek(self, offset: int = 1) -> Optional[str]:
        """Character at current+offset or None if out of bounds."""
        peek_pos = self.position + offset
        if peek_pos < self.length:
            return self.code[peek_pos]
        return None

    def consume(self, expected: str) -> Optional[str]:
        """Return current char and advance if it equals expected; else None."""
        if self.current_char == expected:
            char = self.current_char
            self.advance()
            return char
        return None

    def is_at_end(self) -> bool:
        """True when position reached input end."""
        return self.position >= self.length or self.current_char is None

    def match_sequence(self, sequence: str) -> bool:
        """True if code at position equals sequence (no advance)."""
        if self.position + len(sequence) > self.length:
            return False

        return self.code[self.position : self.position + len(sequence)] == sequence

    def create_error(self, message: str) -> ParseError:
        """Attach position to error for diagnostics."""
        return ParseError(message, self.line, self.column, self.position)

    def record_error(self, message: str) -> None:
        """Collect non-fatal parse errors (rare path)."""
        self.errors.append(self.create_error(message))

    def highlight_streaming(self) -> str:
        """Streaming HTML; keeps hot paths branch-light for speed."""
        code = self.code
        length = self.length

        if not code:
            return '<pre><code class="language-python"></code></pre>'

        # Use simple list append
        result = ['<pre><code class="language-python">']
        result_append = result.append  # Cache method lookup for speed

        # Bulk processing approach - scan for token boundaries, then process ranges
        pos = 0

        # Cache lookups
        python_keywords = PYTHON_KEYWORDS
        python_builtins = PYTHON_BUILTINS
        starhtml_elements = STARHTML_ELEMENTS
        datastar_lookup = DATASTAR_LOOKUP

        while pos < length:
            char = code[pos]

            # Bulk whitespace processing - most common case
            if char in " \t":
                start = pos
                while pos < length and code[pos] in " \t":
                    pos += 1
                result_append(code[start:pos])  # No escaping needed for whitespace
                continue

            # Newlines - direct append
            elif char == "\n":
                pos += 1
                result_append("\n")
                continue

            # Comments - scan to end of line (bulk approach)
            elif char == "#":
                start = pos
                pos += 1
                while pos < length and code[pos] != "\n":
                    pos += 1
                comment_text = code[start:pos]
                result_append(
                    '<span class="token-comment">'
                    + _escape_html(comment_text)
                    + "</span>"
                )
                continue

            # Strings - bulk string processing
            elif char in "\"'":
                start = pos
                quote_char = char
                pos += 1

                # Fast triple quote check
                is_triple = (
                    pos + 1 < length
                    and code[pos] == quote_char
                    and code[pos + 1] == quote_char
                )
                if is_triple:
                    pos = _find_triple_end(code, pos + 2, quote_char)
                else:
                    # Find closing quote, handling escapes
                    while pos < length:
                        if code[pos] == "\\" and pos + 1 < length:
                            pos += 2
                        elif code[pos] == quote_char:
                            pos += 1
                            break
                        elif code[pos] == "\n":
                            break  # Unterminated
                        else:
                            pos += 1

                string_text = code[start:pos]
                result_append(
                    '<span class="token-string">'
                    + _escape_html(string_text)
                    + "</span>"
                )
                continue

            # Identifiers - ultra-fast bulk scanning
            elif char == "_" or "A" <= char <= "Z" or "a" <= char <= "z":
                start = pos
                pos += 1

                # Identifier scanning - direct character ranges
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

                # Check for string prefixes (f, r, b, u, etc.)
                if (
                    pos < length
                    and code[pos] in "\"'"
                    and identifier.lower()
                    in ("f", "r", "b", "u", "fr", "rf", "br", "rb")
                ):
                    # Parse the entire prefixed string
                    string_start = start
                    quote_char = code[pos]
                    pos += 1

                    # Handle triple quotes
                    is_triple = (
                        pos + 1 < length
                        and code[pos] == quote_char
                        and code[pos + 1] == quote_char
                    )
                    if is_triple:
                        pos = _find_triple_end(code, pos + 2, quote_char)
                    else:
                        # Find closing quote
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
                    result_append(
                        '<span class="token-string">'
                        + _escape_html(full_string)
                        + "</span>"
                    )
                    continue

                # Classify identifier - inline templates
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
                else:
                    result_append(
                        '<span class="token-identifier">' + identifier + "</span>"
                    )
                continue

            # Numbers with direct character ranges
            elif ("0" <= char <= "9") or (
                char == "." and pos + 1 < length and "0" <= code[pos + 1] <= "9"
            ):
                start = pos

                # Handle special number formats
                if char == "0" and pos + 1 < length:
                    second = code[pos + 1]
                    if second in "xX":  # Hex
                        pos += 2
                        while pos < length and (
                            ("0" <= code[pos] <= "9")
                            or ("a" <= code[pos] <= "f")
                            or ("A" <= code[pos] <= "F")
                        ):
                            pos += 1
                    elif second in "bB":  # Binary
                        pos += 2
                        while pos < length and code[pos] in "01":
                            pos += 1
                    elif second in "oO":  # Octal
                        pos += 2
                        while pos < length and "0" <= code[pos] <= "7":
                            pos += 1
                    else:  # Regular number
                        pos += 1
                        while pos < length and "0" <= code[pos] <= "9":
                            pos += 1
                else:  # Regular number or float
                    while pos < length and "0" <= code[pos] <= "9":
                        pos += 1

                # Handle decimal part
                if (
                    pos < length
                    and code[pos] == "."
                    and pos + 1 < length
                    and "0" <= code[pos + 1] <= "9"
                ):
                    pos += 1
                    while pos < length and "0" <= code[pos] <= "9":
                        pos += 1

                # Handle exponent
                if pos < length and code[pos] in "eE":
                    exp_pos = pos + 1
                    if exp_pos < length and code[exp_pos] in "+-":
                        exp_pos += 1
                    if exp_pos < length and "0" <= code[exp_pos] <= "9":
                        pos = exp_pos
                        while pos < length and "0" <= code[pos] <= "9":
                            pos += 1

                number = code[start:pos]
                result_append(
                    '<span class="token-number">' + number + "</span>"
                )  # Numbers don't need escaping
                continue

            # Operators - batch process
            elif char in "+-*/%=<>!&|^~":
                start = pos
                pos += 1

                # Check for multi-character operators
                if pos < length:
                    two_char = code[start : pos + 1]
                    if two_char in OP_2CHARS:
                        pos += 1

                operator = code[start:pos]
                # Operators < > need escaping, single concatenation
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

            # Punctuation - direct processing
            elif char in "()[]{}.,:;":
                pos += 1
                result_append('<span class="token-punctuation">' + char + "</span>")
                continue

            # Decorators
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

                decorator = code[start:pos]
                result_append('<span class="token-decorator">' + decorator + "</span>")
                continue

            else:
                # Single unknown character
                pos += 1
                if char in "<>&\"'":
                    result_append(_escape_html(char))
                else:
                    result_append(char)

        # Close HTML and return
        result_append("</code></pre>")
        return "".join(result)
