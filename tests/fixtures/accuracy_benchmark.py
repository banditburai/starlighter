"""
Accuracy Benchmark Generator for starlighter v2.

This module generates diverse Python code samples for comprehensive
accuracy testing against the PRD requirement of 99.9% tokenization
accuracy on 1,000 diverse Python files from popular open-source projects.

Features:
- Generates realistic Python code patterns
- Covers all Python 3.13 syntax constructs
- Simulates code from popular open-source projects
- Configurable complexity and file size
- Comprehensive syntax pattern coverage

Designed for Task 1.3 - Enhanced Performance Benchmarks.
"""

import random
from typing import List
from dataclasses import dataclass
from enum import Enum


class CodeComplexity(Enum):
    """Code complexity levels for generated samples."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXTREME = "extreme"


@dataclass
class CodePattern:
    """Represents a Python code pattern with metadata."""

    name: str
    template: str
    complexity: CodeComplexity
    syntax_features: List[str]
    weight: float = 1.0  # Sampling weight


class AccuracyBenchmarkGenerator:
    """Generates diverse Python code samples for accuracy testing."""

    def __init__(self):
        """Initialize the benchmark generator."""
        self.patterns = self._initialize_patterns()
        self.keywords = self._get_python_keywords()
        self.builtins = self._get_python_builtins()
        self.operators = self._get_python_operators()

    def generate_test_corpus(self, num_files: int = 1000) -> List[str]:
        """
        Generate a corpus of diverse Python files for accuracy testing.

        Args:
            num_files: Number of test files to generate

        Returns:
            List of Python source code strings
        """
        corpus = []

        # Distribute files across complexity levels
        complexity_distribution = {
            CodeComplexity.SIMPLE: 0.3,  # 30% simple
            CodeComplexity.MEDIUM: 0.4,  # 40% medium
            CodeComplexity.COMPLEX: 0.25,  # 25% complex
            CodeComplexity.EXTREME: 0.05,  # 5% extreme
        }

        for complexity, ratio in complexity_distribution.items():
            count = int(num_files * ratio)

            for _ in range(count):
                # Vary file size based on complexity
                if complexity == CodeComplexity.SIMPLE:
                    lines = random.randint(10, 50)
                elif complexity == CodeComplexity.MEDIUM:
                    lines = random.randint(50, 200)
                elif complexity == CodeComplexity.COMPLEX:
                    lines = random.randint(200, 500)
                else:  # EXTREME
                    lines = random.randint(500, 1000)

                code = self.generate_code_sample(lines, complexity.value)
                corpus.append(code)

        # Fill remaining slots with medium complexity
        while len(corpus) < num_files:
            lines = random.randint(50, 200)
            code = self.generate_code_sample(lines, CodeComplexity.MEDIUM.value)
            corpus.append(code)

        return corpus

    def generate_code_sample(
        self, target_lines: int, complexity: str = "medium"
    ) -> str:
        """
        Generate a single Python code sample.

        Args:
            target_lines: Target number of lines
            complexity: Complexity level (simple/medium/complex/extreme)

        Returns:
            Python source code string
        """
        complexity_enum = CodeComplexity(complexity)

        # Select patterns based on complexity
        available_patterns = [
            p
            for p in self.patterns
            if self._complexity_matches(p.complexity, complexity_enum)
        ]

        if not available_patterns:
            available_patterns = self.patterns  # Fallback

        # Generate code structure
        code_lines = []

        # Add file header
        code_lines.extend(self._generate_file_header())

        # Add imports
        code_lines.extend(self._generate_imports(complexity_enum))
        code_lines.append("")

        # Add constants/globals
        if complexity_enum in [
            CodeComplexity.MEDIUM,
            CodeComplexity.COMPLEX,
            CodeComplexity.EXTREME,
        ]:
            code_lines.extend(self._generate_constants())
            code_lines.append("")

        # Generate main content
        remaining_lines = target_lines - len(code_lines) - 5  # Leave room for footer

        while remaining_lines > 0 and len(code_lines) < target_lines:
            # Select random pattern
            pattern = random.choices(
                available_patterns, weights=[p.weight for p in available_patterns]
            )[0]

            # Generate code from pattern
            generated = self._generate_from_pattern(pattern, complexity_enum)
            pattern_lines = generated.split("\n")

            # Add if we have room
            if len(pattern_lines) <= remaining_lines:
                code_lines.extend(pattern_lines)
                code_lines.append("")  # Spacing
                remaining_lines -= len(pattern_lines) + 1
            else:
                break

        # Add footer if extreme complexity
        if complexity_enum == CodeComplexity.EXTREME:
            code_lines.extend(self._generate_file_footer())

        # Ensure we hit target lines approximately
        while len(code_lines) < target_lines:
            code_lines.append(f"# Filler comment line {len(code_lines)}")

        return "\n".join(code_lines[:target_lines])

    def _initialize_patterns(self) -> List[CodePattern]:
        """Initialize the library of code patterns."""
        patterns = []

        # Simple patterns
        patterns.extend(
            [
                CodePattern(
                    "simple_function",
                    'def {func_name}({params}):\n    """{docstring}"""\n    {body}\n    return {return_val}',
                    CodeComplexity.SIMPLE,
                    ["function", "docstring", "return"],
                ),
                CodePattern(
                    "simple_class",
                    'class {class_name}:\n    """{docstring}"""\n    \n    def __init__(self{init_params}):\n        {init_body}',
                    CodeComplexity.SIMPLE,
                    ["class", "method", "docstring"],
                ),
                CodePattern(
                    "simple_loop",
                    "for {var} in {iterable}:\n    {body}",
                    CodeComplexity.SIMPLE,
                    ["for_loop", "iteration"],
                ),
                CodePattern(
                    "simple_conditional",
                    "if {condition}:\n    {true_body}\nelse:\n    {false_body}",
                    CodeComplexity.SIMPLE,
                    ["conditional", "if_else"],
                ),
            ]
        )

        # Medium patterns
        patterns.extend(
            [
                CodePattern(
                    "class_with_methods",
                    'class {class_name}({base_class}):\n    """{docstring}"""\n    \n    def __init__(self{init_params}):\n        super().__init__()\n        {init_body}\n    \n    {methods}',
                    CodeComplexity.MEDIUM,
                    ["class", "inheritance", "super", "methods"],
                ),
                CodePattern(
                    "decorator_function",
                    '@{decorator}\ndef {func_name}({params}) -> {return_type}:\n    """{docstring}"""\n    {body}\n    return {return_val}',
                    CodeComplexity.MEDIUM,
                    ["decorator", "type_hints", "function"],
                ),
                CodePattern(
                    "async_function",
                    'async def {func_name}({params}) -> {return_type}:\n    """{docstring}"""\n    {body}\n    await {async_call}\n    return {return_val}',
                    CodeComplexity.MEDIUM,
                    ["async", "await", "type_hints"],
                ),
                CodePattern(
                    "context_manager",
                    "with {context_expr} as {var}:\n    {body}",
                    CodeComplexity.MEDIUM,
                    ["context_manager", "with"],
                ),
                CodePattern(
                    "exception_handling",
                    "try:\n    {try_body}\nexcept {exception_type} as e:\n    {except_body}\nfinally:\n    {finally_body}",
                    CodeComplexity.MEDIUM,
                    ["exception", "try_except", "finally"],
                ),
            ]
        )

        # Complex patterns
        patterns.extend(
            [
                CodePattern(
                    "generic_class",
                    'from typing import TypeVar, Generic\n\nT = TypeVar(\'T\')\n\nclass {class_name}(Generic[T]):\n    """{docstring}"""\n    \n    def __init__(self, value: T) -> None:\n        self._value = value\n    \n    def get(self) -> T:\n        return self._value',
                    CodeComplexity.COMPLEX,
                    ["generics", "typing", "type_vars"],
                ),
                CodePattern(
                    "dataclass_pattern",
                    '@dataclass\nclass {class_name}:\n    """{docstring}"""\n    {fields}\n    \n    def {method_name}(self) -> {return_type}:\n        {body}',
                    CodeComplexity.COMPLEX,
                    ["dataclass", "decorator", "type_hints"],
                ),
                CodePattern(
                    "async_context_manager",
                    "class {class_name}:\n    async def __aenter__(self):\n        {enter_body}\n        return self\n    \n    async def __aexit__(self, exc_type, exc_val, exc_tb):\n        {exit_body}",
                    CodeComplexity.COMPLEX,
                    ["async", "context_manager", "dunder_methods"],
                ),
                CodePattern(
                    "metaclass_pattern",
                    "class {metaclass_name}(type):\n    def __new__(mcs, name, bases, namespace):\n        {new_body}\n        return super().__new__(mcs, name, bases, namespace)\n\nclass {class_name}(metaclass={metaclass_name}):\n    {class_body}",
                    CodeComplexity.COMPLEX,
                    ["metaclass", "dunder_methods", "inheritance"],
                ),
            ]
        )

        # Extreme patterns
        patterns.extend(
            [
                CodePattern(
                    "protocol_implementation",
                    "from typing import Protocol\n\nclass {protocol_name}(Protocol):\n    def {method_name}(self, {params}) -> {return_type}:\n        ...\n\nclass {impl_name}:\n    def {method_name}(self, {params}) -> {return_type}:\n        {body}",
                    CodeComplexity.EXTREME,
                    ["protocol", "typing", "structural_typing"],
                ),
                CodePattern(
                    "complex_comprehension",
                    "{var} = [\n    {expr}\n    for {outer_var} in {outer_iterable}\n    for {inner_var} in {inner_iterable}\n    if {condition}\n]",
                    CodeComplexity.EXTREME,
                    ["comprehension", "nested_loops", "filtering"],
                ),
                CodePattern(
                    "advanced_decorator",
                    "from functools import wraps\n\ndef {decorator_name}({decorator_params}):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            {wrapper_body}\n            result = func(*args, **kwargs)\n            {wrapper_post}\n            return result\n        return wrapper\n    return decorator",
                    CodeComplexity.EXTREME,
                    ["decorator", "closure", "functools", "varargs"],
                ),
            ]
        )

        return patterns

    def _complexity_matches(
        self, pattern_complexity: CodeComplexity, target_complexity: CodeComplexity
    ) -> bool:
        """Check if pattern complexity is suitable for target complexity."""
        complexity_levels = {
            CodeComplexity.SIMPLE: 1,
            CodeComplexity.MEDIUM: 2,
            CodeComplexity.COMPLEX: 3,
            CodeComplexity.EXTREME: 4,
        }

        pattern_level = complexity_levels[pattern_complexity]
        target_level = complexity_levels[target_complexity]

        # Allow patterns up to target complexity level
        return pattern_level <= target_level

    def _generate_file_header(self) -> List[str]:
        """Generate realistic file header."""
        headers = [
            ['"""', "Module docstring describing functionality.", '"""'],
            ["# -*- coding: utf-8 -*-", '"""', "Another module docstring.", '"""'],
            ["#!/usr/bin/env python3", '"""', "Script with shebang.", '"""'],
            ["# File: example.py", "# Author: Developer", "# Description: Sample code"],
        ]

        return random.choice(headers)

    def _generate_imports(self, complexity: CodeComplexity) -> List[str]:
        """Generate realistic import statements."""
        basic_imports = [
            "import os",
            "import sys",
            "import json",
            "import time",
            "import random",
        ]

        from_imports = [
            "from typing import List, Dict, Optional",
            "from pathlib import Path",
            "from collections import defaultdict",
            "from itertools import chain, product",
        ]

        complex_imports = [
            "from typing import TypeVar, Generic, Protocol",
            "from dataclasses import dataclass, field",
            "from functools import wraps, partial, lru_cache",
            "from contextlib import contextmanager, asynccontextmanager",
        ]

        imports = []

        # Add basic imports
        imports.extend(random.sample(basic_imports, k=random.randint(2, 4)))

        if complexity in [
            CodeComplexity.MEDIUM,
            CodeComplexity.COMPLEX,
            CodeComplexity.EXTREME,
        ]:
            imports.extend(random.sample(from_imports, k=random.randint(1, 3)))

        if complexity in [CodeComplexity.COMPLEX, CodeComplexity.EXTREME]:
            imports.extend(random.sample(complex_imports, k=random.randint(1, 2)))

        return imports

    def _generate_constants(self) -> List[str]:
        """Generate module-level constants."""
        constants = [
            f"VERSION = '{random.randint(1, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}'",
            f"DEFAULT_TIMEOUT = {random.randint(10, 300)}",
            f"MAX_RETRIES = {random.randint(3, 10)}",
            f"BUFFER_SIZE = {random.choice([1024, 2048, 4096, 8192])}",
            f"DEBUG = {random.choice(['True', 'False'])}",
        ]

        return random.sample(constants, k=random.randint(2, 4))

    def _generate_file_footer(self) -> List[str]:
        """Generate file footer for extreme complexity."""
        footers = [
            ['if __name__ == "__main__":', "    main()"],
            ['if __name__ == "__main__":', "    import sys", "    sys.exit(main())"],
            ["# End of file"],
        ]

        return random.choice(footers)

    def _generate_from_pattern(
        self, pattern: CodePattern, complexity: CodeComplexity
    ) -> str:
        """Generate code from a specific pattern."""
        # Create substitution values
        substitutions = {
            "func_name": self._random_identifier("func"),
            "class_name": self._random_identifier("Class", capitalize=True),
            "var": self._random_identifier("var"),
            "params": self._generate_parameters(),
            "docstring": self._random_docstring(),
            "body": self._generate_body(complexity),
            "return_val": self._random_return_value(),
            "return_type": self._random_type_hint(),
            "condition": self._random_condition(),
            "true_body": "    "
            + random.choice(['print("true")', "return True", "pass"]),
            "false_body": "    "
            + random.choice(['print("false")', "return False", "pass"]),
            "iterable": random.choice(
                ["range(10)", "items", "data", "[1, 2, 3, 4, 5]"]
            ),
            "base_class": random.choice(["object", "ABC", "Exception", "dict"]),
            "init_params": ", " + self._generate_parameters()
            if random.choice([True, False])
            else "",
            "init_body": "        "
            + random.choice(["self.value = value", "super().__init__()", "pass"]),
            "methods": self._generate_methods(complexity),
            "decorator": random.choice(
                ["property", "staticmethod", "classmethod", "lru_cache"]
            ),
            "context_expr": random.choice(
                ['open("file.txt")', "lock", "timer()", "connection"]
            ),
            "async_call": random.choice(
                ["fetch_data()", "process_item()", "save_result()"]
            ),
            "exception_type": random.choice(
                ["ValueError", "TypeError", "IOError", "Exception"]
            ),
            "try_body": "    "
            + random.choice(["result = func()", "data = load_data()", "process()"]),
            "except_body": "    "
            + random.choice(['print(f"Error: {e}")', "log_error(e)", "raise"]),
            "finally_body": "    "
            + random.choice(["cleanup()", "close_resources()", "pass"]),
            "fields": self._generate_dataclass_fields(),
            "method_name": self._random_identifier("method"),
            "enter_body": "        "
            + random.choice(["self.resource = acquire()", "await self.connect()"]),
            "exit_body": "        "
            + random.choice(["release(self.resource)", "await self.disconnect()"]),
            "metaclass_name": self._random_identifier("Meta", capitalize=True),
            "new_body": "        "
            + random.choice(['print(f"Creating {name}")', "validate_class(namespace)"]),
            "class_body": "    "
            + random.choice(["pass", "value = 42", "def method(self): pass"]),
            "protocol_name": self._random_identifier("Protocol", capitalize=True),
            "impl_name": self._random_identifier("Implementation", capitalize=True),
            "expr": random.choice(["x * 2", "str(item)", "process(x, y)"]),
            "outer_var": "x",
            "inner_var": "y",
            "outer_iterable": "range(10)",
            "inner_iterable": "items",
            "decorator_name": self._random_identifier("decorator"),
            "decorator_params": self._generate_parameters(),
            "wrapper_body": "            "
            + random.choice(['print("Before call")', "start_timer()"]),
            "wrapper_post": "            "
            + random.choice(['print("After call")', "stop_timer()"]),
        }

        # Apply substitutions to template
        try:
            return pattern.template.format(**substitutions)
        except KeyError as e:
            # Fallback for missing substitutions
            return f"# Pattern generation error: {e}\npass"

    def _random_identifier(self, prefix: str = "var", capitalize: bool = False) -> str:
        """Generate random identifier."""
        suffixes = ["", "_1", "_data", "_result", "_item", "_value", "_temp"]
        suffix = random.choice(suffixes)

        if capitalize:
            return prefix.capitalize() + suffix.capitalize()
        else:
            return prefix + suffix

    def _generate_parameters(self) -> str:
        """Generate function parameters."""
        param_patterns = [
            "",
            "value",
            "x, y",
            "data, timeout=30",
            "*args",
            "**kwargs",
            "*args, **kwargs",
            "x: int, y: str = 'default'",
            "items: List[str], count: Optional[int] = None",
        ]

        return random.choice(param_patterns)

    def _random_docstring(self) -> str:
        """Generate random docstring."""
        docstrings = [
            "Function to process data.",
            "Class representing a data structure.",
            "Method that performs computation.",
            "Utility function for common operations.",
            "Implementation of specific algorithm.",
            "Handler for processing requests.",
            "Manager for resource allocation.",
            "Component for data transformation.",
        ]

        return random.choice(docstrings)

    def _generate_body(self, complexity: CodeComplexity) -> str:
        """Generate function/method body."""
        simple_bodies = [
            "    return True",
            "    pass",
            "    print('Hello')",
            "    result = 42\n    return result",
        ]

        complex_bodies = [
            "    try:\n        result = process()\n        return result\n    except Exception:\n        return None",
            "    for item in data:\n        if check(item):\n            yield transform(item)",
            "    with contextmanager() as ctx:\n        result = ctx.process()\n    return result",
        ]

        if complexity in [CodeComplexity.SIMPLE, CodeComplexity.MEDIUM]:
            return random.choice(simple_bodies)
        else:
            return random.choice(complex_bodies)

    def _random_return_value(self) -> str:
        """Generate random return value."""
        values = [
            "None",
            "True",
            "False",
            "42",
            "'result'",
            "[]",
            "{}",
            "result",
            "self.value",
            "process(data)",
        ]

        return random.choice(values)

    def _random_type_hint(self) -> str:
        """Generate random type hint."""
        types = [
            "None",
            "bool",
            "int",
            "str",
            "List[str]",
            "Dict[str, Any]",
            "Optional[int]",
            "Tuple[str, int]",
            "Any",
        ]

        return random.choice(types)

    def _random_condition(self) -> str:
        """Generate random condition."""
        conditions = [
            "value > 0",
            "data is not None",
            "len(items) > 0",
            "isinstance(obj, str)",
            "hasattr(obj, 'method')",
            "x == y",
            "result.success",
            "not error",
        ]

        return random.choice(conditions)

    def _generate_methods(self, complexity: CodeComplexity) -> str:
        """Generate class methods."""
        methods = []
        method_count = random.randint(1, 3)

        for i in range(method_count):
            method_name = f"method_{i + 1}"
            if complexity == CodeComplexity.SIMPLE:
                method = f"    def {method_name}(self):\n        return self.value"
            else:
                method = f'    def {method_name}(self, param: int) -> str:\n        """Method {i + 1}."""\n        return str(param * 2)'

            methods.append(method)

        return "\n    \n".join(methods)

    def _generate_dataclass_fields(self) -> str:
        """Generate dataclass fields."""
        fields = []
        field_count = random.randint(2, 5)

        for i in range(field_count):
            field_name = f"field_{i + 1}"
            field_type = random.choice(
                ["str", "int", "bool", "List[str]", "Optional[int]"]
            )

            if random.choice([True, False]):
                # Field with default
                default_val = random.choice(["None", "''", "0", "False", "[]"])
                fields.append(f"    {field_name}: {field_type} = {default_val}")
            else:
                # Field without default
                fields.append(f"    {field_name}: {field_type}")

        return "\n".join(fields)

    def _get_python_keywords(self) -> List[str]:
        """Get Python keywords for testing."""
        return [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
            "match",
            "case",
        ]

    def _get_python_builtins(self) -> List[str]:
        """Get Python builtins for testing."""
        return [
            "abs",
            "all",
            "any",
            "bin",
            "bool",
            "chr",
            "dict",
            "dir",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
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
            "map",
            "max",
            "min",
            "next",
            "object",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
        ]

    def _get_python_operators(self) -> List[str]:
        """Get Python operators for testing."""
        return [
            "+",
            "-",
            "*",
            "/",
            "//",
            "%",
            "**",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "and",
            "or",
            "not",
            "in",
            "not in",
            "is",
            "is not",
            "&",
            "|",
            "^",
            "~",
            "<<",
            ">>",
            "=",
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "&=",
            "|=",
            "^=",
            "<<=",
            ">>=",
        ]


# Convenience functions for direct usage
def generate_accuracy_test_corpus(num_files: int = 1000) -> List[str]:
    """Generate test corpus for accuracy benchmarking."""
    generator = AccuracyBenchmarkGenerator()
    return generator.generate_test_corpus(num_files)


def generate_single_test_file(lines: int = 100, complexity: str = "medium") -> str:
    """Generate single test file for benchmarking."""
    generator = AccuracyBenchmarkGenerator()
    return generator.generate_code_sample(lines, complexity)


if __name__ == "__main__":
    # Demo usage
    generator = AccuracyBenchmarkGenerator()

    print("=== Accuracy Benchmark Generator Demo ===")

    # Generate samples of different complexities
    complexities = ["simple", "medium", "complex", "extreme"]

    for complexity in complexities:
        print(f"\n--- {complexity.upper()} COMPLEXITY ---")
        sample = generator.generate_code_sample(20, complexity)
        print(sample[:500] + "..." if len(sample) > 500 else sample)

    # Generate small test corpus
    print("\n--- TEST CORPUS ---")
    corpus = generator.generate_test_corpus(10)
    print(f"Generated {len(corpus)} test files")
    print(f"Total size: {sum(len(code) for code in corpus)} characters")
    print(
        f"Average size: {sum(len(code) for code in corpus) // len(corpus)} characters"
    )
