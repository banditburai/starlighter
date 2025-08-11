"""
Sample Python code for integration testing.

This module contains typical Python patterns and constructs that should be
properly highlighted by starlighter. These samples represent common code
patterns found in production Python applications.
"""


# Basic function definitions
def simple_function():
    """Simple function with docstring."""
    return 42


def function_with_params(x, y=10, *args, **kwargs):
    """Function with various parameter types."""
    result = x + y
    for arg in args:
        result += arg
    return result


# Class definitions
class SampleClass:
    """A sample class demonstrating various Python features."""

    class_variable = "shared"

    def __init__(self, name: str, value: int = 0):
        """Initialize with type hints."""
        self.name = name
        self.value = value
        self._private = "private"

    @property
    def formatted_name(self) -> str:
        """Property with decorator and return type hint."""
        return f"Name: {self.name}"

    @staticmethod
    def utility_function(data: list) -> bool:
        """Static method example."""
        return len(data) > 0

    @classmethod
    def from_dict(cls, data: dict):
        """Class method constructor."""
        return cls(data.get("name", ""), data.get("value", 0))


# Inheritance example
class ChildClass(SampleClass):
    """Child class demonstrating inheritance."""

    def __init__(self, name: str, value: int, extra: str = ""):
        super().__init__(name, value)
        self.extra = extra

    def override_method(self):
        """Override parent method."""
        return f"{super().formatted_name} - {self.extra}"


# String variations
SINGLE_QUOTE_STRING = "Single quoted string"
DOUBLE_QUOTE_STRING = "Double quoted string"
TRIPLE_QUOTE_STRING = """
Triple quoted string
with multiple lines
and "quotes" inside
"""

# F-strings and formatting
name = "World"
count = 42
f_string_simple = f"Hello {name}!"
f_string_complex = f"Count: {count:,} items at {count * 1.5:.2f} each"
f_string_nested = f"Result: {f'nested {name}'}"

# Raw and byte strings
raw_string = r"Raw string with \n no escape sequences"
byte_string = b"Byte string content"
rf_string = rf"Raw f-string with {name} and \n no escapes"

# Number literals
integer_decimal = 42
integer_hex = 0xFF
integer_octal = 0o777
integer_binary = 0b1010
float_simple = 3.14
float_scientific = 1.23e-4
float_large = 1.5e6
complex_number = 3 + 4j
underscored_numbers = 1_000_000

# Boolean and None
boolean_true = True
boolean_false = False
none_value = None

# Collections
list_example = [1, 2, 3, "four", 5.0]
tuple_example = (1, "two", 3.0)
dict_example = {"key1": "value1", 2: "value2", "nested": {"inner": True}}
set_example = {1, 2, 3, 4, 5}

# Comprehensions
list_comp = [x * 2 for x in range(10) if x % 2 == 0]
dict_comp = {str(i): i**2 for i in range(5)}
set_comp = {x for x in "hello world" if x.isalpha()}
gen_comp = (x * 3 for x in range(5))


# Control flow
def control_flow_example(value):
    """Example of various control flow constructs."""
    if value > 10:
        return "large"
    elif value > 5:
        return "medium"
    else:
        return "small"

    # Loop examples
    for i in range(5):
        if i == 3:
            continue
        if i == 4:
            break
        print(i)

    while value > 0:
        value -= 1
        if value == 5:
            pass  # Do nothing

    # Exception handling
    try:
        result = 10 / value
    except ZeroDivisionError:
        result = float("inf")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        print("Cleanup")

    return result


# Context managers
def context_manager_example():
    """Context manager usage."""
    with open("file.txt", "r") as f:
        content = f.read()

    with open("output.txt", "w") as out, open("input.txt", "r") as inp:
        out.write(inp.read())

    return content


# Decorators
def my_decorator(func):
    """Custom decorator function."""

    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result

    return wrapper


@my_decorator
def decorated_function():
    """Function with custom decorator."""
    return "decorated result"


# Multiple decorators
@staticmethod
@property
def multiple_decorators():
    """Function with multiple decorators."""
    return "multi-decorated"


# Lambda functions
def square(x):
    return x**2
def add_numbers(x, y):
    return x + y
def filter_even(nums):
    return [n for n in nums if n % 2 == 0]

# Built-in functions
builtin_examples = [
    len([1, 2, 3]),
    max(1, 2, 3, 4, 5),
    min([10, 5, 8, 2]),
    sum(range(10)),
    abs(-42),
    round(3.14159, 2),
    sorted([3, 1, 4, 1, 5]),
    enumerate(["a", "b", "c"]),
    zip([1, 2, 3], ["a", "b", "c"]),
    map(str.upper, ["hello", "world"]),
    filter(None, [0, 1, "", "text", False, True]),
]

# Import statements
from typing import List, Dict, Optional, Union, Tuple


# Type hints and annotations
def type_annotated_function(
    text: str,
    numbers: List[int],
    mapping: Dict[str, Union[int, str]],
    optional_value: Optional[float] = None,
) -> Tuple[str, int]:
    """Function with comprehensive type annotations."""
    processed = text.upper()
    total = sum(numbers)
    return processed, total


# Async/await patterns
import asyncio


async def async_function():
    """Async function example."""
    await asyncio.sleep(1)
    return "async result"


async def async_comprehension():
    """Async comprehension example."""
    results = [await async_function() for _ in range(3)]
    return results


# Generator functions
def fibonacci_generator():
    """Generator function example."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def generator_with_send():
    """Generator with send() functionality."""
    value = yield
    while value is not None:
        yield f"Received: {value}"
        value = yield


# Metaclass example (advanced)
class MetaClass(type):
    """Metaclass example."""

    def __new__(cls, name, bases, dct):
        dct["class_id"] = f"meta_{name}"
        return super().__new__(cls, name, bases, dct)


class MetaExample(metaclass=MetaClass):
    """Class using metaclass."""

    pass


# Constants and module-level variables
VERSION = "1.0.0"
DEBUG = True
MAX_CONNECTIONS = 100
ALLOWED_EXTENSIONS = {".txt", ".py", ".json"}
CONFIG = {"host": "localhost", "port": 8080, "debug": DEBUG}

# Comments and documentation
# This is a single-line comment
"""
This is a module-level docstring
that spans multiple lines and contains
various formatting elements.
"""


def example_with_comments():
    # Comment before variable
    x = 42  # Inline comment

    # Multi-line comment block
    # explaining complex logic
    # with multiple lines
    y = x * 2

    return y  # Return comment


# Edge cases and special characters
unicode_string = "Unicode: 你好世界 🌍 café naïve résumé"
escaped_chars = "String with\nNewlines\tTabs\r\nAndMore"
quote_variations = 'String with "double" quotes'
double_quote_variations = "String with 'single' quotes"

# Main execution guard
if __name__ == "__main__":
    print("Sample code executed directly")
    result = control_flow_example(7)
    print(f"Result: {result}")
