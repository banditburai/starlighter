"""
Complex Python code examples for integration testing.

This module contains advanced Python patterns, edge cases, and real-world
code samples that stress-test the starlighter syntax highlighter.
Includes patterns from popular open-source projects.
"""

# Complex nested structures
DEEPLY_NESTED_DICT = {
    "level1": {
        "level2": {
            "level3": {
                "level4": {"data": [{"id": i, "value": f"item_{i}"} for i in range(10)]}
            }
        }
    }
}


# Multiple decorators with arguments
def parametrized_decorator(param1, param2="default"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Decorating with {param1}, {param2}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


@parametrized_decorator("test", param2="value")
@staticmethod
def complex_decorated_function():
    """Function with complex decorators."""
    return "result"


# Django-style model (adapted from Django source)
class User:
    """
    Simplified Django-style user model.
    Adapted from Django's AbstractUser model.
    """

    # Field definitions with defaults
    username = None  # CharField equivalent
    email = ""  # EmailField equivalent
    first_name = ""  # CharField equivalent
    last_name = ""  # CharField equivalent
    is_staff = False  # BooleanField equivalent
    is_active = True  # BooleanField equivalent
    date_joined = None  # DateTimeField equivalent

    def __init__(self, username, email="", **kwargs):
        self.username = username
        self.email = email
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_full_name(self):
        """Return the first_name plus the last_name, with a space in between."""
        full_name = f"{self.first_name} {self.last_name}"
        return full_name.strip()

    def get_short_name(self):
        """Return the short name for the user."""
        return self.first_name

    def email_user(self, subject, message, from_email=None, **kwargs):
        """Send an email to this user."""
        # In real Django, this would send actual email
        print(f"Sending email to {self.email}: {subject}")
        return True

    def __str__(self):
        return self.username

    def __repr__(self):
        return f"<User: {self.username}>"


# Flask-style application structure (adapted from Flask examples)
class FlaskApp:
    """
    Simplified Flask application structure.
    Adapted from Flask framework patterns.
    """

    def __init__(self, import_name):
        self.import_name = import_name
        self.routes = {}
        self.config = {}
        self.before_request_funcs = []
        self.after_request_funcs = []

    def route(self, rule, **options):
        """Decorator to register a route."""

        def decorator(f):
            endpoint = options.pop("endpoint", None)
            if endpoint is None:
                endpoint = f.__name__
            self.routes[rule] = {
                "function": f,
                "endpoint": endpoint,
                "options": options,
            }
            return f

        return decorator

    def before_request(self, f):
        """Register a function to run before each request."""
        self.before_request_funcs.append(f)
        return f

    def after_request(self, f):
        """Register a function to run after each request."""
        self.after_request_funcs.append(f)
        return f

    def run(self, host="127.0.0.1", port=5000, debug=False):
        """Run the application."""
        print(f"Running on http://{host}:{port} (debug={debug})")


app = FlaskApp(__name__)


@app.route("/")
def index():
    """Home page route."""
    return "Hello, World!"


@app.route("/user/<username>")
def show_user_profile(username):
    """Show user profile."""
    return f"User: {username}"


# Data Science / Pandas-style operations
class DataAnalyzer:
    """
    Simplified data analysis operations.
    Patterns common in data science and pandas usage.
    """

    def __init__(self, data):
        self.data = data
        self.processed = False

    def filter_data(self, condition_func):
        """Filter data using a condition function."""
        if not callable(condition_func):
            raise ValueError("Condition must be callable")

        filtered = [item for item in self.data if condition_func(item)]
        return DataAnalyzer(filtered)

    def transform(self, transform_func, *args, **kwargs):
        """Transform data using a function."""
        transformed = [transform_func(item, *args, **kwargs) for item in self.data]
        analyzer = DataAnalyzer(transformed)
        analyzer.processed = True
        return analyzer

    def aggregate(self, agg_func=sum):
        """Aggregate data using aggregation function."""
        if not self.data:
            return None

        # Handle different aggregation types
        if agg_func == sum:
            return sum(x for x in self.data if isinstance(x, (int, float)))
        elif agg_func == len:
            return len(self.data)
        elif callable(agg_func):
            return agg_func(self.data)
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")

    def groupby(self, key_func):
        """Group data by key function."""
        from collections import defaultdict

        groups = defaultdict(list)
        for item in self.data:
            key = key_func(item)
            groups[key].append(item)

        return {k: DataAnalyzer(v) for k, v in groups.items()}


# Complex string formatting and templates
def format_report(data, template_type="default"):
    """
    Complex string formatting with multiple f-strings.
    Common pattern in reporting and template generation.
    """

    timestamp = "2024-01-01 12:00:00"
    total_items = len(data)
    avg_value = sum(data) / len(data) if data else 0

    if template_type == "detailed":
        return f"""
        ╔═══════════════════════════════════════╗
        ║              DATA REPORT              ║
        ╠═══════════════════════════════════════╣
        ║ Generated: {timestamp:>22} ║
        ║ Items:     {total_items:>22} ║  
        ║ Average:   {avg_value:>22.2f} ║
        ║ Range:     {min(data) if data else 0:>10} - {max(data) if data else 0:<10} ║
        ╚═══════════════════════════════════════╝
        
        Data Summary:
        {chr(10).join(f"  [{i:03d}] {value:>10.2f}" for i, value in enumerate(data[:10]))}
        {f"  ... and {len(data) - 10} more items" if len(data) > 10 else ""}
        """
    else:
        return f"Report: {total_items} items, avg={avg_value:.2f}, generated at {timestamp}"


# Async/await complex patterns
import asyncio
from typing import AsyncIterator


async def async_generator() -> AsyncIterator[int]:
    """Async generator function."""
    for i in range(10):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i


async def async_context_manager():
    """Async context manager example."""

    class AsyncResource:
        async def __aenter__(self):
            print("Acquiring async resource")
            await asyncio.sleep(0.1)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print("Releasing async resource")
            await asyncio.sleep(0.1)

    async with AsyncResource():
        await asyncio.sleep(0.5)
        return "async work complete"


async def async_comprehensions():
    """Complex async comprehensions."""
    # Async list comprehension
    async_results = [await async_operation(x) for x in range(5) if x % 2 == 0]

    # Async generator expression
    async_gen = (
        await async_operation(x)
        async for x in async_generator()
        if await async_condition(x)
    )

    return async_results, async_gen


async def async_operation(value):
    """Simulate async operation."""
    await asyncio.sleep(0.01)
    return value * 2


async def async_condition(value):
    """Simulate async condition check."""
    await asyncio.sleep(0.01)
    return value > 5


# Complex class hierarchies and descriptors
class Descriptor:
    """Property descriptor example."""

    def __init__(self, name=None):
        self.name = name
        self.private_name = f"_{name}" if name else "_value"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected number, got {type(value)}")
        setattr(obj, self.private_name, value)

    def __delete__(self, obj):
        delattr(obj, self.private_name)


class ValidatedClass:
    """Class using descriptors for validation."""

    value = Descriptor("value")

    def __init__(self, initial_value=0):
        self.value = initial_value


# Metaclass with complex behavior
class SingletonMeta(type):
    """Singleton metaclass implementation."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __repr__(cls):
        return f"<Singleton class '{cls.__name__}'>"


class DatabaseConnection(metaclass=SingletonMeta):
    """Database connection using singleton pattern."""

    def __init__(self):
        self.connected = False
        self.host = "localhost"
        self.port = 5432

    def connect(self):
        """Connect to database."""
        if not self.connected:
            print(f"Connecting to {self.host}:{self.port}")
            self.connected = True
        return self.connected

    def disconnect(self):
        """Disconnect from database."""
        if self.connected:
            print("Disconnecting from database")
            self.connected = False


# Complex exception handling patterns
class CustomException(Exception):
    """Custom exception with context."""

    def __init__(self, message, error_code=None, context=None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

    def __str__(self):
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


def complex_error_handling(operation_type, data=None):
    """Complex nested exception handling."""

    errors_encountered = []

    try:
        if operation_type == "risky":
            try:
                result = risky_operation(data)
            except ValueError as ve:
                errors_encountered.append(f"Value error: {ve}")
                # Try alternative approach
                try:
                    result = alternative_operation(data)
                except Exception as ae:
                    errors_encountered.append(f"Alternative failed: {ae}")
                    raise CustomException(
                        "All operations failed",
                        error_code="OP_FAILED",
                        context={
                            "errors": errors_encountered,
                            "data_type": type(data).__name__,
                        },
                    )
            except CustomException:
                raise  # Re-raise custom exceptions
            except Exception as e:
                raise CustomException(
                    f"Unexpected error in {operation_type}",
                    error_code="UNEXPECTED",
                    context={"original_error": str(e)},
                )
        else:
            result = safe_operation(data)

    except CustomException as ce:
        print(f"Custom error handled: {ce}")
        result = {"error": True, "message": str(ce)}
    except Exception as e:
        print(f"Unhandled error: {e}")
        result = {"error": True, "message": "Unknown error"}
    finally:
        if errors_encountered:
            print(f"Errors during processing: {len(errors_encountered)}")

    return result


def risky_operation(data):
    """Operation that might fail."""
    if not data:
        raise ValueError("No data provided")
    if len(data) > 100:
        raise CustomException("Data too large", error_code="DATA_SIZE")
    return sum(data)


def alternative_operation(data):
    """Alternative operation."""
    return len(data) if data else 0


def safe_operation(data):
    """Safe operation."""
    return data or []


# Complex comprehensions and generators
def complex_data_processing():
    """Complex data processing with comprehensions."""

    # Nested comprehensions
    matrix = [[i * j for j in range(1, 6)] for i in range(1, 6)]

    # Complex filtering with multiple conditions
    filtered_data = [
        item
        for sublist in matrix
        for item in sublist
        if item % 2 == 0 and item > 5 and item < 20
    ]

    # Dictionary comprehension with conditional logic
    processed_dict = {
        f"item_{i}": {
            "value": val,
            "category": "high" if val > 10 else "low",
            "processed": True,
        }
        for i, val in enumerate(filtered_data)
        if val is not None
    }

    # Generator with complex yield logic
    def complex_generator():
        for key, data in processed_dict.items():
            if data["category"] == "high":
                yield f"HIGH: {key} = {data['value']}"
            else:
                yield f"low: {key} = {data['value']}"

            # Conditional yielding
            if data["value"] % 3 == 0:
                yield f"  -> Divisible by 3: {data['value']}"

    return list(complex_generator())


# Real-world pattern: Configuration management
class ConfigManager:
    """Configuration management with environment variable support."""

    def __init__(self, config_file=None):
        self.config = {}
        self.env_prefix = "APP_"
        if config_file:
            self.load_from_file(config_file)
        self.load_from_env()

    def load_from_file(self, filename):
        """Load configuration from JSON file."""
        try:
            import json

            with open(filename, "r") as f:
                file_config = json.load(f)
                self.config.update(file_config)
        except FileNotFoundError:
            print(f"Config file {filename} not found")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")

    def load_from_env(self):
        """Load configuration from environment variables."""
        import os

        env_config = {
            key[len(self.env_prefix) :].lower(): value
            for key, value in os.environ.items()
            if key.startswith(self.env_prefix)
        }

        # Type conversion for common patterns
        for key, value in env_config.items():
            if value.lower() in ("true", "false"):
                env_config[key] = value.lower() == "true"
            elif value.isdigit():
                env_config[key] = int(value)
            elif self._is_float(value):
                env_config[key] = float(value)

        self.config.update(env_config)

    def _is_float(self, value):
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get(self, key, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config


# Edge case: Very long lines and complex expressions
VERY_LONG_CONFIGURATION = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp",
        "user": "admin",
        "password": "secret123",
    },
    "cache": {
        "host": "redis.example.com",
        "port": 6379,
        "ttl": 3600,
        "prefix": "myapp:",
        "max_connections": 100,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": ["console", "file"],
    },
    "features": {
        "feature_a": True,
        "feature_b": False,
        "feature_c": True,
        "experimental": {"new_ui": False, "beta_api": True},
    },
    "external_services": {
        "api_key": "abc123xyz789",
        "base_url": "https://api.example.com/v1",
        "timeout": 30,
        "retries": 3,
    },
}

# Complex lambda and functional programming patterns
from functools import reduce, partial, wraps

# Functional composition
def compose(f, g):
    return lambda x: f(g(x))
def pipe(x, *fns):
    return reduce(lambda acc, fn: fn(acc), fns, x)

# Complex lambda expressions
def complex_lambda(data):
    return pipe(
    data,
    partial(filter, lambda x: x % 2 == 0),
    partial(map, lambda x: x**2),
    partial(filter, lambda x: x > 10),
    list,
    partial(sorted, reverse=True),
)


# Higher-order functions
def memoize(func):
    """Memoization decorator with cache statistics."""
    cache = {}
    stats = {"hits": 0, "misses": 0}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))

        if key in cache:
            stats["hits"] += 1
            return cache[key]
        else:
            stats["misses"] += 1
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    wrapper.cache_stats = stats
    wrapper.cache_clear = cache.clear
    return wrapper


@memoize
def fibonacci(n):
    """Memoized fibonacci calculation."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Test execution patterns
if __name__ == "__main__":
    # Complex initialization and testing
    print("Running complex examples...")

    # Test data processing
    analyzer = DataAnalyzer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    filtered = analyzer.filter_data(lambda x: x % 2 == 0)
    transformed = filtered.transform(lambda x: x**2)
    result = transformed.aggregate()

    print(f"Data processing result: {result}")

    # Test async operations (would need actual async runtime)
    # asyncio.run(async_context_manager())

    # Test configuration
    config = ConfigManager()
    config["test_key"] = "test_value"

    print("Complex examples completed successfully")
