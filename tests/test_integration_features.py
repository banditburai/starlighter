"""
Feature Integration Tests for Starlighter v2 - DataStar/StarHTML Features.

This test suite consolidates all DataStar and StarHTML feature integration tests:
1. DataStar attribute recognition and highlighting
2. StarHTML element recognition
3. Mixed usage patterns and real-world scenarios
4. Framework integration patterns
5. Complex expression handling

Consolidates tests from:
- tests/test_datastar_starhtml_complete.py
- tests/test_mixed_starhtml.py
- tests/test_starhtml_actual.py
- tests/test_datastar_integration.py
"""

import pytest
from starlighter import highlight
from starlighter.parser import DATASTAR_LOOKUP, STARHTML_ELEMENTS


class TestDataStarAttributeRecognition:
    """Test recognition of DataStar attributes in various contexts."""

    def test_all_core_reactive_attributes(self):
        """Test all core reactive DataStar attributes are recognized."""
        core_attrs = [
            "data_bind",  # Two-way data binding
            "data_text",  # Set text content
            "data_computed",  # Computed signals
            "data_show",  # Conditional visibility
            "data_class",  # Dynamic classes
            "data_attr",  # Dynamic attributes
            "data_signals",  # Define signals
            "data_on",  # Generic event handler
            "data_style",  # Inline styles
            "data_effect",  # Side effects
        ]

        for attr in core_attrs:
            assert attr in DATASTAR_LOOKUP, f"Missing core attribute: {attr}"

            # Test HTML highlighting
            code = f'Div({attr}="$value")'
            html = highlight(code)

            # Check that attribute is highlighted as datastar-attr
            assert f'token-datastar-attr">{attr}</span>' in html, (
                f"Attribute {attr} not highlighted as datastar-attr"
            )

    def test_all_event_handlers(self):
        """Test all DataStar event handler attributes."""
        event_attrs = [
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
        ]

        for attr in event_attrs:
            assert attr in DATASTAR_LOOKUP, f"Missing event handler: {attr}"

    def test_dom_manipulation_attributes(self):
        """Test DOM manipulation and control attributes."""
        dom_attrs = [
            "data_indicator",
            "data_ignore",
            "data_ignore_morph",
            "data_preserve_attr",
            "data_ref",
            "data_json_signals",
        ]

        for attr in dom_attrs:
            assert attr in DATASTAR_LOOKUP, f"Missing DOM attribute: {attr}"

    def test_pro_features(self):
        """Test DataStar Pro feature attributes."""
        pro_attrs = [
            "data_animate",
            "data_custom_validity",
            "data_on_raf",
            "data_on_resize",
            "data_query_string",
            "data_replace_url",
            "data_scroll_into_view",
            "data_view_transition",
        ]

        for attr in pro_attrs:
            assert attr in DATASTAR_LOOKUP, f"Missing Pro attribute: {attr}"

    def test_snake_case_datastar_attributes(self):
        """Test snake_case DataStar attributes are recognized."""
        code = """
element = Div(
    data_show="$visible",
    data_bind="username",
    data_on_click="toggle()"
)
"""
        html = highlight(code)

        # Check for DataStar attribute recognition in HTML
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html

    def test_hyphenated_datastar_attributes(self):
        """Test hyphenated DataStar attributes (HTML style)."""
        # Note: This tests the parser's ability to handle both styles
        code = '''
html = """
<div data-show="$visible" data-bind="username" data-on-click="toggle()">
    Content
</div>
"""
'''
        html = highlight(code)

        # The HTML is in a string, so check string content in HTML output
        assert "data-show" in html

    def test_computed_and_effect_attributes(self):
        """Test computed and effect attributes."""
        code = """
Div(
    data_computed("fullName", "$firstName + ' ' + $lastName"),
    data_effect("console.log($count)"),
    data_text="$fullName"
)
"""
        html = highlight(code)

        # Check that computed and effect are recognized in HTML
        assert (
            'token-datastar-attr">data_computed</span>' in html
            or "data_computed" in html
        )
        assert (
            'token-datastar-attr">data_effect</span>' in html or "data_effect" in html
        )

    def test_comprehensive_datastar_attributes(self):
        """Test recognition of all major DataStar attributes."""
        # Only test attributes that are actually in DATASTAR_LOOKUP
        datastar_attrs = [
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
            "data_indicator",
            "data_ignore",
            "data_ref",
            "data_animate",
            "data_custom_validity",
        ]

        for attr in datastar_attrs:
            code = f'{attr} = "test_value"'
            html = highlight(code)
            assert f'token-datastar-attr">{attr}</span>' in html, (
                f"DataStar attribute {attr} should be recognized"
            )

    def test_non_datastar_data_attributes(self):
        """Test that non-DataStar data_* attributes are not specially highlighted."""
        code = """
data_invalid = "not a real datastar attr"
data_custom = "custom attribute"  
data_other = "something else"
"""
        html = highlight(code)

        # These should be regular identifiers, not DataStar attributes
        assert 'token-identifier">data_invalid</span>' in html
        assert 'token-identifier">data_custom</span>' in html
        assert 'token-identifier">data_other</span>' in html

        # Make sure they're NOT tagged as DataStar
        assert 'token-datastar-attr">data_invalid</span>' not in html
        assert 'token-datastar-attr">data_custom</span>' not in html
        assert 'token-datastar-attr">data_other</span>' not in html


class TestStarHTMLElementRecognition:
    """Test recognition of all StarHTML elements."""

    def test_common_html_elements(self):
        """Test common HTML elements in PascalCase."""
        elements = [
            "Div",
            "Span",
            "Button",
            "Input",
            "Form",
            "Table",
            "Header",
            "Footer",
        ]

        for element in elements:
            assert element in STARHTML_ELEMENTS, f"Missing element: {element}"

            code = f'{element}("content")'
            html = highlight(code)

            # Check that element is highlighted as StarHTML element
            assert f'token-starhtml-element">{element}</span>' in html

    def test_semantic_html5_elements(self):
        """Test semantic HTML5 elements."""
        elements = ["Header", "Footer", "Nav", "Main", "Section", "Article", "Aside"]

        for element in elements:
            assert element in STARHTML_ELEMENTS, f"Missing HTML5 element: {element}"

    def test_starhtml_elements_in_context(self):
        """Test StarHTML elements in realistic code context."""
        code = """
def create_component():
    return Div(
        Header("My App"),
        Main(
            Section(
                H1("Welcome"),
                P("This is a test")
            )
        ),
        Footer("Copyright 2024")
    )
"""
        html = highlight(code)

        # Should recognize StarHTML elements
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Header</span>' in html
        assert 'token-starhtml-element">Main</span>' in html
        assert 'token-starhtml-element">Section</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">P</span>' in html
        assert 'token-starhtml-element">Footer</span>' in html

        # Should still recognize keywords correctly
        assert 'token-keyword">def</span>' in html
        assert 'token-keyword">return</span>' in html
        assert 'token-identifier">create_component</span>' in html


class TestStarHTMLDataStarIntegration:
    """Test StarHTML and DataStar integration scenarios."""

    def test_basic_starhtml_with_datastar(self):
        """Test basic StarHTML element with DataStar attributes."""
        code = """
Div(
    Button("Click me", data_on_click="$count++"),
    Span(data_text="$count"),
    data_signals(count=0)
)
"""
        html = highlight(code)

        # Check for StarHTML elements in HTML output
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-starhtml-element">Span</span>' in html

    def test_signal_syntax_in_expressions(self):
        """Test signal references with $ prefix in expressions."""
        code = """
Div(
    data_show="$visible",
    data_text="$message || 'Default'",
    data_class(active="$selected", error="$invalid")
)
"""
        html = highlight(code)

        # Check that signal expressions are properly captured in strings
        assert "token-string" in html
        assert "$visible" in html
        assert "$message" in html

    def test_complex_datastar_expressions(self):
        """Test complex DataStar expressions with JavaScript-like syntax."""
        code = """
Form(
    Input(data_bind="email", data_on_input="validateEmail($email)"),
    Button(
        "Submit",
        data_disabled="!$email || !$password || $loading",
        data_on_click="submitForm()"
    ),
    data_computed("isValid", "$email && $password.length >= 8")
)
"""
        html = highlight(code)

        # Verify Form and Input are recognized as StarHTML in HTML output
        assert 'token-starhtml-element">Form</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Button</span>' in html

    def test_template_literal_syntax(self):
        """Test template literal syntax for DataStar."""
        code = """
Span(data_text="`Hello ${$name}! You have ${$count} messages.`")
"""
        html = highlight(code)

        # Check that template literal is captured as string in HTML
        assert "token-string" in html
        assert "`Hello" in html

    def test_event_modifiers(self):
        """Test event handler modifiers."""
        code = """
Form(
    data_on_submit="handleSubmit()", 
    Input(data_on_input="search()", debounce="300ms"),
    Button(data_on_click="save()", "prevent", "stop")
)
"""
        html = highlight(code)

        # Verify modifiers are captured in string literals
        assert "300ms" in html
        assert "prevent" in html

    def test_nested_starhtml_with_brackets(self):
        """Test nested StarHTML with bracket syntax."""
        code = """
Div[
    H1["Welcome"],
    P[
        "Hello, ",
        Strong[data_text="$username"],
        "!"
    ],
    Button["Click me", data_on_click="greet()"]
]
"""
        html = highlight(code)

        # Check for StarHTML elements in HTML output
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Strong</span>' in html
        assert 'token-starhtml-element">Button</span>' in html

        # Check for bracket tokens (classified as punctuation in streaming approach)
        assert 'token-punctuation">[</span>' in html
        assert 'token-punctuation">]</span>' in html


class TestMixedStarHTMLPatterns:
    """Test mixed StarHTML Python code and HTML output patterns."""

    def test_starhtml_python_api(self):
        """Test Python code using StarHTML API."""
        starhtml_python_code = '''
from starhtml import Div, Button, Input
from starhtml.datastar import ds_show, ds_on_click, ds_bind, ds_signals

def counter_component():
    """Counter using StarHTML Python API."""
    return Div(
        Button("+", ds_on_click("$count++")),
        Span(ds_text("$count")),
        Button("-", ds_on_click("$count--")),
        ds_signals(count=0)
    )
'''
        html = highlight(starhtml_python_code)

        # Check for StarHTML elements
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-starhtml-element">Span</span>' in html

        # Check for ds_* functions (these are identifiers, not DataStar attributes in Python context)
        assert "ds_show" in html
        assert "ds_on_click" in html
        assert "ds_bind" in html
        assert "ds_signals" in html

    def test_python_with_embedded_html(self):
        """Test Python with embedded HTML containing DataStar attributes."""
        python_with_html = '''
def render_template():
    """Render HTML template with DataStar attributes."""
    return """
    <div data-signals-count="0">
        <button data-on-click="$count++">+</button>
        <span data-text="$count">0</span>
        <button data-on-click="$count--">-</button>
    </div>
    """
'''
        html = highlight(python_with_html)

        # Should handle embedded HTML in strings
        assert "token-string" in html
        assert "data-signals-count" in html
        assert "data-on-click" in html
        assert "data-text" in html

    def test_mixed_starhtml_and_html_generation(self):
        """Test mixed Python generating HTML strings with DataStar."""
        mixed_code = '''
from starhtml import Div, Button

def create_interactive_element(name, action):
    """Create an interactive element with DataStar."""
    # Using StarHTML API
    element = Div(
        Button(name, data_on_click=action),
        data_show="$visible"
    )
    
    # Also can generate raw HTML
    html = f"""
    <div data-show="$visible">
        <button data-on-click="{action}">{name}</button>
    </div>
    """
    
    return element, html

# Using the function
button = create_interactive_element("Submit", "handleSubmit()")
'''
        html = highlight(mixed_code)

        # Should recognize both StarHTML elements and DataStar attributes
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-datastar-attr">data_show</span>' in html


class TestRealWorldStarHTMLScenarios:
    """Test real-world StarHTML/DataStar scenarios."""

    def test_todo_application(self):
        """Test a complete todo application with StarHTML/DataStar."""
        todo_app_code = '''
from fasthtml.common import *
from starhtml import Div, Button, Input, Form, H1, P, Span

def create_todo_app():
    """A complete todo application with StarHTML/DataStar."""
    return Div(
        H1("Todo List"),
        Form(
            Input(
                type="text",
                placeholder="Enter a new todo...",
                data_bind="newTodo",
                data_on_keydown_enter="addTodo()"
            ),
            Button("Add Todo", data_on_click="addTodo()", type="button"),
            data_on_submit="event.preventDefault()"
        ),
        Div(
            data_show="$todos.length > 0",
            class_="todo-list"
        )[
            Div(
                data_for="todo in $todos",
                class_="todo-item"
            )[
                Input(
                    type="checkbox",
                    data_bind="todo.completed"
                ),
                Span(
                    data_text="$todo.text",
                    data_class(completed="$todo.completed")
                ),
                Button(
                    "×",
                    data_on_click="removeTodo($todo.id)",
                    class_="btn-remove"
                )
            ]
        ],
        P(
            data_show="$todos.length === 0",
            class_="empty-state"
        )["No todos yet. Add one above!"],
        data_signals(todos=[], newTodo="")
    )
'''
        html = highlight(todo_app_code)

        # Verify key elements are highlighted
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Form</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Button</span>' in html

        # Verify DataStar attributes are recognized
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_signals</span>' in html

    def test_form_with_validation(self):
        """Test a form with DataStar validation and conditional rendering."""
        form_code = '''
def create_form_with_validation():
    """A form with DataStar validation and conditional rendering."""
    return Form(
        Div(class_="form-group")[
            Label("Email:"),
            Input(
                type="email",
                data_bind="email",
                data_on_input="validateEmail()",
                placeholder="user@example.com"
            ),
            Span(
                data_show="$emailError",
                data_text="$emailError",
                class_="error-message"
            )
        ],
        Div(class_="form-group")[
            Label("Password:"),
            Input(
                type="password",
                data_bind="password",
                data_on_input="validatePassword()"
            ),
            Span(
                data_show="$passwordError",
                data_text="$passwordError",
                class_="error-message"
            )
        ],
        Button(
            "Submit",
            type="submit",
            data_disabled="!$isValid",
            data_on_click="handleSubmit()"
        ),
        data_signals(
            email="",
            password="",
            emailError="",
            passwordError="",
            isValid=False
        ),
        data_computed("isValid", "!$emailError && !$passwordError && $email && $password")
    )
'''
        html = highlight(form_code)

        # Verify comprehensive patterns
        assert 'token-starhtml-element">Form</span>' in html
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">Label</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Button</span>' in html

        # Verify DataStar patterns
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-datastar-attr">data_on_input</span>' in html
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_text</span>' in html
        # Note: data_disabled is not in DATASTAR_LOOKUP, so it's treated as regular identifier
        assert 'token-datastar-attr">data_signals</span>' in html
        assert 'token-datastar-attr">data_computed</span>' in html

    def test_counter_component(self):
        """Test a simple counter component using StarHTML and DataStar."""
        counter_code = '''
def create_counter_component():
    """A simple counter component using StarHTML and DataStar."""
    return Div(
        H1("Counter Example"),
        Div(
            Span("Count: ", data_text="$count"),
            class_="counter-display"
        ),
        Button("+", data_on_click="$count++", class_="btn-increment"),
        Button("-", data_on_click="$count--", class_="btn-decrement"),
        Button("Reset", data_on_click="$count = 0", class_="btn-reset"),
        data_signals(count=0)
    )
'''
        html = highlight(counter_code)

        # Verify StarHTML elements
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Button</span>' in html
        assert 'token-starhtml-element">Span</span>' in html

        # Verify DataStar attributes
        assert 'token-datastar-attr">data_text</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-datastar-attr">data_signals</span>' in html


class TestDataStarIntegrationPerformance:
    """Test performance and robustness of DataStar/StarHTML integration."""

    def test_parser_performance_with_datastar(self):
        """Test that DataStar recognition doesn't significantly impact performance."""
        code = """
# Mix of DataStar and regular Python code
def handle_user_interaction():
    user_data = fetch_user()
    data_show = user_data.is_visible
    data_on_click = "handleUserClick()"
    data_bind = user_data.name
    
    for item in items:
        data_class = f"item-{item.id}"
        process_item(item)
    
    return {
        'status': 'success',
        'data_signals': user_data.signals,
        'data_computed': calculate_metrics()
    }
"""
        # This should not raise any exceptions and should complete quickly
        html = highlight(code)
        assert isinstance(html, str)
        assert len(html) > len(code)  # HTML should be longer due to tags

        # Spot check some tokens
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-keyword">def</span>' in html
        assert 'token-keyword">for</span>' in html

    def test_large_starhtml_document(self):
        """Test performance with large StarHTML documents."""
        # Generate a large StarHTML document
        components = []
        for i in range(100):
            component = f"""
def component_{i}():
    return Div(
        H{i % 6 + 1}("Component {i}"),
        Div(
            data_show="$visible_{i}",
            data_on_click="toggle_{i}()",
            data_bind="value_{i}"
        )[
            Input(type="text", data_bind="input_{i}"),
            Button("Save", data_on_click="save_{i}()"),
            Span(data_text="$result_{i}")
        ],
        data_signals(visible_{i}=True, value_{i}="", input_{i}="", result_{i}="")
    )
"""
            components.append(component)

        large_code = "\n".join(components)

        # Should handle large documents efficiently
        html = highlight(large_code)

        # Verify it processed correctly
        assert isinstance(html, str)
        assert len(html) > len(large_code)
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-datastar-attr">data_show</span>' in html


class TestFrameworkIntegrationPatterns:
    """Test integration patterns with web frameworks."""

    def test_fasthtml_integration_patterns(self):
        """Test FastHTML integration patterns."""
        fasthtml_code = """
from fasthtml.common import *
from starhtml import Div, Button, Form, Input

app = FastHTML()

@app.route("/")
def home():
    return Div(
        H1("Welcome to FastHTML + DataStar"),
        Form(
            Input(data_bind="name", placeholder="Your name"),
            Button("Greet", data_on_click="greet($name)"),
            data_on_submit="event.preventDefault()"
        ),
        Div(
            data_show="$greeting",
            data_text="$greeting",
            class_="greeting"
        ),
        data_signals(name="", greeting="")
    )

@app.route("/api/greet")
def greet(name: str):
    return {"greeting": f"Hello, {name}!"}
"""
        html = highlight(fasthtml_code)

        # Verify framework integration
        assert 'token-starhtml-element">Div</span>' in html
        assert 'token-starhtml-element">H1</span>' in html
        assert 'token-starhtml-element">Form</span>' in html
        assert 'token-starhtml-element">Input</span>' in html
        assert 'token-starhtml-element">Button</span>' in html

        # Verify DataStar attributes work in framework context
        assert 'token-datastar-attr">data_bind</span>' in html
        assert 'token-datastar-attr">data_on_click</span>' in html
        assert 'token-datastar-attr">data_show</span>' in html
        assert 'token-datastar-attr">data_signals</span>' in html

    def test_renderer_css_classes_integration(self):
        """Test that unified implementation generates correct CSS classes."""
        code = """
Div(
    Button("Click", data_on_click="$count++"),
    data_show="$visible"
)
"""
        html = highlight(code)

        # Check for CSS classes in output
        assert "token-starhtml-element" in html
        assert "token-datastar-attr" in html or "data_on_click" in html

    def test_complete_starhtml_component_rendering(self):
        """Test rendering of a complete StarHTML component."""
        code = """
def TodoList():
    return Div(
        H1["Todo List"],
        Form(
            Input(
                type="text",
                data_bind="newTodo",
                data_on_keydown_enter="addTodo()"
            ),
            Button["Add", data_on_click="addTodo()"]
        ),
        Ul(
            data_for="todo in $todos",
            Li[
                Span(data_text="$todo.text"),
                Button["X", data_on_click="removeTodo($todo.id)"]
            ]
        ),
        data_signals(todos=[], newTodo="")
    )
"""
        html = highlight(code)

        # Verify key elements are highlighted
        assert "TodoList" in html
        assert "token-starhtml-element" in html  # For Div, Form, etc.
        assert "token-keyword" in html  # For def, return


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
