"""Templates for generating prompts or formatting code."""

# Function definition templates
FUNCTION_TEMPLATES = {
    "basic": "def {name}({params}):",
    "with_example": "def {name}({params}): # Example: {example}",
    "with_docstring": 'def {name}({params}):\n    """{docstring}"""',
    "with_types": "def {name}({params}) -> {return_type}:",
}

# Comment templates for examples
EXAMPLE_TEMPLATES = {
    "inline": "# Example: {function_name}({input}) -> {output}",
    "multiline": "# Example:\n# >>> {function_name}({input})\n# {output}",
}

# Docstring templates
DOCSTRING_TEMPLATES = {
    "simple": "{description}",
    "detailed": """{description}
    
    Args:
        {args}
    
    Returns:
        {returns}
    """,
}


def create_prompt(name, params, template="basic", **kwargs):
    """Create a function prompt from template."""
    template_str = FUNCTION_TEMPLATES.get(template, FUNCTION_TEMPLATES["basic"])
    return template_str.format(name=name, params=params, **kwargs)


def format_example(function_name, input_val, output_val, style="inline"):
    """Format an example comment."""
    template_str = EXAMPLE_TEMPLATES.get(style, EXAMPLE_TEMPLATES["inline"])
    return template_str.format(
        function_name=function_name,
        input=input_val,
        output=output_val
    )
